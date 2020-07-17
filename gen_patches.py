from PIL import Image
from PIL import ImageStat
import os
import glob
from removal import Mask

'''
Usage Example:

# to split all images in drsden folder and save it to another folder:
generate_patches('Dresden/Dresden_JPEG', 'Dresden_patches', 256, 256)

# to see the example of using DresdenDataset, one can refer to the 
# usage example in transforms_enhance.py
'''

# split an image into several blocks and save to a directory
def _split_one_img(img_filepath, height, width, save_dir):
    im = Image.open(img_filepath)
    img_name = os.path.basename(img_filepath)
    img_name = os.path.splitext(img_name)[0]
    imgwidth, imgheight = im.size
    k = 0
    for i in range(0,imgheight,height):
        if i + height > imgheight: continue
        for j in range(0,imgwidth,width):
            if j + width > imgwidth: continue
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            s = ImageStat.Stat(a).stddev
            if s[0] < 32 and s[1] < 32 and s[2] < 32: continue
            a.save(os.path.join(save_dir, f'{img_name}_{k}.png'))
            k += 1

# split all images in a directory
def generate_patches(img_dir, save_dir, height, width):
    '''
    img_dir: directory of images to generate patches
    save_dir: directory to save patches
    height: height of patch size
    width: width of patch size
    '''
    for img_filepath in glob.glob(os.path.join(img_dir, '*')):
        _split_one_img(img_filepath, height, width, save_dir)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from random  import randrange

# torch dataset, images in img_dir are already patches
class DresdenDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, '*'))
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.transform is not None:
            img, masking = self.transform(img)
            masking = self.to_tensor(masking)
        img = self.to_tensor(img)
        if self.transform is not None:
            return img, masking
        else:
            return img

# torch dataset, crop the image when calling __getitem__
class DresdenDataset(Dataset):
    def __init__(self, img_dir, height, width, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, '*'))
        self.height = height
        self.width = width
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        a, s = self._random_crop(img)
        while s[0] < 32 and s[1] < 32 and s[2] < 32:
            a, s = self._random_crop(img)
        img = a
        if self.transform is not None:
            img, masking = self.transform(img)
            masking = self.to_tensor(masking)
        img = self.to_tensor(img)
        if self.transform is not None:
            return img, masking
        else:
            return img
    def _random_crop(self, img):
        img_w, img_h = img.size
        x = randrange(0, img_w - self.width)
        y = randrange(0, img_h - self.height)
        box = (x, y, x + self.width, y + self.height)
        a = img.crop(box)
        s = ImageStat.Stat(a).stddev
        return a, s