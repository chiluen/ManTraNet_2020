from PIL import Image
from PIL import ImageStat
import os
import glob

'''
Example:
generate_patches('Dresden/Dresden_JPEG', 'Dresden_patches', 256, 256)
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
            if s[0] < 32 or s[1] < 32 or s[2] < 32: continue
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