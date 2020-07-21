from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class VAL_Dataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        t1 = os.listdir(folder)
        t2 = [t for t in t1 if t[:4] == 'mask']
        t3 = [t for t in t1 if t[:3] == 'img']
        t2.sort()
        t2 = [os.path.join(folder,s) for s in t2]
        t3.sort()
        t3 = [os.path.join(folder,s) for s in t3]
        self.mask_paths = t2
        self.img_paths = t3
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx]).convert('L')
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        return img, mask