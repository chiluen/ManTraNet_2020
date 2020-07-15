import os
import random
import cv2
from PIL import Image, ImageOps
from skimage.util import random_noise
import numpy as np
from removal import Mask

class Enhance():
    def __init__(self, man_list, mask_dir):
        self.man_list = man_list
        self.mask = Mask(mask_dir)
    def __call__(self, img):
        manipulation = random.choice(self.man_list)
        masking = self.mask()
        masking_3ch = np.stack([masking, masking, masking], axis=-1).astype('uint8')
        forged_img = manipulation(img)
        img = masking_3ch * np.array(forged_img) + (1 - masking_3ch) * np.array(img)
        return Image.fromarray(img), masking

class Blur():
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        blur_id = random.randint(0, 2) # 0, 1, 2
        kernel_size = random.randrange(3, 35, 2) # 3, 5, 7, ..., 33
        if blur_id == 0:
            img = cv2.blur(img, (kernel_size, kernel_size))
        elif blur_id == 1:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_id == 2:
            img = cv2.medianBlur(img, kernel_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

class MorphOps():
    def __init__(self):
        self.kernel_size_list = list(range(2, 18, 2)) + list(range(19, 33, 2)) + [34]
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        morph_id = random.randint(0, 3) # 0, 1, 2, 3
        kernel_size = random.choice(self.kernel_size_list)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if morph_id == 0:
            img = cv2.erode(img, kernel, iterations=1)
        elif morph_id == 1:
            img = cv2.dilate(img, kernel, iterations=1)
        elif morph_id == 2:
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif morph_id == 2:
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

class Noise():
    def __call__(self, img):
        img = np.array(img)
        noise_id = random.randint(0, 2) # 0, 1, 2
        if noise_id == 0:
            img = random_noise(img, mode='gaussian')
        elif noise_id == 1:
            img = random_noise(img, mode='s&p')
        elif noise_id == 2:
            img = random_noise(img, mode='poisson')
        img = (255 * img).astype(np.uint8)
        return Image.fromarray(img)

class Quantize():
    def __call__(self, img):
        colors = random.randrange(50, 250, 25)
        img = img.quantize(colors=colors)
        return img.convert('RGB')

class AutoContrast():
    def __call__(self, img):
        cutoff = random.randint(0, 7)
        img = ImageOps.autocontrast(img, cutoff=cutoff)
        return img

class Equilize():
    def __call__(self, img):
        img = np.array(img)
        channel_id = random.randint(1, 7)
        for i in range(3):
            flag = channel_id % 2
            if flag:
                dst = cv2.equalizeHist(img[:,:,i])
                img[:,:,i] = dst
            channel_id >>= 1
        return Image.fromarray(img)

class Compress():
    def __init__(self, tmp_output_dir='./'):
        self.tmp_output_dir = tmp_output_dir
        self.JPEG_quals = [57, 53, 91, 95, 61, 65, 36, 100, 
                           74, 70, 78, 48, 44, 40, 62, 67]
        self.JPEG_double_quals = [(95, 40), (36, 74), (95, 57), (61, 65),
                                  (70, 100), (95, 65), (53, 74), (87, 100),
                                  (78, 57), (87, 48), (95, 100), (70, 65),
                                  (78, 40), (78, 48), (87, 65), (53, 40),
                                  (61, 57), (87, 57), (95, 82), (70, 91),
                                  (53, 57), (61, 40), (61, 74), (44, 48),
                                  (87, 40), (95, 91), (70, 82), (87, 82),
                                  (87, 91), (53, 65), (36, 48), (44, 74), 
                                  (36, 100), (61, 91), (36, 65), (53, 48),
                                  (53, 82), (36, 57), (78, 100), (78, 74),
                                  (53, 91), (70, 57), (87, 74), (61, 48),
                                  (70, 47), (95, 48), (78, 91), (44, 40),
                                  (78, 65), (36, 40), (53, 100), (44, 100),
                                  (36, 91), (36, 82), (70, 74), (61, 100),
                                  (61, 82), (44, 82), (78, 82), (44, 57),
                                  (44, 65), (95, 74), (70, 48), (44, 91)]
        self.WEBP_quals = [53, 87, 82, 100, 65, 61, 95, 91, 
                           36, 57, 78, 70, 74, 40, 44, 48]
    def __call__(self, img):
        compress_id = random.randint(0, 2)
        if compress_id == 0:
            quality = random.choice(self.JPEG_quals)
            tmp_output_path = os.path.join(self.tmp_output_dir, 'tmp')
            img.save(tmp_output_path, format='JPEG', quality=quality)
            img = Image.open(tmp_output_path)
        elif compress_id == 1:
            quality = random.choice(self.JPEG_double_quals)
            tmp_output_path = os.path.join(self.tmp_output_dir, 'tmp')
            img.save(tmp_output_path, format='JPEG', quality=quality[0])
            img = Image.open(tmp_output_path)
            img.save(tmp_output_path, format='JPEG', quality=quality[1])
            img = Image.open(tmp_output_path)
        elif compress_id == 2:
            quality = random.choice(self.JPEG_quals)
            tmp_output_path = os.path.join(self.tmp_output_dir, 'tmp')
            img.save(tmp_output_path, format='WEBP', quality=quality)
            img = Image.open(tmp_output_path)
        return img