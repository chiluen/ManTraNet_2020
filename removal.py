import os
import numpy as np
import cv2
import random
from torchvision import transforms as T

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated


def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def mask(mask_folder):
    masks = os.listdir(mask_folder)
    pth = mask_folder + random.choice(masks)
    mk = cv2.imread( pth, cv2.IMREAD_GRAYSCALE)[...,::-1]
    mk = (1-mk)/255

    _,thsh = cv2.threshold(mk, 0.6, 1, cv2.THRESH_BINARY)

    num = random.randint(3, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(num, num))
    dilated = cv2.dilate(thsh, kernel)

    num = [random.randint(-100,100)for i in range(2)]

    shifted = translate(dilated, num[0], num[1])

    num = random.randint(0, 180)

    rotated = rotate(shifted, num)
    
    cropped = get_random_crop(rotated, 256, 256)
    if np.array_equal(np.zeros((256,256)), cropped):
        masking = mask(mask_folder)
    else:
        masking = cropped
    return masking

# def removal(img_folder = '/home/jayda960825/Documents/Dresden/Dresden_JPEG/',
#     removal_folder = '/home/jayda960825/Documents/removal/'):
#     imgs = os.listdir(img_folder)
#     for i in imgs:
#         masking = mask()
#         masking = masking.astype('uint8')
#         pth = img_folder + i
#         img = cv2.imread(pth, 1)[...,::-1]
#         img = get_random_crop(img, 256, 256)
#         output = cv2.inpaint(img,masking,3,cv2.INPAINT_NS)
#         cv2.imwrite(removal_folder + i[:-4] + '.jpg', output)
#         masking = masking*255
#         cv2.imwrite(removal_folder+ i[:-4] + '_mask.png', masking)

def removal(img, mask_folder):
    masking = mask(mask_folder)
    masking = masking.astype('uint8')
    output = cv2.inpaint(np.uint8(img),masking,3,cv2.INPAINT_NS)
    #cv2.imwrite('/home/jayda960825/Desktop/mask.png', masking*255)
    return output

def rm_transform(mask_folder = '/home/jayda960825/Documents/irregular_mask/disocclusion_img_mask/'):
    data_transforms = T.Compose([
        T.Lambda(lambda img: removal(img, mask_folder)),
        T.ToTensor()
    ])
    return data_transforms