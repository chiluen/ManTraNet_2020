import random
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

'''
Usage Example
json_path = "/home/chiluen/Desktop/coco/annotations/instances_train2017.json"
pic_path = "/home/chiluen/Desktop/coco/train2017" 
dataset = CopyMoveDataset(json_path, pic_path)
# or dataset = SplicingDataset(json_path, pic_path)
img, masking = dataset[0]
'''

class CopyMoveDataset(Dataset):    
    def __init__(self, json_path, pic_path, height, width):
        self.json_path = json_path
        self.pic_path = pic_path
        self.height = height
        self.width = width
        self.coco = COCO(self.json_path)
        self.to_tensor = transforms.ToTensor()
        self.exist_pic = []

    def __len__(self):
        return 100000

    def __getitem__(self, idx):

        img, masking =  self.generate_picture()
        while img.shape[0] < self.height or img.shape[1] < self.width:
            img, masking =  self.generate_picture()
        masking = masking[:, :, :1]
        img, masking = self.to_tensor(img), self.to_tensor(masking)
        # crop to (height, width)
        img_h, img_w = img.shape[-2], img.shape[-1]
        start_h = random.randint(0, img_h - self.height - 1)
        start_w = random.randint(0, img_w - self.width - 1)
        
        terminate_flag = 0
        while masking[:, start_h:start_h+self.height, start_w:start_w+self.width].sum() == 0:
            start_h = random.randint(0, img_h - self.height - 1)
            start_w = random.randint(0, img_w - self.width - 1)
            if terminate_flag == 1000:
                terminate_flag = 0

                #重新generate一張
                while True: 
                    img, masking =  self.generate_picture()
                    if img.shape[0] > self.height and img.shape[1] > self.width:
                        break
                masking = masking[:, :, :1]
                img, masking = self.to_tensor(img), self.to_tensor(masking)
                # crop to (height, width)
                img_h, img_w = img.shape[-2], img.shape[-1]
                start_h = random.randint(0, img_h - self.height - 1)
                start_w = random.randint(0, img_w - self.width - 1)
                
        img = img[:, start_h:start_h+self.height, start_w:start_w+self.width]
        masking = masking[:, start_h:start_h+self.height, start_w:start_w+self.width]

        return img, masking
        
    def generate_picture(self):

        cats = self.coco.loadCats(self.coco.getCatIds())  
        nums_cats=[cat['name'] for cat in cats] #總共80種
        catNms = []
        imgIds = []
        while imgIds == []: #有可能找不到2種種類的搭配
            catNms = []
            catIds = []
            for i in np.random.choice(np.arange(80), 2,replace=False): #隨機選出category
                catNms.append(nums_cats[i])
            catIds = self.coco.getCatIds(catNms=catNms)
            imgIds = self.coco.getImgIds(catIds=catIds)

        imgIds = self.coco.getImgIds(imgIds = np.random.choice(imgIds))   ##這個也要隨機
        img = self.coco.loadImgs([imgIds[np.random.randint(0,len(imgIds))]])[0]
        I = io.imread('%s/%s'%(self.pic_path, img['file_name']))
        
        #有時候會用到黑白相片
        try:
            channel_count = I.shape[2]
        except:
            #print("Load to gray pic")
            return self.generate_picture()
        
        ##確認是否load過
        if img['id'] in self.exist_pic:
            #print("Repeat ID")
            return self.generate_picture()
        
        self.exist_pic.append(img['id'])
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds) #bounding box

        ##選最大面積的
        area = 0
        object_choose = 0
        for i in range(len(anns)):
            if anns[i]['area'] > area:
                area = anns[i]['area']
                object_choose = i 
                
        if area <= 6000:
            #print("Area is too small")
            return self.generate_picture()
        

        try:
            polygon = anns[object_choose]['segmentation'][0]  ##有時候照片會沒有這個選項
        except:
            #print("There arn't any segmentation info")
            return self.generate_picture()

        polygon = [polygon[i:i+2] for i in range(0,len(polygon),2)]
        polygon = np.array(polygon, dtype = np.int32)
        polygon = polygon[np.newaxis,:,:]

        mask = np.zeros(I.shape, dtype=np.uint8)

        ignore_mask_color = (255,)*channel_count #讓它變成彩色
        cv2.fillPoly(mask, polygon, ignore_mask_color)   #這樣子會讓mask變成黑白剪影

        # apply the mask
        masked_image = cv2.bitwise_and(I, mask)


        def img_resize_large(img, bbox):
            size = img.shape
            h, w = size[0], size[1]
            min_side = 100

            scale = max(w, h) / float(min_side)
            new_w, new_h = int(w/scale), int(h/scale)
            box_x = int(bbox[0])
            box_y = int(bbox[1])
            box_w = int(bbox[2])
            box_h = int(bbox[3])
            crop_img = img[box_y : box_y+box_h, box_x : box_x+box_w]

            resize_img = cv2.resize(crop_img, (new_w, new_h)) #以min_side作為scale,進行縮放
            # 填充至min_side * min_side

            if new_w % 2 != 0 and new_h % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            elif new_h % 2 != 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            elif new_h % 2 == 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            else:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
            pad_img = cv2.resize(pad_img, (w,h)) #有時候會有一點誤差, 用resize校正

            return pad_img


        ##不一定要執行, 這都變小
        def img_resize(img):
            size = img.shape
            h, w = size[0], size[1]
            min_side = 5000
            if min_side >= h or min_side >= w:
                min_side = min(h,w)

            scale = max(w, h) / float(min_side)
            new_w, new_h = int(w/scale), int(h/scale)
            resize_img = cv2.resize(img, (new_w, new_h)) #以min_side作為scale,進行縮放
            # 填充至min_side * min_side


            if new_w % 2 != 0 and new_h % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            elif new_h % 2 != 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            elif new_h % 2 == 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            else:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
            pad_img = cv2.resize(pad_img, (w,h)) #有時候會有一點誤差, 用resize校正

            return pad_img

        def img_rotate(img, angle, center=None, scale=1.0):

            (h, w) = img.shape[:2]

            if center is None:
                center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(img, M, (w, h))

            return rotated

        def img_move(img, x, y):

            (h, w) = img.shape[:2]
            M = np.float32([[1, 0, x], [0, 1, y]])
            shifted = cv2.warpAffine(img, M, (w, h))

            return shifted


        #若黑白圖只有黑沒有白, 那就捨棄這一組(代表我move太多)
        def img_binary(img):
            _, black_white = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY) #過thershold, 就會轉成白色(255)
            
            #檢查如果都是黑色, 那就drop這一組
            Flag = False
            if np.all(img == black_white):
                Flag = True
            return black_white, Flag

        def img_paste(masked_image, I):
            """
            概念：
            先把原圖要放圖案的位置歸零
            然後再把圖案做成只有圖案那邊有值
            最後兩者相加就有疊圖效果了
            """

            img2gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY) #把切下來的img轉成gray
            ret,mask = cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY_INV) #製造白黑
            mask_inv = cv2.bitwise_not(mask) #製造黑白
            img1_bg = cv2.bitwise_and(I, I, mask=mask)  #把原圖部份值歸零
            img2_fg = cv2.bitwise_and(masked_image, masked_image,mask = mask_inv) #把切下來的img製作成黑底彩色圖案
            dst = cv2.add(img1_bg,img2_fg) #黏上去
            return dst

        #masked_image = img_resize(masked_image)
        #masked_image = img_resize_large(masked_image, anns[object_choose]['bbox'])
        masked_image = img_rotate(masked_image, 50)
        masked_image = img_move(masked_image, np.random.randint(-50,50), np.random.randint(-150,150)) #隨機移動   
    
    
        ##erosion: 把圖片黑邊消除
        ret,masked_image_temp = cv2.threshold(masked_image,0,255,cv2.THRESH_BINARY)
        erosion = cv2.erode(masked_image_temp,kernel = (3,3), iterations = 5)
        ret, erosion = cv2.threshold(erosion,0,1,cv2.THRESH_BINARY)
        ##確認erosion是否都是0,1
        
        masked_image = np.multiply(masked_image, erosion)
        
        ground_truth, not_available_flag = img_binary(masked_image)

        if not_available_flag:
            #print('Not available because of moving too much')
            return self.generate_picture()

        train_image = img_paste(masked_image, I)
        
        return train_image, ground_truth


class SplicingDataset():
    def __init__(self, json_path, pic_path, height, width):
        self.json_path = json_path
        self.pic_path = pic_path
        self.height = height
        self.width = width
        self.coco = COCO(self.json_path)
        self.to_tensor = transforms.ToTensor()
        self.exist_pic = []

    def __len__(self):
        return 100000

    def __getitem__(self, idx):

        img, masking =  self.generate_picture()
        while img.shape[0] < self.height or img.shape[1] < self.width:
            img, masking =  self.generate_picture()
        masking = masking[:, :, :1]
        img, masking = self.to_tensor(img), self.to_tensor(masking)
        # crop to (height, width)
        img_h, img_w = img.shape[-2], img.shape[-1]
        start_h = random.randint(0, img_h - self.height - 1)
        start_w = random.randint(0, img_w - self.width - 1)
        
        terminate_flag = 0
        while masking[:, start_h:start_h+self.height, start_w:start_w+self.width].sum() == 0:
            start_h = random.randint(0, img_h - self.height - 1)
            start_w = random.randint(0, img_w - self.width - 1)
            if terminate_flag == 1000:
                terminate_flag = 0

                #重新generate一張
                while True: 
                    img, masking =  self.generate_picture()
                    if img.shape[0] > self.height and img.shape[1] > self.width:
                        break
                masking = masking[:, :, :1]
                img, masking = self.to_tensor(img), self.to_tensor(masking)
                # crop to (height, width)
                img_h, img_w = img.shape[-2], img.shape[-1]
                start_h = random.randint(0, img_h - self.height - 1)
                start_w = random.randint(0, img_w - self.width - 1)
                
        img = img[:, start_h:start_h+self.height, start_w:start_w+self.width]
        masking = masking[:, start_h:start_h+self.height, start_w:start_w+self.width]
              
        return img, masking
        
    def generate_picture(self):

        cats = self.coco.loadCats(self.coco.getCatIds())  
        nums_cats=[cat['name'] for cat in cats] #總共80種
        catNms = []
        imgIds = []
        while imgIds == []: #有可能找不到2種種類的搭配
            catNms = []
            catIds = []
            for i in np.random.choice(np.arange(80), 2,replace=False): #隨機選出category
                catNms.append(nums_cats[i])
            catIds = self.coco.getCatIds(catNms=catNms)
            imgIds = self.coco.getImgIds(catIds=catIds)
        imgIds = self.coco.getImgIds(imgIds = np.random.choice(imgIds))   ##這個也要隨機
        img = self.coco.loadImgs([imgIds[np.random.randint(0,len(imgIds))]])[0]
        I = io.imread('%s/%s'%(self.pic_path, img['file_name']))
        
        
        ##src_image
        catNms_src = []
        imgIds_src = []
        while imgIds_src == []:
            catNms_src = []
            imgIds_src = []
            for i in np.random.choice(np.arange(80), 2,replace=False):
                catNms_src.append(nums_cats[i])
            catIds_src = self.coco.getCatIds(catNms = catNms_src)
            imgIds_src = self.coco.getImgIds(catIds = catIds_src)
        imgIds_src = self.coco.getImgIds(imgIds = np.random.choice(imgIds_src))
        img_src = self.coco.loadImgs([imgIds_src[np.random.randint(0,len(imgIds_src))]])[0]
        I_src = io.imread('%s/%s'%(self.pic_path, img_src['file_name']))
        
        
        #有時候會用到黑白相片
        try:
            channel_count = I.shape[2]
            channel_count = I_src.shape[2] #檢查src是不是也是gray
        except:
            #print("Load to gray pic")
            return self.generate_picture()
        
        ##確認是否load過
        if img_src['id'] in self.exist_pic:
            #print("Repeat ID")
            return self.generate_picture()
        
        self.exist_pic.append(img_src['id'])
        
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds) #bounding box

        ##選最大面積的
        area = 0
        object_choose = 0
        for i in range(len(anns)):
            if anns[i]['area'] > area:
                area = anns[i]['area']
                object_choose = i 
                
        if area <= 6000:
            #print("Area is too small")
            return self.generate_picture()
        

        try:
            polygon = anns[object_choose]['segmentation'][0]  ##有時候照片會沒有這個選項
        except:
            #print("There arn't any segmentation info")
            return self.generate_picture()

        polygon = [polygon[i:i+2] for i in range(0,len(polygon),2)]
        polygon = np.array(polygon, dtype = np.int32)
        polygon = polygon[np.newaxis,:,:]

        mask = np.zeros(I.shape, dtype=np.uint8)

        ignore_mask_color = (255,)*channel_count #讓它變成彩色
        cv2.fillPoly(mask, polygon, ignore_mask_color)   #這樣子會讓mask變成黑白剪影

        # apply the mask
        masked_image = cv2.bitwise_and(I, mask)


        def img_resize_large(img, img_src,bbox):
            size = img.shape
            h, w = size[0], size[1]
            size_src = img_src.shape
            h_src, w_src = size_src[0], size_src[1]   
            
            min_side = 100

            scale = max(w, h) / float(min_side)
            new_w, new_h = int(w/scale), int(h/scale)
            box_x = int(bbox[0])
            box_y = int(bbox[1])
            box_w = int(bbox[2])
            box_h = int(bbox[3])
            crop_img = img[box_y : box_y+box_h, box_x : box_x+box_w]

            resize_img = cv2.resize(crop_img, (new_w, new_h)) #以min_side作為scale,進行縮放
            # 填充至min_side * min_side

            if new_w % 2 != 0 and new_h % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            elif new_h % 2 != 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            elif new_h % 2 == 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            else:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
            pad_img = cv2.resize(pad_img, (w_src,h_src)) #resize成 src image size

            return pad_img


        ##不一定要執行, 這都變小
        def img_resize(img):
            size = img.shape
            h, w = size[0], size[1]
            min_side = 5000
            if min_side >= h or min_side >= w:
                min_side = min(h,w)

            scale = max(w, h) / float(min_side)
            new_w, new_h = int(w/scale), int(h/scale)
            resize_img = cv2.resize(img, (new_w, new_h)) #以min_side作為scale,進行縮放
            # 填充至min_side * min_side


            if new_w % 2 != 0 and new_h % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            elif new_h % 2 != 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            elif new_h % 2 == 0 and new_w % 2 == 0:
                top, bottom, left, right = (h-new_h)/2, (h-new_h)/2, (w-new_w)/2, (w-new_w)/2
            else:
                top, bottom, left, right = (h-new_h)/2 + 1, (h-new_h)/2, (w-new_w)/2 + 1, (w-new_w)/2
            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
            pad_img = cv2.resize(pad_img, (w,h)) #有時候會有一點誤差, 用resize校正

            return pad_img

        def img_rotate(img, angle, center=None, scale=1.0):

            (h, w) = img.shape[:2]

            if center is None:
                center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(img, M, (w, h))

            return rotated

        def img_move(img, x, y):

            (h, w) = img.shape[:2]
            M = np.float32([[1, 0, x], [0, 1, y]])
            shifted = cv2.warpAffine(img, M, (w, h))

            return shifted


        #若黑白圖只有黑沒有白, 那就捨棄這一組(代表我move太多)
        def img_binary(img):
            _, black_white = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY) #過thershold, 就會轉成白色(255)
            
            #檢查如果都是黑色, 那就drop這一組
            Flag = False
            if np.all(img == black_white):
                Flag = True
            return black_white, Flag

        def img_paste(masked_image, I):
            """
            概念：
            先把原圖要放圖案的位置歸零
            然後再把圖案做成只有圖案那邊有值
            最後兩者相加就有疊圖效果了
            """

            img2gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY) #把切下來的img轉成gray
            ret,mask = cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY_INV) #製造白黑
            mask_inv = cv2.bitwise_not(mask) #製造黑白
            img1_bg = cv2.bitwise_and(I, I, mask=mask)  #把原圖部份值歸零
            img2_fg = cv2.bitwise_and(masked_image, masked_image,mask = mask_inv) #把切下來的img製作成黑底彩色圖案
            dst = cv2.add(img1_bg,img2_fg) #黏上去
            return dst

        #masked_image = img_resize(masked_image)
        masked_image = img_resize_large(masked_image, I_src, anns[object_choose]['bbox'])
        masked_image = img_rotate(masked_image, 50)
        masked_image = img_move(masked_image, np.random.randint(-50,50), np.random.randint(-150,150)) #隨機移動   
    
    
        ##erosion: 把圖片黑邊消除
        ret,masked_image_temp = cv2.threshold(masked_image,0,255,cv2.THRESH_BINARY)
        erosion = cv2.erode(masked_image_temp,kernel = (3,3), iterations = 5)
        ret, erosion = cv2.threshold(erosion,0,1,cv2.THRESH_BINARY)
        ##確認erosion是否都是0,1
        
        masked_image = np.multiply(masked_image, erosion)
        
        ground_truth, not_available_flag = img_binary(masked_image)

        if not_available_flag:
            #print('Not available because of moving too much')
            return self.generate_picture()

        train_image = img_paste(masked_image, I_src)
        
        """
        plt.subplot(1, 2, 1)
        plt.imshow(train_image)

        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth)
        """
        return train_image, ground_truth