import transforms_enhance as T
from removal import RemoveTransform
from coco_dataset import CopyMoveDataset, SplicingDataset
from gen_patches import DresdenDataset
from model_ASPP import create_model, model_load_weights
from val_dataset import VAL_Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter     
from torch.optim.lr_scheduler import ReduceLROnPlateau  

import os
from tqdm import tqdm
import numpy as np
import argparse

"""
Put all the data and pretrained model in this folder
"""
path_root = "./mydata"
writer = SummaryWriter('log')

class trainer():
    def __init__(self, epoch, iteration, lr):
        self.epoch = epoch
        self.iter = iteration
        self.lr = lr

        self.global_step = 0
        self.num_imgs = 0
        
    def infinite_iter(self, dataloader):
        it = iter(dataloader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(dataloader)
    
    def prepare_model(self):
        mantranet = create_model(4, True)
        mantranet = model_load_weights(os.path.join(path_root,"ManTraNet_Ptrain4.h5"), mantranet)
        return mantranet
    

    def train(self, model, optim, num_iter, iters, vals, criterion, scheduler=None,epochs = 30, valid_loss_min = np.Inf):
        valid_loss_list = []
        for epoch in range(epochs):
            model.train()
            for i in range(num_iter):
                rm_img, rm_masking = next(iters['rm'])
                en_img, en_masking = next(iters['en'])
                cp_img, cp_masking = next(iters['cp'])
                sp_img, sp_masking = next(iters['sp'])

                img = torch.cat([rm_img, en_img, cp_img, sp_img], dim=0)
                gt_masking = torch.cat([rm_masking, en_masking, cp_masking, sp_masking], dim=0)
                img = img.cuda()
                gt_masking = gt_masking.cuda()
                pred_masking = model(img)
                loss = criterion(pred_masking, gt_masking)

                optim.zero_grad()
                loss.backward()
                optim.step()
                print("Epoch: %03d | Iter: %03d | Loss: %0.5f" % (epoch+1, i+1, loss.item()))
                writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.global_step += 1

            model.eval()
            valid_loss = 0.0
            print("#---Enter validation---#")
            for i in tqdm(range(int(self.num_imgs/3))):

                cp_img, cp_masking = vals['cp'].__getitem__(i)
                sp_img, sp_masking = vals['sp'].__getitem__(i)
                rm_img, rm_masking = vals['rm'].__getitem__(i)
                en_img, en_masking = vals['en'].__getitem__(i)
                
                img = torch.cat([rm_img.unsqueeze(0), en_img.unsqueeze(0), cp_img.unsqueeze(0), sp_img.unsqueeze(0)], dim=0)
                gt_masking = torch.cat([rm_masking.unsqueeze(0), en_masking.unsqueeze(0), cp_masking.unsqueeze(0), sp_masking.unsqueeze(0)], dim=0)
                img = img.cuda()
                gt_masking = gt_masking.cuda()
                with torch.no_grad():
                    pred_masking = model(img)
                loss = criterion(pred_masking, gt_masking)
                
                valid_loss += loss.item()
                
            valid_loss = valid_loss / self.num_imgs
            if scheduler:
                scheduler.step(valid_loss)

            ##terminal condition
            try:
                if valid_loss > valid_loss_list[-1] and valid_loss > valid_loss_list[-2]:
                    print("Terminate because valid loss isn't decline")
                    return
            except:
                pass

            valid_loss_list.append(valid_loss)

            if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                    torch.save(model.state_dict(), os.path.join(".", "checkpoints",str(epoch)+'_mantra.pth'))
                    valid_loss_min = valid_loss
    
    def run(self):

        #--------Prepare dataset----------#
        # removal dataset
        rm_train_transform = RemoveTransform(os.path.join(path_root,"disocclusion_img_mask"))
        rm_train_dataset = DresdenDataset(os.path.join(path_root, "train"), 256, 256, transform=rm_train_transform)

        # enhancement dataset
        man_list = [T.Blur(),
                    T.MorphOps(),
                    T.Noise(),
                    T.Quantize(),
                    T.AutoContrast(),
                    T.Equilize(),
                    T.Compress()]

        en_train_transform = T.Enhance(man_list, os.path.join(path_root,"disocclusion_img_mask"))
        en_train_dataset = DresdenDataset(os.path.join(path_root, "train"), 256, 256, transform=en_train_transform)

        # copy-move and spliced dataset
        json_path = os.path.join(path_root,"instances_train2017.json")
        pic_path = os.path.join(path_root, "train2017")
        cp_train_dataset = CopyMoveDataset(json_path, pic_path, 256, 256)
        sp_train_dataset = SplicingDataset(json_path, pic_path ,256, 256)

        #--------Prepare dataloader----------#
        rm_train_dataloader = DataLoader(rm_train_dataset, batch_size=2, shuffle=True)
        en_train_dataloader = DataLoader(en_train_dataset, batch_size=2, shuffle=True)
        cp_train_dataloader = DataLoader(cp_train_dataset, batch_size=2, shuffle=True)
        sp_train_dataloader = DataLoader(sp_train_dataset, batch_size=2, shuffle=True)

        rm_train_iter = self.infinite_iter(rm_train_dataloader)
        en_train_iter = self.infinite_iter(en_train_dataloader)
        cp_train_iter = self.infinite_iter(cp_train_dataloader)
        sp_train_iter = self.infinite_iter(sp_train_dataloader)

        rm_val = VAL_Dataset(os.path.join(path_root,'rm_val'))
        en_val = VAL_Dataset(os.path.join(path_root,'en_val'))
        sp_val = VAL_Dataset(os.path.join(path_root,'Sliced_val_coco'))
        cp_val = VAL_Dataset(os.path.join(path_root,'Copymove_val_coco'))
        self.num_imgs = rm_val.__len__()

        #--------train----------#
        mantranet = self.prepare_model()
        optim = torch.optim.Adam(mantranet.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=20)
        criterion = nn.BCELoss()
        iters = {'rm': rm_train_iter,
                'en': en_train_iter,
                'cp': cp_train_iter,
                'sp': sp_train_iter}

        vals = {'rm': rm_val,
                'en': en_val,
                'cp': cp_val,
                'sp': sp_val}

        self.train(mantranet, optim, self.iter, iters, vals, criterion, scheduler, self.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30,help="epoch")
    parser.add_argument("--lr", type=float, default=1e-04,help="lr")
    parser.add_argument("--iter", type = int, default=100, help="iteration")
    args = parser.parse_args()

    t = trainer(args.epoch, args.iter, args.lr)
    t.run()


















