import transforms_enhance as T
from removal import RemoveTransform
from coco_dataset import CopyMoveDataset, SplicingDataset
from gen_patches import DresdenDataset
#from model_ASPP import create_model, model_load_weights
#from model_deeplab_v2 import create_model, model_load_weights
from model_threshold_ASPP import create_model, model_load_weights
from val_dataset import VAL_Dataset
from module import dice_loss, tversky_loss

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter     
from torch.optim.lr_scheduler import ReduceLROnPlateau  

import os
from tqdm import tqdm
import numpy as np
import argparse
import ipdb
from datetime import datetime
from apex import amp
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

"""
Put all the data and pretrained model in this folder
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=30,help="epoch")
parser.add_argument("--lr", type=float, default=1e-04,help="lr")
parser.add_argument("--iter", type = int, default=100, help="iteration")
parser.add_argument("--finetune", type = str, default = "", help = "Enter finetune model.pth path")
parser.add_argument("--fp16", action="store_true", help = "Use fp16")
parser.add_argument("--aspp-loss", type = str, default = "tversky")
parser.add_argument("--mantra-loss", type = str, default= "BCE")
parser.add_argument("--threshold", type = float, default = 0.5)
args = parser.parse_args()


path_root = "/media/chiluen/HDD"
#path_root = "./mydata"
os.mkdir(os.path.join("./log", TIMESTAMP))
os.mkdir(os.path.join("./checkpoints", args.aspp_loss + "_" +args.mantra_loss + "_" +str(args.threshold)))
checkpoint_dir = os.path.join("./checkpoints", args.aspp_loss + "_" +args.mantra_loss + "_" +str(args.threshold))
writer = SummaryWriter(os.path.join("./log", TIMESTAMP))


class trainer():
    def __init__(self, epoch, iteration, lr, finetune, fp16):
        self.epoch = epoch
        self.iter = iteration
        self.lr = lr
        self.finetune = finetune
        self.fp16 = fp16

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
        if not self.finetune:    
            mantranet = create_model(4, False, threshold=args.threshold)
            mantranet = model_load_weights(os.path.join(path_root,"ManTraNet_Ptrain4.h5"), mantranet)
        else:
            mantranet = create_model(4, False)
            mantranet.load_state_dict(torch.load(self.finetune))
            mantranet = mantranet.cuda()
        
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
                #先aspp
                pred_masking = model(img, aspp = True)
                loss = criterion[0](pred_masking, gt_masking) #dice loss
                optim[0].zero_grad()
                if self.fp16:
                    with amp.scale_loss(loss, optim[0]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optim[0].step()

                #再整個model
                pred_masking = model(img)
                loss = criterion[1](pred_masking, gt_masking)

                optim[1].zero_grad()
                if self.fp16:
                    with amp.scale_loss(loss, optim[1]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optim[1].step()
                if self.finetune:
                    model.Featex.b1c1.apply_bayar_constraint()
                    #model.outlierTrans.apply_constraint()
                    #model.glbStd.apply_clamp()

                print("Epoch: %03d | Iter: %03d | Loss: %0.5f" % (epoch+1, i+1, loss.item()))
                writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.global_step += 1
            model.eval()
            valid_loss = 0.0
            print("#---Enter validation---#")
            
            for i in tqdm(range(int(self.num_imgs/30))):

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
                loss = criterion[1](pred_masking, gt_masking)
                
                valid_loss += loss.item()
                
            valid_loss = valid_loss / self.num_imgs
            if scheduler:
                scheduler.step(valid_loss)

            ##terminal condition
            """
            try:
                if valid_loss > valid_loss_list[-1] and valid_loss > valid_loss_list[-2]:
                    print("Terminate because valid loss isn't decline")
                    return
            except:
                pass
            """
            valid_loss_list.append(valid_loss)
            """
            if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                    torch.save(model.state_dict(), os.path.join(".", "checkpoints",str(epoch)+'_mantra.pth'))
                    valid_loss_min = valid_loss
            """
            
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(epoch)+'_mantra.pth'))
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
        batch_size = 1
        rm_train_dataloader = DataLoader(rm_train_dataset, batch_size=batch_size, shuffle=True)
        en_train_dataloader = DataLoader(en_train_dataset, batch_size=batch_size, shuffle=True)
        cp_train_dataloader = DataLoader(cp_train_dataset, batch_size=batch_size, shuffle=True)
        sp_train_dataloader = DataLoader(sp_train_dataset, batch_size=batch_size, shuffle=True)

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
        optim_1 = torch.optim.Adam([{'params':mantranet.Featex.parameters()}, {'params':mantranet.aspp.parameters()}, {'params':mantranet.unet.parameters()}], lr = self.lr)
        optim_2 = torch.optim.Adam(mantranet.parameters(), lr = self.lr)
        optim = [optim_1, optim_2]
        """
        optim = torch.optim.Adam([
                    {'params':mantranet.Featex.parameters()},
                    {'params': [p for n, p in mantranet.named_parameters() if 'Featex' not in n],'lr': 1e-4}
                ], lr=5e-5)
        """



        if self.fp16:
            mantranet, optim = amp.initialize(mantranet, optim, opt_level="O1")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim[1], factor=0.5, patience=20)


        if args.aspp_loss == "tversky":
            criterion_1 = tversky_loss.TverskyLoss()
        else:
            criterion_1 = nn.BCEWithLogitsLoss()

        if args.mantra_loss == "BCE":
            criterion_2 = nn.BCEWithLogitsLoss()
        else:
            criterion_2 = tversky_loss.TverskyLoss()

        #criterion_1 = tversky_loss.TverskyLoss()
        #criterion_2 = nn.BCEWithLogitsLoss()
        criterion = [criterion_1, criterion_2]
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

    t = trainer(args.epoch, args.iter, args.lr, args.finetune, args.fp16)
    t.run()