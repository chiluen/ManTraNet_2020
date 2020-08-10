import transforms_enhance as T
from removal import RemoveTransform
from coco_dataset import CopyMoveDataset, SplicingDataset
from gen_patches import DresdenDataset
#from model_ASPP import create_model, model_load_weights
#from model_deeplab_v2 import create_model, model_load_weights
from model import create_model, model_load_weights
from val_dataset import VAL_Dataset
from discriminator import FCDiscriminator, spatial_CE

import torch
from torch import nn
import torch.nn.functional as F
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
path_root = "/media/chiluen/HDD"
#path_root = "./mydata"
os.mkdir(os.path.join("./log", TIMESTAMP))
writer = SummaryWriter(os.path.join("./log", TIMESTAMP))

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=30,help="epoch")
parser.add_argument("--lr", type=float, default=1e-04,help="lr")
parser.add_argument("--iter", type = int, default=100, help="iteration")
parser.add_argument("--finetune", type = str, default = "", help = "Enter finetune model.pth path")
parser.add_argument("--fp16", action="store_true", help = "Use fp16")
parser.add_argument("--dis-pretrain", action="store_true", help = "Training discriminator only")
parser.add_argument("--multi-D", type = int, default=2, help="Perform 1GXD training")
args = parser.parse_args()

class trainer():
    def __init__(self, model, optim, num_iter, criterion, scheduler, lr, finetune, fp16, flag="Gen"):
        
        self.model = model
        self.optim = optim
        self.num_iter = num_iter
        self.criterion = criterion
        self.scheduler = scheduler
        self.lr = lr
        self.finetune = finetune
        self.fp16 = fp16
        self.flag = flag
        self.global_step = 0

        self.valid_loss_min = np.Inf  

    def train(self, iters, epoch, oppo_model):
        self.model.train()
        for i in range(self.num_iter):

            if i % 2:
                rm_img, rm_masking = next(iters['rm'])
                cp_img, cp_masking = next(iters['cp'])
                img_list = [rm_img, cp_img]
                msk_list = [rm_masking, cp_masking]
            else:
                en_img, en_masking = next(iters['en'])
                sp_img, sp_masking = next(iters['sp'])
                img_list = [en_img, sp_img]
                msk_list = [en_masking, sp_masking]
            img = torch.cat(img_list, dim=0)
            gt_masking = torch.cat(msk_list, dim=0)
            #rm_img, rm_masking = next(iters['rm'])
            #en_img, en_masking = next(iters['en'])
            #cp_img, cp_masking = next(iters['cp'])
            #sp_img, sp_masking = next(iters['sp'])
            #img = torch.cat([rm_img, en_img, cp_img, sp_img], dim=0)
            #gt_masking = torch.cat([rm_masking, en_masking, cp_masking, sp_masking], dim=0)
            #img = torch.cat([rm_img, en_img], dim=0)
            #gt_masking = torch.cat([rm_masking, en_masking], dim=0)
            img = img.cuda()
            gt_masking = gt_masking.cuda()

            if self.flag == "Gen":   ##我還沒加上adv 的算法, 還要新增dis做的事情

                pred_masking = self.model(img)
                with torch.no_grad():
                    reward = oppo_model(pred_masking) # (1,H,W)
                #loss = self.criterion(pred_masking, gt_masking, reward)
                BCE, adv = self.criterion(pred_masking, gt_masking, reward)
                loss = 0.99*BCE + 0.01*adv
                self.optim.zero_grad()
                if self.fp16:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optim.step()

                if self.finetune:
                    self.model.Featex.b1c1.apply_bayar_constraint()
                    #model.outlierTrans.apply_constraint()
                    #model.glbStd.apply_clamp()
                print(self.flag + ("| Epoch: %03d | Iter: %03d | Loss: %0.5f | BCE: %0.5f | adv: %0.5f" % (epoch+1, i+1, loss.item(), BCE.item(), adv.item())))

            else: #self.flag == "Dis"

                self.optim.zero_grad()
        
                pred_masking = oppo_model(img).detach()
                fake_loss = self.criterion(self.model(pred_masking), real = False)
                real_loss = self.criterion(self.model(gt_masking), real = True)
                #ipdb.set_trace()
                loss = (fake_loss + real_loss) * 0.5

                if self.fp16:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step() #兩次backward之後再一次step
                
                print(self.flag + ("| Epoch: %03d | Iter: %03d | Real_loss: %0.5f | Fake_loss: %0.5f" % (epoch+1, i+1, real_loss.item(), fake_loss.item())))
            #writer.add_scalar('Train/Loss', loss.item(), self.global_step)
            #self.global_step += 1
            
    def valid(self, vals, epoch, oppo_model):
        self.model.eval()
        valid_loss = 0.0
        num_imgs = int(vals['rm'].__len__() / 10)
        print("#---Enter validation---#")

        for i in tqdm(range(num_imgs)):

            cp_img, cp_masking = vals['cp'].__getitem__(i)
            sp_img, sp_masking = vals['sp'].__getitem__(i)
            rm_img, rm_masking = vals['rm'].__getitem__(i)
            en_img, en_masking = vals['en'].__getitem__(i)
            
            img = torch.cat([rm_img.unsqueeze(0), en_img.unsqueeze(0), cp_img.unsqueeze(0), sp_img.unsqueeze(0)], dim=0)
            gt_masking = torch.cat([rm_masking.unsqueeze(0), en_masking.unsqueeze(0), cp_masking.unsqueeze(0), sp_masking.unsqueeze(0)], dim=0)
            img = img.cuda()
            gt_masking = gt_masking.cuda()

            if self.flag == "Gen":
                with torch.no_grad():
                    pred_masking = self.model(img)
                    reward = oppo_model(pred_masking)
                BCE, adv = self.criterion(pred_masking, gt_masking, reward)  
                loss = 0.99*BCE + 0.01*adv
                valid_loss += loss.item()

            else:  ##self.flag == "Dis"
                with torch.no_grad():
                    pred_masking = oppo_model(img)
                fake_loss = self.criterion(self.model(pred_masking), real = False)
                real_loss = self.criterion(self.model(gt_masking), real = True)
                loss = (fake_loss + real_loss) * 0.5
                valid_loss += loss.item()
                
        valid_loss = valid_loss / num_imgs #等同一個iteration的loss
        if self.scheduler:
            self.scheduler.step(valid_loss)        

        if valid_loss <= self.valid_loss_min:
                print(self.flag + '| Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), os.path.join(".", "checkpoints",str(epoch)+ "_" + self.flag + '_mantra.pth'))
                self.valid_loss_min = valid_loss

class train():
    def __init__(self, gen_trainer, dis_trainer, epoch):
        self.gen_trainer = gen_trainer
        self.dis_trainer = dis_trainer
        self.epoch = epoch

    def infinite_iter(self, dataloader):
        it = iter(dataloader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(dataloader)

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

        iters = {'rm': rm_train_iter,
                'en': en_train_iter,
                'cp': cp_train_iter,
                'sp': sp_train_iter}

        vals = {'rm': rm_val,
                'en': en_val,
                'cp': cp_val,
                'sp': sp_val}

        #--------Train----------#
        if not args.dis_pretrain:
            for i in range(self.epoch):   
                self.gen_trainer.train(iters, i, dis_trainer.model)
                for j in range(args.multi_D):
                    self.dis_trainer.train(iters, i, gen_trainer.model)
                
                #validation
                self.gen_trainer.valid(vals, i, dis_trainer.model)
                self.dis_trainer.valid(vals, i, gen_trainer.model)
        else:
            for i in range(self.epoch):
                self.dis_trainer.train(iters, i, gen_trainer.model)
                self.dis_trainer.valid(vals, i, gen_trainer.model)
            

def prepare_model():  
    mantranet = create_model(4, True)
    #mantranet = model_load_weights(os.path.join(path_root,"ManTraNet_Ptrain4.h5"), mantranet)
    mantranet.load_state_dict(torch.load("/media/chiluen/HDD/finetuned_weights.pth"))
    mantranet = mantranet.cuda()
    dis = FCDiscriminator(1)
    if not args.dis_pretrain:
        dis.load_state_dict(torch.load("/home/chiluen/Desktop/ManTraNet_2020/checkpoints/1_Dis_mantra.pth"))
    dis = dis.cuda()
    return mantranet, dis 

def adv_loss(pred_masking, gt_masking, dis_output, lambda_adv = 0.01):
    BCE = F.binary_cross_entropy(pred_masking, gt_masking)
    adv = F.binary_cross_entropy(dis_output, torch.full(dis_output.shape, 1).cuda())  
    #adv = -1 * torch.log(dis_output).mean()
    #ipdb.set_trace()
    return BCE, adv
    #return lambda_adv * adv + (1 - lambda_adv) * BCE


if __name__ == '__main__':

    #--------Construct trainer-----------#
    gen, dis = prepare_model()
    optim = torch.optim.Adam(gen.parameters(), lr = 1e-4)
    optim_dis = torch.optim.Adam(dis.parameters(), lr = 2.5e-4)
    if args.fp16:
        gen, optim = amp.initialize(gen, optim, opt_level="O1")
        dis, optim_dis = amp.initialize(dis, optim_dis, opt_level="O1")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=20)
    scheduler_dis = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_dis, factor=0.5, patience=20)
    criterion = adv_loss
    criterion_dis = spatial_CE

    gen_trainer = trainer(gen, optim, args.iter, criterion, scheduler, args.lr, args.finetune, args.fp16, "Gen")
    dis_trainer = trainer(dis, optim_dis, args.iter, criterion_dis, scheduler_dis, args.lr, args.finetune, args.fp16, "Dis")

    #--------Construct train function-----------#
    t = train(gen_trainer, dis_trainer, args.epoch)
    t.run()