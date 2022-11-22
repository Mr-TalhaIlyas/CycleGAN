import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import cv2, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import images_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer(object):
    def __init__(self, gen_A, gen_B, disc_A, disc_B, optim_disc, optim_gen, l1, mse, g_scaler, d_scaler):
        self.gen_A = gen_A
        self.gen_B = gen_B
        self.disc_A = disc_A
        self.disc_B = disc_B
        self.optim_disc = optim_disc
        self.optim_gen = optim_gen
        self.l1 = l1
        self.mse = mse
        self.g_scaler = g_scaler
        self.d_scaler = d_scaler
    
    def training_step(self, batched_data, return_all=False):
        # imgA_batch = images_transform(batched_data['imgA']) # to tensor and to cuda
        # imgB_batch = images_transform(batched_data['imgB'])
        imgA_batch, imgB_batch = batched_data
        imgA_batch = imgA_batch.to(DEVICE)
        imgB_batch = imgB_batch.to(DEVICE)
        # TRAIN DISCRIMINATORS
        with torch.cuda.amp.autocast():
            # train 1st discriminator
            fake_A = self.gen_A(imgB_batch)
            real_A_feats = self.disc_A(imgA_batch)
            fake_A_feats = self.disc_A(fake_A.detach())

            real_A_loss = self.mse(real_A_feats, torch.ones_like(real_A_feats))
            fake_A_loss = self.mse(fake_A_feats, torch.zeros_like(fake_A_feats))

            disc_A_loss = real_A_loss + fake_A_loss
            
            # train 2nd discriminator
            fake_B = self.gen_B(imgA_batch)
            real_B_feats = self.disc_B(imgB_batch)
            fake_B_feats = self.disc_B(fake_B.detach())

            real_B_loss = self.mse(real_B_feats, torch.ones_like(real_B_feats))
            fake_B_loss = self.mse(fake_B_feats, torch.zeros_like(fake_B_feats))

            disc_B_loss = real_B_loss + fake_B_loss

            disc_loss = (disc_A_loss + disc_B_loss) / 2
        
        self.optim_disc.zero_grad()
        self.d_scaler.scale(disc_loss).backward()
        self.d_scaler.step(self.optim_disc)
        self.d_scaler.update()

        # TRAIN GENERATOR
        with torch.cuda.amp.autocast():
            # tain 1st generator
            fake_A_feats = self.disc_A(fake_A) # they should not be able to tell real and fake apart.
            fake_B_feats = self.disc_B(fake_B) # if generators are doing a good job.

            gen_A_loss = self.mse(fake_A_feats, torch.ones_like(fake_A_feats))
            gen_B_loss = self.mse(fake_B_feats, torch.ones_like(fake_B_feats))

            # cycle loss:: A-> B then B -> A
            cycle_A = self.gen_A(fake_B)
            cycle_B = self.gen_B(fake_A)
            cycleA_loss = self.l1(imgA_batch, cycle_A)
            cycleB_loss = self.l1(imgB_batch, cycle_B)

            # IDENTITY loss A->A then A -> A
            identity_A = self.gen_A(imgA_batch)
            identity_B = self.gen_B(imgB_batch)
            identityA_loss = self.l1(imgA_batch, identity_A)
            identityB_loss = self.l1(imgB_batch, identity_B)

            G_loss = (gen_A_loss
                      + gen_B_loss
                      + cycleA_loss * config['LAMBDA_CYCLE']
                      + cycleB_loss * config['LAMBDA_CYCLE']
                      + identityA_loss * config['LAMBDA_IDENTITY']
                      + identityB_loss * config['LAMBDA_IDENTITY']
                      )
        
        self.optim_gen.zero_grad()
        self.g_scaler.scale(G_loss).backward()
        self.g_scaler.step(self.optim_gen)
        self.g_scaler.update()

        if return_all:
            return (G_loss, disc_loss, cycleA_loss, cycleB_loss, identityA_loss,
                    identityB_loss, real_A_loss, real_B_loss, fake_A_loss, fake_B_loss, fake_A, fake_B)
        else:
            return (G_loss.item(), disc_loss.item(), fake_A, fake_B)


