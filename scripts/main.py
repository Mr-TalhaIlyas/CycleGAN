#%%
import os
os.chdir('/home/user01/data/talha/cyclegan/scripts/')
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
               config_include_keys=config.keys(), config=config)

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import CycelGANDataset
from data_utils import collate
from torch.utils.data import DataLoader
import numpy as np
import imgviz
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm
from utils import Trainer, save_checkpoint
from torchvision.utils import save_image
from lr_scheduler import CycleGAN_LR

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

torch.backends.cudnn.benchmark = config['BENCHMARK']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = CycelGANDataset(config['data_dir'], config['img_height'], config['img_width'],
                             True, config['Normalize_data'], config['Augment_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                          collate_fn=collate,
                          pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=True)

val_data = CycelGANDataset(config['data_dir'], config['img_height'], config['img_width'],
                           False, config['Normalize_data'], config['Augment_data'])

val_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                          collate_fn=collate,
                          pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=True)
# DataLoader Sanity Checks
batch = next(iter(train_loader))
s=127.5
img_ls = []
[img_ls.append(((batch['imgA'][i]+1)*s).astype(np.uint8)) for i in range(config['batch_size'])]
[img_ls.append(((batch['imgB'][i]+1)*s).astype(np.uint8)) for i in range(config['batch_size'])]
plt.title('Sample Batch')
plt.imshow(imgviz.tile(img_ls, shape=(2,config['batch_size']), border=(255,0,0)))
plt.axis('off')                   
#%%
'''
1. Make 2 Discirminators
    1 to disc. dataset A fake images from dataset A real images
    2nd for disc. dataset B fake images from dataset B real images
2. make 2 generators
    1 to generate data A images
    2nd to generate data B images
3. make 2 optimizers 
    one for discriminators
    other for generators
4. loss functions for generators and discriminators.
'''

gen_A = Generator(inChannel=3, genChannel=64, residual_blocks=9).to(DEVICE)
gen_B = Generator(inChannel=3, genChannel=64, residual_blocks=9).to(DEVICE)

disc_A = Discriminator(inChannel=3).to(DEVICE)
disc_B = Discriminator(inChannel=3).to(DEVICE)

optim_disc = optim.Adam(
    list(disc_A.parameters()) + list(disc_B.parameters()),
    lr = float(config['learning_rate']),
    betas = (0.5, 0.999)
)

optim_gen = optim.Adam(
    list(gen_A.parameters()) + list(gen_B.parameters()),
    lr = float(config['learning_rate']),
    betas = (0.5, 0.999)
    )
lambda1 = lambda epoch: 1.0 - max(0, epoch + config['start_epoch'] - config['decay_epoch']) / (config['epochs'] - config['decay_epoch'])

gen_lr_scheduler = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=lambda1)
disc_lr_scheduler = optim.lr_scheduler.LambdaLR(optim_disc, lr_lambda=lambda1)

L1 = nn.L1Loss()
mse = nn.MSELoss()

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(gen_A, gen_B, disc_A, disc_B, optim_disc, optim_gen,
                  L1, mse, g_scaler, d_scaler)

#%%
save_chkpt = False # auto save
for epoch in range(config['epochs']):

    pbar = tqdm(train_loader)
    gl, dl, tr, tf = [], [], [], []
    for step, data_batch in enumerate(pbar):

        out = trainer.training_step(step, epoch, data_batch, save_chkpt)
        save_chkpt = False # RESET

        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - G_loss {out["Gen_Loss"]:.4f} - D_loss {out["Disc_Loss"]:.4f}')
        gl.append(out["Gen_Loss"])
        dl.append(out["Disc_Loss"])

    # change LR every epoch
    gen_lr_scheduler.step()
    disc_lr_scheduler.step()

    print(f'=> Average G-loss: {np.nanmean(gl)}, Average D-loss: {np.nanmean(dl)}')

    if (epoch + 1) % 10 == 0:
        FA = list(((out["fake_a"]*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
        FB = list(((out["fake_b"]*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
        RA = list(((out["img_a"]*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
        RB = list(((out["img_b"]*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
        # becaluse fake iamge A is made from raal image B.
        tiled = imgviz.tile(RA+RB+FB+FA, shape=(2,2), border=(255,0,0))

    if (epoch + 1) % 50 == 0:
        save_chkpt = True

        if config['LOG_WANDB']:
            wandb.log({'Generations': wandb.Image(tiled)}, step=epoch+1)
        
    if config['LOG_WANDB']:
        wandb.log({"G_Loss": np.nanmean(gl), "D_Loss": np.nanmean(dl),
                    "Gen_LR" : out["Gen_LR"], "Disc_LR" : out["Disc_LR"]}, step=epoch+1)
    
if config['LOG_WANDB']:
    wandb.run.finish()
#%%

# x = []
# for i in range(200):
#     optim_disc.step()
#     disc_lr_scheduler.step()
#     x.append(optim_disc.param_groups[0]['lr'])