#%%
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
from utils import Trainer
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = CycelGANDataset(config['data_dir'], config['img_height'], config['img_width'],
                             True, config['Normalize_data'], config['Augment_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                        #   collate_fn=collate,
                          pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=True)

val_data = CycelGANDataset(config['data_dir'], config['img_height'], config['img_width'],
                           False, config['Normalize_data'], config['Augment_data'])

val_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                        #   collate_fn=collate,
                          pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=True)
# # DataLoader Sanity Checks
# batch = next(iter(train_loader))
# s=127.5
# img_ls = []
# [img_ls.append(((batch['imgA'][i]+1)*s).astype(np.uint8)) for i in range(config['batch_size'])]
# [img_ls.append(((batch['imgB'][i]+1)*s).astype(np.uint8)) for i in range(config['batch_size'])]
# plt.title('Sample Batch')
# plt.imshow(imgviz.tile(img_ls, shape=(2,config['batch_size']), border=(255,0,0)))
# plt.axis('off')                   
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

L1 = nn.L1Loss()
mse = nn.MSELoss()

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(gen_A, gen_B, disc_A, disc_B, optim_disc, optim_gen, L1, mse, g_scaler, d_scaler)

#%%
for epoch in range(config['epochs']):

    pbar = tqdm(train_loader)
    gl, dl = [], []
    for step, data_batch in enumerate(pbar):

        gen_loss, disc_loss, fake_A, fake_B = trainer.training_step(data_batch)
        
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - G_loss {gen_loss:.4f} - D_loss {disc_loss:.4f}')
        gl.append(gen_loss)
        dl.append(disc_loss)

    print(f'=> Average G-loss: {np.nanmean(gl)}, Average D-loss: {np.nanmean(dl)}')

    if (epoch + 1) % 10 == 0:
        A = list((fake_A*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8))
        B = list((fake_B*0.5+0.5).detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8))
        tiled = imgviz.tile(A+B, shape=(2,1), border=(255,0,0))
        save_image(fake_A*0.5+0.5, f"{config['log_directory']}horse_{epoch}.png")
        save_image(fake_B*0.5+0.5, f"{config['log_directory']}zebra_{epoch}.png")

        if config['LOG_WANDB']:
            wandb.log({'Generations': wandb.Image(tiled)}, step=epoch+1)
        
    if config['LOG_WANDB']:
        wandb.log({"G_Loss": np.nanmean(gl), "D_Loss": np.nanmean(dl)}, step=epoch+1)

if config['LOG_WANDB']:
    wandb.run.finish()
#%%