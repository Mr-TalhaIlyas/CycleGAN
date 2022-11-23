#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:57:52 2022

@author: user01
"""



class CycleGAN_LR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# gen_lr_scheduler = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=CycleGAN_LR(
#                         config['epochs'], config['start_epoch'], config['decay_epoch']).step)