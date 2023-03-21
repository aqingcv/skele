#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
# import yaml
import math
import numpy as np
from optimizer import LARS
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from tqdm import tqdm


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def forward_loss(p, z): # negative cosine similarity
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

class PT_Processor(Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """
    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        print(self.model)
    def load_optimizer(self):

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'LARS':
            self.optimizer = LARS(self.model.parameters(),
                                    lr=self.arg.base_lr, 
                                    momentum=0.9, 
                                    weight_decay=self.arg.weight_decay)     
        else:
            raise ValueError()

    def adjust_lr(self):
        if  self.arg.step:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * self.meta_info['epoch'] / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))

            for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.lr = lr    

        elif self.arg.cos:  # cosine lr schedule
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * self.meta_info['epoch'] / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * 0.5 * (1. + math.cos(math.pi * (self.meta_info['epoch']-self.arg.warm_up_epoch) / (self.arg.num_epoch-self.arg.warm_up_epoch)))

            for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.lr = lr    

        else:
            self.lr = self.arg.base_lr

    def train(self, epoch):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        # loss_j_value = []
        # loss_m_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)

            label = label.long().to(self.dev, non_blocking=True)        
            p1, p2, z1, z2 =  self.model(data1, data2)
            loss = forward_loss(p1,z2)/2 + forward_loss(p2,z1)/2
            # backward
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            # self.iter_info['loss_joint'] = loss_joint.data.item()
            # self.iter_info['loss_motion'] = loss_motion.data.item()            

            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            # loss_j_value.append(self.iter_info['loss_joint'])
            # loss_m_value.append(self.iter_info['loss_motion'])            
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        # self.epoch_info['train_mean_loss_j'] = np.mean(loss_j_value)
        # self.epoch_info['train_mean_loss_m'] = np.mean(loss_m_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        # self.train_writer.add_scalar('loss_joint', self.epoch_info['train_mean_loss_j'], epoch)
        # self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_m'], epoch)

        self.show_epoch_info()


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--cos', type=int, default=0, help='use cosine lr schedule')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        return parser



