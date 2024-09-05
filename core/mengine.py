import os
import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from toolkit.cmetric import MultiClassificationMetric, MultilabelClassificationMetric, simple_accuracy
from toolkit.chelper import load_model
from torch import distributed as dist
from sklearn.metrics import roc_auc_score
import numpy as np
import time


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def gather_tensor(tensor, n):
    rt = [torch.zeros_like(tensor) for _ in range(n)]
    dist.all_gather(rt, tensor)
    return torch.cat(rt, dim=0)


class TrainEngine(object):
    def __init__(self, local_rank, world_size=0, DDP=False, SyncBatchNorm=False):
        # init setting
        self.local_rank = local_rank
        self.world_size = world_size
        self.device_ = f'cuda:{local_rank}'
        # create tool
        self.cls_meter_ = MultilabelClassificationMetric()
        self.loss_meter_ = MultiClassificationMetric()
        self.top1_meter_ = MultiClassificationMetric()
        self.DDP = DDP
        self.SyncBN = SyncBatchNorm

    def create_env(self, cfg):
        # create network
        self.netloc_ = load_model(cfg.network.name, cfg.network.class_num, self.SyncBN)
        print(self.netloc_)

        self.netloc_.cuda()
        if self.DDP:
            if self.SyncBN:
                self.netloc_ = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.netloc_)
            self.netloc_ = DDP(self.netloc_,
                               device_ids=[self.local_rank],
                               broadcast_buffers=True,
                               )

        # create loss function
        self.criterion_ = nn.CrossEntropyLoss().cuda()

        # create optimizer
        self.optimizer_ = torch.optim.AdamW(self.netloc_.parameters(), lr=cfg.optimizer.lr,
                                                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2), eps=cfg.optimizer.eps,
                                                weight_decay=cfg.optimizer.weight_decay)

        # create scheduler
        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_, cfg.train.epoch_num,
                                                                         eta_min=cfg.scheduler.min_lr)

    def train_multi_class(self, train_loader, epoch_idx, ema_start):
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.train()
        self.loss_meter_.reset()
        self.top1_meter_.reset()
        # train
        train_loader = tqdm(train_loader, desc='train', ascii=True)
        for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(train_loader):
            # set cuda
            imgs_tensor = imgs_tensor.cuda()  # [256, 3, 224, 224]
            imgs_label = imgs_label.cuda()
            # clear gradients(zero the parameter gradients)
            self.optimizer_.zero_grad()
            # calc forward
            preds = self.netloc_(imgs_tensor)
            # calc acc & loss
            loss = self.criterion_(preds, imgs_label)

            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer_.step()

            # EMA update
            if ema_start:
                self.ema_model.update(self.netloc_)

            # accumulate loss & acc
            acc1 = simple_accuracy(preds, imgs_label)
            if self.DDP:
                loss = reduce_tensor(loss, self.world_size)
                acc1 = reduce_tensor(acc1, self.world_size)
            self.loss_meter_.update(loss.data.item())
            self.top1_meter_.update(acc1.item())

        # eval
        top1 = self.top1_meter_.mean
        loss = self.loss_meter_.mean
        endtime = datetime.datetime.now()
        self.lr_ = self.optimizer_.param_groups[0]['lr']
        if self.local_rank == 0:
            print('log: epoch-%d, train_top1 is %f, train_loss is %f, lr is %f, time is %d' % (
            epoch_idx, top1, loss, self.lr_, (endtime - starttime).seconds))
        # return
        return top1, loss, self.lr_

    def val_multi_class(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.eval()
        self.loss_meter_.reset()
        self.top1_meter_.reset()
        self.all_probs = []
        self.all_labels = []
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda()
                imgs_label = imgs_label.cuda()
                # calc forward
                preds = self.netloc_(imgs_tensor)
                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                acc1 = simple_accuracy(preds, imgs_label)

                outputs_scores = nn.functional.softmax(preds, dim=1)
                outputs_scores = torch.cat((outputs_scores, imgs_label.unsqueeze(-1)), dim=-1)

                if self.DDP:
                    loss = reduce_tensor(loss, self.world_size)
                    acc1 = reduce_tensor(acc1, self.world_size)
                    outputs_scores = gather_tensor(outputs_scores, self.world_size)

                outputs_scores, label = outputs_scores[:, -2], outputs_scores[:, -1] 
                self.all_probs += [float(i) for i in outputs_scores]
                self.all_labels += [ float(i) for i in label]
                self.loss_meter_.update(loss.item())
                self.top1_meter_.update(acc1.item())
        # eval
        top1 = self.top1_meter_.mean
        loss = self.loss_meter_.mean
        auc = roc_auc_score(self.all_labels, self.all_probs)

        endtime = datetime.datetime.now()
        if self.local_rank == 0:
            print('log: epoch-%d, val_top1   is %f, val_loss   is %f, auc is %f, time is %d' % (
            epoch_idx, top1, loss, auc, (endtime - starttime).seconds))

        # update lr
        self.scheduler_.step()

        # return
        return top1, loss, auc

    def val_ema(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.ema_model.module.eval()
        self.loss_meter_.reset()
        self.top1_meter_.reset()
        self.all_probs = []
        self.all_labels = []
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda()
                imgs_label = imgs_label.cuda()
                # calc forward
                preds = self.ema_model.module(imgs_tensor)

                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                acc1 = simple_accuracy(preds, imgs_label)

                outputs_scores = nn.functional.softmax(preds, dim=1)
                outputs_scores = torch.cat((outputs_scores, imgs_label.unsqueeze(-1)), dim=-1)

                if self.DDP:
                    loss = reduce_tensor(loss, self.world_size)
                    acc1 = reduce_tensor(acc1, self.world_size)
                    outputs_scores = gather_tensor(outputs_scores, self.world_size)

                outputs_scores, label = outputs_scores[:, -2], outputs_scores[:, -1]
                self.all_probs += [float(i) for i in outputs_scores]
                self.all_labels += [ float(i) for i in label]
                self.loss_meter_.update(loss.item())
                self.top1_meter_.update(acc1.item())
        # eval
        top1 = self.top1_meter_.mean
        loss = self.loss_meter_.mean
        auc = roc_auc_score(self.all_labels, self.all_probs)

        endtime = datetime.datetime.now()
        if self.local_rank == 0:
            print('log: epoch-%d, ema_val_top1   is %f, ema_val_loss   is %f, ema_auc is %f, time is %d' % (
            epoch_idx, top1, loss, auc, (endtime - starttime).seconds))

        # return
        return top1, loss, auc

    def save_checkpoint(self, file_root, epoch_idx, train_map, val_map, ema_start):

        file_name = os.path.join(file_root,
                                 time.strftime('%Y%m%d-%H-%M', time.localtime()) + '-' + str(epoch_idx) + '.pth')

        if self.DDP:
            stact_dict = self.netloc_.module.state_dict()
        else:
            stact_dict = self.netloc_.state_dict()

        torch.save(
            {
                'epoch_idx': epoch_idx,
                'state_dict': stact_dict,
                'train_map': train_map,
                'val_map': val_map,
                'lr': self.lr_,
                'optimizer': self.optimizer_.state_dict(),
                'scheduler': self.scheduler_.state_dict()
            }, file_name)

        if ema_start:
            ema_file_name = os.path.join(file_root,
                                                     time.strftime('%Y%m%d-%H-%M', time.localtime()) + '-EMA-' + str(epoch_idx) + '.pth')
            ema_stact_dict = self.ema_model.module.module.state_dict()
            torch.save(
                {
                    'epoch_idx': epoch_idx,
                    'state_dict': ema_stact_dict,
                    'train_map': train_map,
                    'val_map': val_map,
                    'lr': self.lr_,
                    'optimizer': self.optimizer_.state_dict(),
                    'scheduler': self.scheduler_.state_dict()
                }, ema_file_name)
