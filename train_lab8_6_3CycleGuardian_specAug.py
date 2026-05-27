#!/usr/bin/python






import argparse
import os

import torch
import torch.nn  as nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler

import matplotlib
matplotlib.use('Agg')

# load external modules

import  operator
from  copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from  torch.utils.tensorboard import SummaryWriter
from  torchvision import  transforms
#from config.augmentation import SpecAugment
try:
    from timm.utils.model_ema import ModelEmaV2, ModelEmaV3
except ImportError:
    # timm<=0.4.x provides ModelEmaV2 but not ModelEmaV3
    from timm.utils.model_ema import ModelEmaV2

    ModelEmaV3 = ModelEmaV2

from ICBHIDataset_v5_1 import *
from nets.CycleGuardian_v5_1_3 import group_uni_net as create_model
from  nets.CycleGuardian_v5_1_3 import  Projector, GroupMixConLoss



#lab1,  使用 cnn, + vit,  vit 替代gru 的原因是， gru 各组之间会受到时序信息的影响；

print ("Train import done successfully")
# input argmuments
parser = argparse.ArgumentParser(description='Temporal_convolution_transformer: Lung Sound Classification')
parser.add_argument('--lr_h', default=1e-2, type=float, help='High learning rate')
parser.add_argument('--lr_m', default=1e-3, type=float, help='middle learning rate')
parser.add_argument('--lr_l', default=5e-4, type=float, help='low learning rate')


parser.add_argument('--weight_decay', default=0.0005,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')

parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--event_dir', type=str, default= './data/events')
parser.add_argument('--split_method', default=0, type=int, help='0: official 6-4 split; 1: five folds split, 2: random 8-2 split')

# if the official 6-4 split, provide the  train- test  split file;
parser.add_argument('--dataset_split_file', type=str, default=None)

# if  tqwt 3 componment ,use following parameter
parser.add_argument('--train_dir', type=str, help='data directory')
parser.add_argument('--test_dir', type=str, help='data directory')

# if 5 fold split , provide  5 folds split file.
parser.add_argument('--folds_file', type=str, help='folds text file')
parser.add_argument('--test_fold', default=4, type=int, help='Test Fold ID')


parser.add_argument('--aug_scale', default=None, type=int, help='Augmentation multiplier')
parser.add_argument('--specaug_policy', default='icbhi_ast_sup', type=str, help='policy for spec augemnt')
parser.add_argument('--specaug_mask', default='mean', type=str, help='spec aug mask value', choices=['mean', 'zero'])
parser.add_argument('--model_path', type=str, default='./models_out', help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')
parser.add_argument('--stetho_id', default=-1, type=int, help='Stethoscope device id')
parser.add_argument("--annealing_epoch", type=int, default=50)

# Optimizer / scheduler techniques (ported from AST+SAM experiments)
parser.add_argument('--use_sam', action='store_true', help='Use SAM optimizer')
parser.add_argument('--use_asam', action='store_true', help='Use ASAM optimizer (adaptive SAM)')
parser.add_argument('--asam_rho', type=float, default=0.05, help='(A)SAM rho')
parser.add_argument('--asam_adaptive', action='store_true', help='Enable adaptive perturbation scaling (ASAM)')

parser.add_argument('--use_llrd', action='store_true', help='Enable simple LLRD-style param groups (lr_h/lr_m/lr_l)')

parser.add_argument('--lr_scheduler', type=str, default='auto',
                    choices=['auto', 'step', 'multistep', 'cosine_restart'],
                    help='LR scheduler policy')
parser.add_argument('--cosine_t0', type=int, default=10, help='CosineAnnealingWarmRestarts T_0')
parser.add_argument('--cosine_tmult', type=int, default=2, help='CosineAnnealingWarmRestarts T_mult')
parser.add_argument('--cosine_eta_min', type=float, default=0.0, help='CosineAnnealingWarmRestarts eta_min')

# Imbalance handling (no-GAN alternative)
parser.add_argument('--use_weighted_sampler', action='store_true',
                    help='Use WeightedRandomSampler based on training labels')
parser.add_argument('--sampler_normal_boost', type=float, default=1.0,
                    help='Multiply sampling weight of the normal class when using --use_weighted_sampler')
parser.add_argument('--cls_loss', type=str, default='ce', choices=['ce', 'focal'],
                    help='Classification loss for class_out')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
parser.add_argument('--focal_alpha_mode', type=str, default='normal_boost',
                    choices=['uniform', 'inv_freq', 'inv_freq_damp', 'normal_boost'],
                    help='How to build focal alpha weights')
parser.add_argument('--focal_alpha_dampening', type=float, default=0.5,
                    help='If focal_alpha_mode=inv_freq_damp, use (1/count)^dampening')
parser.add_argument('--focal_normal_boost', type=float, default=1.5,
                    help='If focal_alpha_mode=normal_boost, multiply class-0 alpha by this then renormalize')

# Normal-threshold technique (treat class-0 as positive only if p0>=T)
parser.add_argument('--normal_threshold', type=float, default=None,
                    help='If set, override argmax: predict Normal when p0>=T else best abnormal')
parser.add_argument('--sweep_normal_threshold', action='store_true',
                    help='Sweep normal threshold on validation set to maximize Score')
parser.add_argument('--normal_threshold_min', type=float, default=0.05)
parser.add_argument('--normal_threshold_max', type=float, default=0.50)
parser.add_argument('--normal_threshold_step', type=float, default=0.05)

# LR schedule (MultiStep milestones are tuned for very long runs; fall back for short runs)
parser.add_argument('--lr_step_size', type=int, default=5, help='StepLR step_size (used when falling back)')
parser.add_argument('--lr_gamma', type=float, default=0.4, help='LR decay gamma (used when falling back)')

# Checkpoint selection
parser.add_argument('--save_metric', type=str, default='score', choices=['score', 'acc'],
                    help='Which metric to use for "best" checkpoint')

# Fast debug/ablation runs (useful on CPU)
parser.add_argument('--max_train_batches', type=int, default=None,
                    help='If set, limit number of training batches per epoch (for quick experiments)')
parser.add_argument('--max_eval_batches', type=int, default=None,
                    help='If set, limit number of evaluation batches (for quick experiments)')
parser.add_argument('--shuffle_eval', action='store_true',
                    help='Shuffle the validation dataloader (recommended when using --max_eval_batches)')



# use for  constrative learning
# 设置用于 对比学习的参数，  temperature,  args.target_type= grad_flow,  p_cl;
parser.add_argument('--proj_dim', type=int, default=768)
parser.add_argument('--negative_pair', type=str, default='all',
                    help='the method for selecting negative pair', choices=['all', 'diff_label'])

parser.add_argument('--mix_beta', default=1.0, type=float,    #用于生成  设置每个样本 group 的混合比例
                    help='patch-mix interpolation coefficient')
parser.add_argument('--temperature', type=float, default=0.06)

# use  for  different loss  part  weight,
parser.add_argument('--p_cluster', default=  1.0 , type=float, help='loss weight  for  the global cluster loss')
parser.add_argument('--p_cos_sim', default= 0.2 , type=float, help='loss weight  for  the cos sim loss')
parser.add_argument('--p_contra',  default= 0.500, type=float,)
parser.add_argument('--p_class', default= 1.0, type=float, help='loss weight  for  the global fusion loss')



parser.add_argument('--target_type', type=str, default='project_flow',
                    help='how to make target representation',
                    choices=['grad_block', 'grad_flow', 'project_block', 'project_flow'])

args = parser.parse_args()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


def _build_param_groups_llrd(net: nn.Module, args) -> list:
    high, mid, low = [], [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if any(k in lname for k in ['classifier', 'class', 'fc', 'head', 'out', 'proj', 'projector']):
            high.append(param)
        elif any(k in lname for k in ['transformer', 'encoder', 'vit', 'attn', 'block']):
            mid.append(param)
        else:
            low.append(param)

    groups = []
    if high:
        groups.append({'params': high, 'lr': float(args.lr_h)})
    if mid:
        groups.append({'params': mid, 'lr': float(args.lr_m)})
    if low:
        groups.append({'params': low, 'lr': float(args.lr_l)})
    return groups


def _predict_with_normal_threshold(logits: torch.Tensor, normal_threshold: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    p0 = probs[:, 0]
    abnormal = probs[:, 1:]
    abnormal_argmax = torch.argmax(abnormal, dim=1) + 1
    return torch.where(p0 >= float(normal_threshold), torch.zeros_like(abnormal_argmax), abnormal_argmax)


def _score_from_preds(labels_np: np.ndarray, preds_np: np.ndarray):
    labels_np = labels_np.astype(np.int64)
    preds_np = preds_np.astype(np.int64)
    class_counts = (np.bincount(labels_np, minlength=4).astype(np.float64) + 1e-7).tolist()
    class_hits = [0.0, 0.0, 0.0, 0.0]
    for k in range(4):
        class_hits[k] = float(np.sum((labels_np == k) & (preds_np == k)))

    Sp = class_hits[0] / class_counts[0]
    Se = (class_hits[1] + class_hits[2] + class_hits[3]) / (class_counts[1] + class_counts[2] + class_counts[3])
    Sc = (Se + Sp) / 2.0
    Acc = float(np.mean(labels_np == preds_np))
    return class_hits, class_counts, float(Sp), float(Se), float(Sc), Acc


def _sweep_normal_threshold(logits: torch.Tensor, labels: torch.Tensor, args):
    best_t = float(args.normal_threshold_min)
    best = None
    t = float(args.normal_threshold_min)
    while t <= float(args.normal_threshold_max) + 1e-12:
        preds = _predict_with_normal_threshold(logits, t)
        _, _, Sp, Se, Sc, Acc = _score_from_preds(labels.cpu().numpy(), preds.cpu().numpy())
        if best is None or Sc > best[2]:
            best = (Sp, Se, Sc, Acc)
            best_t = t
        t += float(args.normal_threshold_step)

    Sp, Se, Sc, Acc = best
    return best_t, Sp, Se, Sc, Acc


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha[targets] * (1 - pt) ** self.gamma * ce
        return focal.mean()

################################MIXUP#####################################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##############################################################################
#@torch.compile
def get_score(hits, counts, pflag=False):
    eps = 1e-10
    sp = hits[0] / (counts[0] + eps)
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3] + eps)
    sc = (se+sp) / 2.0

    # normal accuracy
    int_sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    int_se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    int_sc = (int_sp + int_se) / 2.0


    if pflag:
        print("*************The official Metrics******************")
        print("The frac format Sp: {}, Se: {}, Score: {}".format(sp, se, sc))
        print("The int  format S_p: {}, S_e: {}, Score: {}".format(int_sp, int_se, int_sc))

        print("Normal: {}, Crackle: {}, Wheeze: {}, Both: {} \n ".format(
            hits[0] / (counts[0] + eps),
            hits[1] / (counts[1] + eps),
            hits[2] / (counts[2] + eps),
            hits[3] / (counts[3] + eps),
        ))




from itertools import combinations
#@torch.compile
def cos_sim_loss_fun_v2(vec_a, vec_b, vec_c, vec_d, vec_e):
    vectors = [vec_a, vec_b, vec_c, vec_d, vec_e]
    cos_loss = 0
    # Iterate over all unique pairs of vectors
    for vec1, vec2 in combinations(vectors, 2):
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
        cos_loss += torch.mean(torch.abs(cos_sim))

    return cos_loss


# @torch.compile
def kld_loss_function(q, p):
    kld = p * torch.log(p / (q+1e-10))
    return kld.sum()



class Trainer:
    def __init__(self):
        self.args = args

        if self.args.split_method  == 0:
            print(" this modeling  on  the  official split methon  6-4 split: \n ", )

        elif self.args.split_method  == 1:
            print(" this modeling  on  the  Five  folds  split: \n ", )
        else:
            print(" this modeling  on  the  random 8-2  split: \n ", )


        self.writter_train = SummaryWriter("runs_log/Train")
        self.writter_vaild = SummaryWriter("runs_log/Vaild")


        # args.h, args.w = 798, 128
        # args.resz = 1
        # train_transform = [transforms.ToTensor(),
        #                    SpecAugment(args),
        #                    transforms.Resize(antialias=True,size=(int(args.h * args.resz), int(args.w * args.resz)))]
        # val_transform = [transforms.ToTensor(),
        #                  transforms.Resize(antialias=True,size=(int(args.h * args.resz), int(args.w * args.resz)))]
        # # train_transform.append(transforms.Normalize(mean=mean, std=std))
        # # val_transform.append(transforms.Normalize(mean=mean, std=std))
        #
        # train_transform = transforms.Compose(train_transform)
        # val_transform = transforms.Compose(val_transform)

        train_dataset = ICBHIDataset_with_event(data_dir=self.args.data_dir,
                                     event_data_dir= self.args.event_dir,
                                     dataset_split=self.args.split_method,
                                     dataset_split_file=self.args.dataset_split_file,
                                     test_fold=self.args.test_fold, stetho_id=-1,
                                     train_flag=True,
                                     aug_audio= False, aug_audio_scale=1, aug_feature=False,
                                     desired_time=8, sample_rate1= 22000, sample_rate2=22000,
                                     n_filters1=84, n_filters2=42,
                                     input_transform=None,  # train_transform,
                                     )

        test_dataset = ICBHIDataset_with_event(data_dir=self.args.data_dir,
                                    event_data_dir=self.args.event_dir,
                                    dataset_split=self.args.split_method,
                                    dataset_split_file=self.args.dataset_split_file,
                                    test_fold=self.args.test_fold, stetho_id=-1,
                                    train_flag=False,
                                    aug_audio=False, aug_audio_scale=1, aug_feature=False,
                                    desired_time=8, sample_rate1= 22000, sample_rate2=22000,
                                    n_filters1=84, n_filters2=42,
                                    input_transform=None,  # val_transform,
                                    )

        self.test_ids = np.array(test_dataset.identifiers)
        self.test_paths = test_dataset.filenames_with_labels



        # loading checkpoint
        self.net = create_model(num_classes=4, mix_beta=self.args.mix_beta)  # .cuda()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.s20_projector = Projector(128, 128).to(device)

        # note,  初始化EmaV2 模型, 并将其移动到与model 同一个设备上；
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        self.net_copy = self.custom_deepcopy(self.net)
        self.ema = ModelEmaV3(self.net_copy, decay=0.1)
        self.ema.module.to(device)
        self.net.to(device)


        if self.args.checkpoint is not None:
            checkpoint = torch.load(self.args.checkpoint)
            self.net.load_state_dict(checkpoint)
            # uncomment in case fine-tuning, specify block layer
            # before block_layer, all layers will be frozen durin training
            # self.net.fine_tune(block_layer=5)
            print("Pre-trained Model Loaded:", self.args.checkpoint)
        if device.type == 'cuda':
            self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)

        # Weighted sampler (optional)
        sampler = None
        shuffle = True
        if getattr(self.args, 'use_weighted_sampler', False):
            labels = np.asarray(getattr(train_dataset, 'labels'))
            class_counts = np.bincount(labels, minlength=4).astype(np.float64)
            class_weights = 1.0 / np.maximum(class_counts, 1.0)
            normal_boost = float(getattr(self.args, 'sampler_normal_boost', 1.0))
            if normal_boost != 1.0:
                class_weights[0] = class_weights[0] * normal_boost
            sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
            shuffle = False
            print(f"[Sampler] Using WeightedRandomSampler with counts={class_counts.tolist()} (normal_boost={normal_boost})")

        # dataLoader 　用于一次取出batch size 个数据，送到网络中，　
        # 注意 如果指定sampler，　则表明使用这种规则的方式获取样本的索引，　则此时，　shuffle 使用默认值 False;
        # shuffle 为False 时，且没有指定sampler时，　按照顺序采样样本；　
        self.train_data_loader = DataLoader(train_dataset, num_workers=self.args.num_worker,
            batch_size=self.args.batch_size, sampler=sampler, shuffle=shuffle)
        val_shuffle = bool(getattr(self.args, 'shuffle_eval', False) or getattr(self.args, 'max_eval_batches', None) is not None)
        self.val_data_loader = DataLoader(test_dataset, num_workers=self.args.num_worker,
            batch_size=self.args.batch_size, shuffle=val_shuffle)
        print("DATA LOADED")

        # Optimizer (optionally SAM/ASAM + LLRD param groups)
        if getattr(self.args, 'use_llrd', False):
            param_groups = _build_param_groups_llrd(self.net, self.args)
            print(f"[LLRD] Enabled param groups: {len(param_groups)}")
        else:
            params_to_update = []
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
            param_groups = [{'params': params_to_update, 'lr': float(self.args.lr_h)}]

        use_sam = bool(getattr(self.args, 'use_sam', False) or getattr(self.args, 'use_asam', False))
        if use_sam:
            adaptive = bool(getattr(self.args, 'asam_adaptive', False) or getattr(self.args, 'use_asam', False))
            self.optimizer = SAM(
                param_groups,
                optim.Adam,
                lr=float(self.args.lr_h),
                rho=float(getattr(self.args, 'asam_rho', 0.05)),
                adaptive=adaptive,
                weight_decay=float(getattr(self.args, 'weight_decay', 0.0005)),
            )
            print(f"[Opt] Using {'ASAM' if adaptive else 'SAM'}(rho={getattr(self.args, 'asam_rho', 0.05)})")
        else:
            self.optimizer = optim.Adam(
                param_groups,
                lr=float(self.args.lr_h),
                weight_decay=float(getattr(self.args, 'weight_decay', 0.0005)),
            )
        # self.cl_optimizer = optim.Adam(cl_params, lr=self.args.lr_l)
        # self.cl_optimizer = optim.SGD(params_to_update, lr=self.args.lr_l, momentum=0.9, weight_decay=self.args.weight_decay)

        # LR scheduler
        optim_for_sched = self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer
        milestones = [200, 350, 450, 550]
        sched_policy = getattr(self.args, 'lr_scheduler', 'auto')
        if sched_policy == 'cosine_restart':
            self.exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optim_for_sched,
                T_0=int(getattr(self.args, 'cosine_t0', 10)),
                T_mult=int(getattr(self.args, 'cosine_tmult', 2)),
                eta_min=float(getattr(self.args, 'cosine_eta_min', 0.0)),
            )
            print(f"[LR] Using CosineAnnealingWarmRestarts(T_0={self.args.cosine_t0}, T_mult={self.args.cosine_tmult})")
        elif sched_policy == 'step' or (sched_policy == 'auto' and int(self.args.epochs) < min(milestones)):
            self.exp_lr_scheduler = lr_scheduler.StepLR(
                optim_for_sched,
                step_size=int(getattr(self.args, 'lr_step_size', 5)),
                gamma=float(getattr(self.args, 'lr_gamma', 0.4)),
            )
            print(f"[LR] Using StepLR(step_size={self.args.lr_step_size}, gamma={self.args.lr_gamma})")
        else:
            self.exp_lr_scheduler = lr_scheduler.MultiStepLR(
                optim_for_sched,
                milestones=milestones,
                gamma=0.33,
                last_epoch=-1,
            )





        # weights for the loss function
        #weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        # weights = torch.tensor(train_dataset.class_ratio, dtype=torch.float32)
        # weights = weights / weights.sum()
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        # weights = weights.cuda()
        
        # Classification loss (optional focal loss)
        if getattr(self.args, 'cls_loss', 'ce') == 'focal':
            labels = np.asarray(getattr(train_dataset, 'labels'))
            counts = np.bincount(labels, minlength=4).astype(np.float32)
            alpha_mode = getattr(self.args, 'focal_alpha_mode', 'normal_boost')
            if alpha_mode == 'inv_freq':
                alpha_w = 1.0 / np.maximum(counts, 1.0)
                alpha_w = alpha_w / alpha_w.sum()
            elif alpha_mode == 'inv_freq_damp':
                damp = float(getattr(self.args, 'focal_alpha_dampening', 0.5))
                alpha_w = (1.0 / np.maximum(counts, 1.0)) ** damp
                alpha_w = alpha_w / alpha_w.sum()
            elif alpha_mode == 'uniform':
                alpha_w = np.ones_like(counts, dtype=np.float32)
                alpha_w = alpha_w / alpha_w.sum()
            else:
                # default: mild normal-class boost (class 0)
                alpha_w = np.ones_like(counts, dtype=np.float32)
                alpha_w[0] = float(getattr(self.args, 'focal_normal_boost', 1.5))
                alpha_w = alpha_w / alpha_w.sum()

            alpha_t = torch.tensor(alpha_w, dtype=torch.float32).to(device)
            self.loss_func = FocalLoss(alpha=alpha_t, gamma=float(getattr(self.args, 'focal_gamma', 2.0))).to(device)
            print(f"[Loss] Using FocalLoss(gamma={self.args.focal_gamma}, alpha={alpha_w.tolist()})")
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=None)

        self.loss_nored = nn.CrossEntropyLoss(reduction='none')
        self.cl_criterion = GroupMixConLoss(
            temperature=self.args.temperature,
            negative_pair=self.args.negative_pair,
        ).to(self.device)



        self.mix_beta = self.args.mix_beta
        self.p_cluster =  self.args.p_cluster
        self.p_sim  = self.args.p_cos_sim
        self.p_contra = self.args.p_contra
        self.p_class  = self.args.p_class



    def custom_deepcopy(self, model):
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return  model_copy



    def train(self):
        train_losses = []
        test_losses = []

        test_acc = []
        best_acc = -1

        best_Se = -1
        best_Sp = -1
        best_Sc = -1
        best_Confusion_Matrix = []


        tb_writer = SummaryWriter()

        # 　开始一个epoch 的训练；
        for _, epoch in enumerate(range(self.args.start_epochs, self.args.epochs)):



            cla_losses = []
            cluster_losses = []
            fusion_rep_losses = []
            contra_losses = []


            losses = []
            class_hits = [0.0, 0.0, 0.0, 0.0]
            class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]


            # 以下两个参数用于计算已经遍历过了的batch上的正确率
            running_corrects = 0.0   # 已经遍历了的batch上正确数量
            denom = 0.0 # 已经遍历了的batch的样本数量

            classwise_train_losses = [[], [], [], []]  # 每个类别的损失；
                
            # 从dataloader 中读取一个batch的数据， 并且通过enumerate() 逐个取出该batch　中的每个样本到网络中；
            for i, (spec, label) in enumerate(tqdm(self.train_data_loader,  desc=' training process')):
                spec_data = spec.to(self.device).float()
                label = label.to(self.device).long()
                # label = label.long() # ;

                # in case using mixup, uncomment 2 lines below
                # image, label_a, label_b, lam = mixup_data(image, label, alpha=0.5)
                # image, label_a, label_b = map(Variable, (image, label_a, label_b))

                ori_s20_clu_loss, cluster4_fusion_vec_loss, s20_glo_vec, class_out = self.net(spec_data, group_mix=False, label=None)

                cla_loss = self.loss_func(class_out, label)  #  融合特征的分类损失；

                if args.target_type == 'grad_block':
                    proj1 = deepcopy(s20_glo_vec[0].detach())
                elif args.target_type == 'grad_flow':
                    proj1 =s20_glo_vec[0]
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(self.s20_projector(s20_glo_vec[0]).detach())
                elif args.target_type == 'project_flow':
                    proj1_s20 = self.s20_projector(s20_glo_vec)


                # mix_features  相比与原始 的features,  需要经过一个 projector() 层；
                # cl_info: list[0]:bt_label, list[1]:bt_index, [2]: lam_ratio;  [3]: mix_glo_vec
                hyb_s20_clu_loss,  hyb_cluster4_fusion_vec_loss, s20_cl_info = self.net(spec_data, group_mix=True,label=label)
                proj2_s20 = self.s20_projector(s20_cl_info[3])
                cl_s20_loss = self.cl_criterion(proj1_s20, proj2_s20, label, s20_cl_info[0], s20_cl_info[2], s20_cl_info[1])


                clu_loss =  self.p_cluster *  (ori_s20_clu_loss +  hyb_s20_clu_loss)  # 1000
                contra_loss =  self.p_contra * cl_s20_loss
                fusion_rep_loss =  self.p_sim  *  (cluster4_fusion_vec_loss + hyb_cluster4_fusion_vec_loss)   #


                loss  =  cla_loss +  clu_loss   + contra_loss  + fusion_rep_loss


                fus_nored = self.loss_nored(class_out, label)
                loss_nored =  fus_nored

                # spec_pred = torch.argmax(g_spec_out, 1)
                prob_fus,  fus_pred = torch.max(class_out, 1)
                preds = fus_pred
                
                running_corrects += torch.sum(preds == label.data)
                denom += len(label.data)


                #class 计算在训练数据中（真实值）每一类对应样本数量和每一类预测正确的样本数量
                for idx in range(preds.shape[0]):
                    class_counts[label[idx].item()] += 1.0
                    if preds[idx].item() == label[idx].item():
                         class_hits[label[idx].item()] += 1.0
                    classwise_train_losses[label[idx].item()].append(loss_nored[idx].item())

                if isinstance(self.optimizer, SAM):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    # second forward-backward step at perturbed weights
                    ori_s20_clu_loss2, cluster4_fusion_vec_loss2, s20_glo_vec2, class_out2 = self.net(spec_data, group_mix=False, label=None)
                    cla_loss2 = self.loss_func(class_out2, label)

                    proj1_s20_2 = self.s20_projector(s20_glo_vec2)
                    hyb_s20_clu_loss2, hyb_cluster4_fusion_vec_loss2, s20_cl_info2 = self.net(spec_data, group_mix=True, label=label)
                    proj2_s20_2 = self.s20_projector(s20_cl_info2[3])
                    cl_s20_loss2 = self.cl_criterion(proj1_s20_2, proj2_s20_2, label, s20_cl_info2[0], s20_cl_info2[2], s20_cl_info2[1])

                    clu_loss2 = self.p_cluster * (ori_s20_clu_loss2 + hyb_s20_clu_loss2)
                    contra_loss2 = self.p_contra * cl_s20_loss2
                    fusion_rep_loss2 = self.p_sim * (cluster4_fusion_vec_loss2 + hyb_cluster4_fusion_vec_loss2)
                    loss2 = cla_loss2 + clu_loss2 + contra_loss2 + fusion_rep_loss2
                    loss2.backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                #self.cl_optimizer.step()

                # note, 在模型完成反向传播之后使用， 这里更新ema 的模型
                self.ema.update(self.net)


                cluster_losses.append(clu_loss.data.cpu().numpy())
                fusion_rep_losses.append(fusion_rep_loss.data.cpu().numpy())
                contra_losses.append(contra_loss.data.cpu().numpy()) # 对比损失；
                cla_losses.append(cla_loss.data.cpu().numpy())  # 分类损失
                losses.append(loss.data.cpu().numpy())

                max_train_batches = getattr(self.args, 'max_train_batches', None)
                should_eval = (i == (len(self.train_data_loader) - 1)) or (
                    max_train_batches is not None and (i + 1) >= max_train_batches
                )

                if should_eval:
                    print(" \n ==================================================")
                    print("epoch {} iter {}/{} Train Total loss: {}".format(epoch,i, len(self.train_data_loader), np.mean(losses)))

                    print("Train Accuracy: {}".format(running_corrects.double() / denom))
                    print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(
                        np.mean(classwise_train_losses[0]),
                        np.mean(classwise_train_losses[1]),
                        np.mean(classwise_train_losses[2]),
                        np.mean(classwise_train_losses[3])))

                    print("\n -----show the training info -----------")
                    get_score(class_hits, class_counts, True)


                    print("testing......")
                    acc, test_loss, conf_mat, Sp, Se, Sc = self.evaluate(self.net, epoch, i)
                    self.writter_vaild.add_scalar("Sc", Sc, epoch)
                    self.writter_vaild.add_scalar("Se", Se, epoch)
                    self.writter_vaild.add_scalar("Sp", Sp, epoch)


                    save_metric = getattr(self.args, 'save_metric', 'score')
                    current_metric = Sc if save_metric == 'score' else acc
                    best_metric = best_Sc if save_metric == 'score' else best_acc

                    if current_metric > best_metric:
                        best_acc = acc
                        best_Se = Se
                        best_Sp = Sp
                        best_Sc = Sc
                        best_Confusion_Matrix = conf_mat
                        self.writter_vaild.add_scalar(f"best_{save_metric}", current_metric, epoch)

                        os.makedirs(args.model_path, exist_ok=True)
                        ckpt_name = (
                            f"lab8-6CycGuradin_epoch_{epoch}_"
                            f"Sc_{float(Sc):.4f}_Se_{float(Se):.4f}_Sp_{float(Sp):.4f}_Acc_{float(acc):.4f}.pkl"
                        )
                        state_dict = self.net.module.state_dict() if isinstance(self.net, nn.DataParallel) else self.net.state_dict()
                        torch.save(state_dict, os.path.join(args.model_path, ckpt_name))
                        print(f"Best checkpoint saved by {save_metric}: {float(current_metric):.4f}")

                    if save_metric == 'score':
                        print("BEST SCORE TILL NOW", best_Sc)
                    else:
                        print("BEST ACCURACY TILL NOW", best_acc)

                    train_losses.append(np.mean(losses))
                    test_losses.append(test_loss)
                    test_acc.append(acc)

                if max_train_batches is not None and (i + 1) >= max_train_batches:
                    break

            train_acc = running_corrects.double() / denom
            train_loss = np.mean(losses)

            tags = ["train_acc", "train_loss", "val_acc", "val_loss", ]

            self.writter_train.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            self.writter_train.add_scalar("train_acc", train_acc, epoch)
            self.writter_train.add_scalars("Train_Class_acc",
                                            {"Noraml": class_hits[0] / class_counts[0],
                                             "Crackle": class_hits[1] / class_counts[1],
                                             "Wheeze": class_hits[2] / class_counts[2],
                                             "Both": class_hits[3] / class_counts[3]},
                                            epoch)
            
            # 可视化每个epoch 上总损失 ，以及 四个分量上的损失；
            self.writter_train.add_scalar("Train_total_Loss", train_loss, epoch)
            self.writter_train.add_scalars("Train_multi_loss",
                                            {
                                                "cluster_loss": np.mean(cluster_losses),
                                                "fusion_rep_vec_loss": np.mean(fusion_rep_losses),
                                                "contrastive_loss": np.mean(contra_losses),
                                                "classification_loss": np.mean(cla_losses),
                                            },
                                            epoch)

            if getattr(self.args, 'lr_scheduler', 'auto') == 'cosine_restart':
                self.exp_lr_scheduler.step(epoch + 1)
            else:
                self.exp_lr_scheduler.step()
            print(f"best_Se{best_Se}\tbest_Sp{best_Sp}\tbest_Sc{best_Sc}\tbest_Acc{best_acc}")
            print(f"ds combine best_Confusion_matrix:\n{best_Confusion_Matrix}")


    def evaluate(self, net, epoch, iteration):

        self.ema.module.eval()
        test_losses = []

        denom = 0.0
        running_corrects = 0.0
        classwise_test_losses = [[], [], [], []]
        all_logits = []
        all_labels = []



        max_eval_batches = getattr(self.args, 'max_eval_batches', None)

        with torch.no_grad():
            # for i, (image, label) in tqdm(enumerate(self.val_data_loader)):
            for i, (spec,label) in enumerate(self.val_data_loader, ):
                spec_data = spec.to(self.device).float()
                label = label.to(self.device).long()

                ori_s20_clu_loss,  cluster4_fusion_vec_loss, s20_glo_vec, class_out = self.ema.module(spec_data,  group_mix=False,label=None)

                cla_loss = self.loss_func(class_out, label)
                clu_loss = self.p_cluster * (ori_s20_clu_loss )  # 1000
                fusion_rep_loss = self.p_sim * (cluster4_fusion_vec_loss )  #
                loss =  cla_loss +  clu_loss  + fusion_rep_loss

                fus_nored = self.loss_nored(class_out, label)
                loss_nored = fus_nored

                all_logits.append(class_out.detach().cpu())
                all_labels.append(label.detach().cpu())

                test_losses.append(loss.data.cpu().numpy())

                denom += len(label.data)
                for idx in range(label.shape[0]):
                    classwise_test_losses[label[idx].item()].append(loss_nored[idx].item())

                if max_eval_batches is not None and (i + 1) >= max_eval_batches:
                    break

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        threshold_used = getattr(self.args, 'normal_threshold', None)
        if getattr(self.args, 'sweep_normal_threshold', False):
            t, Sp_s, Se_s, Sc_s, Acc_s = _sweep_normal_threshold(all_logits, all_labels, self.args)
            threshold_used = t
            preds = _predict_with_normal_threshold(all_logits, threshold_used)
            print(f"[Threshold] Best normal-threshold on val: {threshold_used:.2f} (Score={Sc_s:.4f}, Se={Se_s:.4f}, Sp={Sp_s:.4f})")
        else:
            if threshold_used is None:
                preds = torch.argmax(all_logits, dim=1)
            else:
                preds = _predict_with_normal_threshold(all_logits, threshold_used)

        labels_np = all_labels.numpy()
        preds_np = preds.numpy()
        running_corrects = float(np.sum(labels_np == preds_np))

        print("Val Accuracy by  fusion result : {}".format(running_corrects / float(denom)))
        print("epoch {}, Validation BCE loss: {}".format(epoch, np.mean(test_losses)))
        print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(
            np.mean(classwise_test_losses[0]),
            np.mean(classwise_test_losses[1]),
            np.mean(classwise_test_losses[2]),
            np.mean(classwise_test_losses[3])))

        print("\n -----show the validation info -----------")
        class_hits, class_counts, Sp, Se, Sc, Acc = _score_from_preds(labels_np, preds_np)
        get_score(class_hits, class_counts, True)
        if threshold_used is not None:
            print(f"Normal-threshold used: {float(threshold_used):.2f}")

        self.writter_vaild.add_scalar("Acc_test", running_corrects / float(denom), epoch)
        self.writter_vaild.add_scalars("Class_acc_test",
                                      {"Noraml": class_hits[0] / class_counts[0],
                                       "Crackle": class_hits[1] / class_counts[1],
                                       "Wheeze": class_hits[2] / class_counts[2],
                                       "Both": class_hits[3] / class_counts[3]},
                                      epoch)


        self.writter_vaild.add_scalars("Class_Losss_test",
                                      {"Noraml": np.mean(classwise_test_losses[0]),
                                       "Crackle": np.mean(classwise_test_losses[1]),
                                       "Wheeze": np.mean(classwise_test_losses[2]),
                                       "Both": np.mean(classwise_test_losses[3])},
                                      epoch)

        self.writter_vaild.add_scalar("Total_Loss_test", np.mean(test_losses), epoch)
        # 验证集上， 四个部分的损失

        conf_label = labels_np.astype(np.int64)
        conf_pred = preds_np.astype(np.int64)

        # the following  code relize the for the exceed  part,
        # 以下情况是当输入超过8s的部分， 将超过的部分通过重叠的方式，重新组成一个新的样本，
        # If a cycle  exceed the 8s , like 9.6s,
        # the over part 1.6s  will  also be used padded byself  and generate the new sample,
        # and the generate new sample's label  will  use the  same label;
        # y_pred, y_true = [], []
        # for pt in self.test_paths:
        #     y_pred.append(np.argmax(np.bincount(conf_pred[np.where(self.test_ids == pt)])))
        #     y_true.append(int(pt.split('_')[-1]))

        conf_matrix = confusion_matrix(conf_label, conf_pred)
        acc = accuracy_score(conf_label, conf_pred)


        print("*************The Helper Metrics******************")
        print("Confusion Matrix \n", conf_matrix)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Classwise Scores -->: ", conf_matrix.diagonal())

        print("Accuracy Score ---> :", acc)
        print(f" Micro F1 Score: {f1_score(conf_label, conf_pred, average= 'micro')}")
        prec = precision_score(conf_label, conf_pred, average='weighted', zero_division= np.nan)
        rec = recall_score(conf_label, conf_pred, average='weighted')
        print(f" weighted Precision: {prec}")
        print(f" weighted Recall: {rec}")


        self.net.train()
        return acc, np.mean(test_losses), conf_matrix, Sp, Se, Sc,


if __name__ == "__main__":

    '''
    for test_id in range(0, 5):
        args.test_fold =  test_id
        args.epochs = 30
        args.lr = 3e-3
        args.arg_scale = 1
        args.checkpoint = './models_out/pitch_lab2_6ch_best_acc.pkl'
    '''
    trainer = Trainer()
    trainer.train()

"""
python  train_lab3_1GuardianUni.py    --data_dir ./data/ICBHI_final_database --dataset_split_file ./data/patient_trainTest6_4.txt --model_path ./models_out --lr_h 0.001 --lr_l 0.001 --batch_size 8 --num_worker 0 --start_epochs 0 --epochs 700
"""
