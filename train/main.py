import os, sys
import math

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, accuracy, distribute_bn

from data import create_dataloader
from logger import MetricLogger, SmoothedValue, WandBLogger
from utils import init_distributed_mode, fix_random_seed

import torchvision.transforms as transforms
import random
# from tool.pvt_v2 import pvt_v2_b0


torch.backends.cudnn.benchmark = True


def train_one_epoch(cfg, epoch, dataloader, mixup_fn, model, criterion, optimizer, loss_scaler, lr_scheduler, n_iter):
    model.train()
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    trainlogger = MetricLogger(delimiter=' ')
    trainlogger.add_meter('train_loss', SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    trainlogger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))

    header = f'Epoch: [{epoch:03}/{cfg.epochs:03}]'
    print_freq = cfg.logger.print_iter_freq
    if cfg.gaus_filter.use:

        transform =  transforms.Compose([
            transforms.GaussianBlur(kernel_size=3)
            ])

    for idx, data in enumerate(trainlogger.log_every(dataloader, n_iter, print_freq, header)):
        
        if cfg.gaus_filter.use and cfg.gaus_filter.use <  random.uniform(0, 1):
            data[0] =  transform(data[0])

        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            train_loss = criterion(outputs, labels)

        loss_value = train_loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(
            loss=train_loss,
            optimizer=optimizer,
            clip_grad=cfg.model.scaler.clip_grad,
            clip_mode=cfg.model.scaler.clip_mode,
            parameters=model.parameters(),
            create_graph=is_second_order
        )

        if not cfg.model.scheduler.args.step_per_epoch:
            lr_scheduler.step_update(epoch * len(dataloader) + idx)

        torch.cuda.synchronize()
        trainlogger.update(train_loss=loss_value)
        trainlogger.update(lr=optimizer.param_groups[0]['lr'])

    trainlogger.synchronize_between_processes()
    print(f'Averaged stats: {trainlogger}')
    return {f'{k}': meter.global_avg for k, meter in trainlogger.meters.items()}


@torch.no_grad()
def evaluate(dataloader, model, n_iter):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model.eval()

    evalloagger = MetricLogger(delimiter=' ')
    evalloagger.add_meter('eval_loss', SmoothedValue(window_size=1, fmt='{global_avg:.4f}'))
    evalloagger.add_meter('eval_acc1', SmoothedValue(window_size=1, fmt='{value:.3f}'))
    
    header = 'Val:'
    for data in evalloagger.log_every(dataloader, n_iter, 10, header):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)

        eval_loss = criterion(outputs, labels)
        acc1, _ = accuracy(outputs, labels, topk=(1, 5))

        torch.cuda.synchronize()
        batch_size = images.size(0)
        evalloagger.update(eval_loss=eval_loss.item())
        evalloagger.update(eval_acc1=acc1.item(), n=batch_size)

    evalloagger.synchronize_between_processes()
    print('* Acc@1: {top1.global_avg:.3f} Eval loss: {losses.global_avg:.3f}'.format(top1=evalloagger.eval_acc1, losses=evalloagger.eval_loss))
    return {f'{k}': meter.global_avg for k, meter in evalloagger.meters.items()}


@hydra.main(config_path='./configs', config_name='main')
def main(cfg):
    # Initialize torch.distributed using MPI
    init_distributed_mode(cfg.dist)

    # Fix random seed
    if cfg.seed != -1:
        fix_random_seed(cfg.seed + dist.get_rank())

    # Create logger
    logger = None
    if dist.get_rank() == 0:
        print(cfg)
        logger = WandBLogger(cfg)

    # Create Dataloader
    world_size = dist.get_world_size()
    total_batch_size = cfg.data.loader.batch_size * world_size
    trainloader = create_dataloader(cfg.data)
    n_train_iter = cfg.data.baseinfo.train_imgs // total_batch_size
    if cfg.mode == 'finetune':
        valloader = create_dataloader(cfg.data, is_training=False)
        n_val_iter = cfg.data.baseinfo.val_imgs // total_batch_size

    # Setup mixup / cutmix
    mixup_fn = None
    mixup_enable = (cfg.data.mixup.mixup_alpha > 0.) or (cfg.data.mixup.cutmix_alpha > 0.)
    if mixup_enable:
        mixup_fn = instantiate(cfg.data.mixup, num_classes=cfg.data.baseinfo.num_classes)
        print(f'MixUp/Cutmix was enabled\n')

    # create model
    if cfg.model.arch.model_name == "pvt_v2_b0":
        model = pvtv2_models.__dict__["pvt_v2_b0"](img_size=224, num_classes=cfg.data.baseinfo.num_classes,
                                           drop_path_rate=0.1)
        # model = pvt_v2_b0(pretrained=False, num_classes=cfg.data.baseinfo.num_classes)
    else:
        model = instantiate(cfg.model.arch, num_classes=cfg.data.baseinfo.num_classes)
    print(f'Model[{cfg.model.arch.model_name}] was created')
    if logger is not None:
        logger.save_architecture(model)
    
    if 'resnet' in cfg.model.arch.model_name:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('BatchNorm converted to SyncBatchNorm')

    # load pretrained weights
    if cfg.mode == 'finetune':
        if cfg.ckpt is not None:
            ckpt = torch.load(cfg.ckpt, map_location='cpu')
            print(ckpt.keys())
            if cfg.ckpt.endswith('.pth'):
                # ckpt_model = ckpt
                ckpt_model = ckpt['model']
                state_dict = model.state_dict()
            elif cfg.ckpt.endswith('.torch'):
                ckpt_model = ckpt['classy_state_dict']['base_model']['model']['trunk']
                ckpt_model['cls_token'] = ckpt_model.pop('class_token')
                ckpt_model['pos_embed'] = ckpt_model.pop('pos_embedding')
                state_dict = model.state_dict()
                
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
                if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                    print(f'Remove key [{k}] from pretrained checkpoint')
                    del ckpt_model[k]
            model.load_state_dict(ckpt_model, strict=False)
            print(f'Checkpoint was loaded from {cfg.ckpt}\n')
        else:
            print(f'Model[{cfg.model.arch.model_name}] will be trained from scratch\n')      



    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.dist.local_rank])
    model_without_ddp = model.module
    
    # optimizer
    scaled_lr = cfg.model.optim.learning_rate * cfg.data.loader.batch_size * world_size / 512.0
    scaled_warmup_lr = cfg.model.scheduler.args.warmup_lr * cfg.data.loader.batch_size * world_size / 512.0
    scaled_min_lr = cfg.model.scheduler.args.min_lr * cfg.data.loader.batch_size * world_size / 512.0
    cfg.model.optim.learning_rate = scaled_lr
    cfg.model.scheduler.args.warmup_lr = scaled_warmup_lr
    cfg.model.scheduler.args.min_lr = scaled_min_lr
    optimizer = instantiate(cfg.model.optim, model=model)
    print(f'Optimizer: \n{optimizer}\n')
    
    # scheduler
    if cfg.model.scheduler.args.step_per_epoch:
        lr_scheduler, _ = instantiate(cfg.model.scheduler, optimizer=optimizer)
    else:
        lr_scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer, n_iter_per_epoch=len(trainloader))
    print(f'Scheduler: \n{lr_scheduler}\n')

    # criterion
    if cfg.data.mixup.mixup_alpha > 0.:
        criterion = SoftTargetCrossEntropy().cuda()
        print('SoftTargetCrossEntropy is used for criterion\n')
    elif cfg.data.mixup.label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(cfg.data.mixup.label_smoothing).cuda()
        print('LabelSmoothingCrossEntropy is used for criterion\n')
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
        print('CrossEntropyLoss is used for criterion\n')
    loss_scaler = NativeScaler()

    # Load resume
    start_epoch = 1
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume was loaded from {cfg.resume}\n')
        print(f'contine training from {start_epoch}')

    if cfg.mode == 'finetune':
        max_accuracy = 0.0

    print('Start training')
    for epoch in range(start_epoch, cfg.epochs + 1):
        trainloader.sampler.set_epoch(epoch)

        log_stats = train_one_epoch(
            cfg,
            epoch,
            trainloader,
            mixup_fn,
            model,
            criterion,
            optimizer,
            loss_scaler,
            lr_scheduler,
            n_train_iter
        )

        if 'resnet' in cfg.model.arch.model_name:
            distribute_bn(model, dist.get_world_size(), reduce=True)

        if cfg.model.scheduler.args.step_per_epoch:
            lr_scheduler.step(epoch)

        if cfg.mode == 'finetune':
            eval_stats = evaluate(valloader, model, n_val_iter)
            log_stats.update(**eval_stats)
            max_accuracy = max(max_accuracy, eval_stats['eval_acc1'])

        if logger is not None:
            logger.log_items(log_stats, epoch)

        if dist.get_rank() == 0 and epoch % cfg.logger.save_epoch_freq == 0:
            save_path = f'{cfg.model.arch.model_name}_{cfg.data.baseinfo.name}_{epoch:03}ep.pth'
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'epoch': epoch
            }, save_path)
            
    if dist.get_rank() == 0:
        save_path = f'{cfg.model.arch.model_name}_{cfg.data.baseinfo.name}_{epoch:03}ep.pth'
        torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                    'epoch': epoch
                }, save_path)

    if cfg.mode == 'finetune':
        print(f'* Max Acc@1: {max_accuracy:.3f}')

    if logger is not None:
        logger.finish()


if __name__ == '__main__':
    main()
