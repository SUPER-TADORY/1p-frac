#!/bin/sh

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export NGPUS=8
export NPERNODE=4


mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=colorimagefolder \
    data.baseinfo.name=CIFAR10 data.baseinfo.num_classes=10 \
    data.trainset.root=/PATH/TO/CIFAR10/TRAIN data.baseinfo.train_imgs=50000 \
    data.valset.root=/PATH/TO/CIFAR10/VAL data.baseinfo.val_imgs=10000 \
    data.loader.batch_size=96 \
    ckpt=./output/pretrain/pretrain_deit_base_VisualAtom21000_1.0e-3/last.pth.tar \
    model=vit model.arch.model_name=vit_base_patch16_224 \
    model.optim.optimizer_name=sgd model.optim.learning_rate=0.01 \
    model.optim.weight_decay=1.0e-4 model.scheduler.args.warmup_epochs=10 \
    epochs=1000 mode=finetune \
    logger.entity=YOUR_WANDB_ENTITY_NAME logger.project=YOUR_WANDB_PROJECT_NAME logger.group=YOUR_WANDB_GROUP_NAME \
    logger.experiment=finetune_deit_base_CIFAR10_batch768_from_VisualAtom21000_1.0e-3 \
    logger.save_epoch_freq=100 \
    output_dir=./output/finetune

