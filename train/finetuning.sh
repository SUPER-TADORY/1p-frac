#!/bin/sh

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export NGPUS=8 # Set total GPU num
export NPERNODE=4 # Set total node num

DATASET_NAME=CIFAR100 
CLASS_NUM=100 # set number of classes
TRAIN_DATADIR='/path/to/cifar100/train' # set training data path
VAL_DATADIR='/path/to/cifar100/val' # set validation data path
CKPT_PATH='/path/to/pretrained weights' # set checkpoint path
MODEL_NAME='vit_tiny_patch16_224' # set timm name model

mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=colorimagefolder \
    data.baseinfo.name=${DATASET_NAME} data.baseinfo.num_classes=${CLASS_NUM} \
    data.trainset.root=${TRAIN_DATADIR} data.baseinfo.train_imgs=50000 \
    data.valset.root=${VAL_DATADIR} data.baseinfo.val_imgs=10000 \
    data.loader.batch_size=96 \
    ckpt=${CKPT_PATH} \
    model=vit model.arch.model_name=${MODEL_NAME} \
    model.optim.optimizer_name=sgd model.optim.learning_rate=0.01 \
    model.optim.weight_decay=1.0e-4 model.scheduler.args.warmup_epochs=10 \
    epochs=1000 mode=finetune \
    logger.save_epoch_freq=100 \
    output_dir=./output/finetune

