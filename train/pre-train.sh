#!/bin/sh

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export NGPUS=8 # Set total GPU num
export NPERNODE=4 # Set total node num

BASE_DATADIR='/path/to/base/data/directory'
ONEPFRAC_NAME='set_dataset_name' # Set 1p-frac name you generated (ex. sigma3.5_perturb0.1_sample1000_Kift13W5)
SAMPLE_NUM=$(echo "$ONEPFRAC_NAME" | sed -E 's/.*sample([0-9]+).*/\1/')

# If the number of samples used for pre-training is 1k
mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=grayimagefolder \
                data.baseinfo.name=${ONEPFRAC_NAME} \
                data.baseinfo.train_imgs=${SAMPLE_NUM} data.baseinfo.num_classes=${SAMPLE_NUM} \
                data.trainset.root=${BASE_DATADIR} \
                data.loader.batch_size=32 data.transform.no_aug=False data.transform.auto_augment=rand-m9-mstd0.5-inc1 \
                data.transform.re_prob=0.5 data.transform.color_jitter=0.4 data.transform.hflip=0.5 data.transform.vflip=0.5 \
                data.transform.scale=[0.08,1.0] data.transform.ratio=[0.75,1.3333] data.mixup.prob=1.0 data.mixup.mixup_alpha=0.8 \
                data.mixup.cutmix_alpha=1.0 data.mixup.switch_prob=0.5 model=vit \
                model.arch.model_name=vit_tiny_patch16_224 model.optim.learning_rate=0.001 \
                model.scheduler.args.warmup_epochs=5 logger.group=cpmpare_instance_augmentation \
                logger.save_epoch_freq=10000 epochs=80000 mode=pretrain \
                output_dir=./output/pretrain
