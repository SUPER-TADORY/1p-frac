
# Pre-training & Fine-tuning

This repository contains the code for 1p-frac pre-training and fine-tuning for downstream tasks.

## Pre-training

We used scripts similar to those in [Kataoka_2022_CVPR](https://github.com/masora1030/CVPR2022-Pretrained-ViT-PyTorch) and [Nakamura_20223_ICCV](https://github.com/ryoo-nakamura/OFDB) for our pre-training.

To perform pre-training with 1p-frac, first modify the following parts of `pre-training.sh`:

```bash
export NGPUS=XX  # Set total number of GPUs
export NPERNODE=XX  # Set number of nodes

BASE_DATADIR='/path/to/base/data/directory'
ONEPFRAC_NAME='set_dataset_name'  # Set the name of the 1p-frac dataset you generated (e.g., sigma3.5_perturb0.1_sample1000_Kift13W5)
MODEL_NAME='set_model_name'  # Set the model name from timm (e.g., vit_tiny_patch16_224)
```

Then, run the script:

```bash
bash pre-training.sh
```

By default, `main.py` in `pre-training.sh` is configured for a dataset with 1000 samples from 1p-frac. Adjust the hyperparameters based on the sample size and other settings.

- Example: Using 4 GPUs with a total batch size of 256 (64×4):

```bash
mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=grayimagefolder \
data.baseinfo.name=${ONEPFRAC_NAME} \
data.baseinfo.train_imgs=${SAMPLE_NUM} data.baseinfo.num_classes=${SAMPLE_NUM} \
data.trainset.root=${BASE_DATADIR} \
data.loader.batch_size=32 data.transform.no_aug=False data.transform.auto_augment=rand-m9-mstd0.5-inc1 \
data.transform.re_prob=0.5 data.transform.color_jitter=0.4 data.transform.hflip=0.5 data.transform.vflip=0.5 \
data.transform.scale=[0.08,1.0] data.transform.ratio=[0.75,1.3333] data.mixup.prob=1.0 \
data.mixup.mixup_alpha=0.8 data.mixup.cutmix_alpha=1.0 data.mixup.switch_prob=0.5 model=vit \
model.arch.model_name=${MODEL_NAME} model.optim.learning_rate=0.001 \
model.scheduler.args.warmup_epochs=5 logger.group=compare_instance_augmentation \
logger.save_epoch_freq=10000 epochs=80000 mode=pretrain output_dir=./output/pretrain
```

> **Note:**
>
> - `--batch-size` represents the batch size per process. In the above example, using 4 GPUs results in an overall batch size of 256 (64×4).
> - In our experiments, datasets with over 1k categories were pre-trained with a batch size of 256.
> - For multi-node training, set the `MASTER_ADDR` environment variable to the IP address of the rank 0 machine.
> - Adjust the `-npernode` and `-np` arguments in the `mpiexec` command as needed.

When running the script, ensure your dataset is structured as follows:

```misc
/PATH/to/dataset/
    image/
        00000/
          xxxxx.png
        00001/
          xxxxx.png
        ...
        ...
    ...
```

Please refer to the scripts and code files for details on each argument.

### Pre-trained models

Pre-trained models will be available soon.

<!-- Our pre-trained models are available in this [[Link](https://drive.google.com/drive/folders/1GUlRQwRPw0qx56L1Voez6RulYXGLgmuw?usp=share_link)]. -->

## Fine-tuning

We used fine-tuning scripts based on [Nakashima_2022_AAAI](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch).

To fine-tune on other datasets, run the Python script `finetune.py` using your pre-trained model.

Prepare the fine-tuning dataset (e.g., CIFAR-10/100, ImageNet-1k, Pascal VOC 2012) with the following structure:

```misc
/PATH/TO/DATASET/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
    ...
  val/
    class1/
      img3.jpeg
      ...
    class2/
      img4.jpeg
      ...
    ...
```

Next, modify the following sections of `finetune.sh` (CIFAR100 example):

```bash
DATASET_NAME=CIFAR100
CLASS_NUM=100  # Set the number of classes
TRAIN_DATADIR='/path/to/cifar100/train'  # Set the training data path
VAL_DATADIR='/path/to/cifar100/val'  # Set the validation data path
CKPT_PATH='/path/to/pretrained_weights'  # Set the path to the checkpoint
MODEL_NAME='set_model_name'  # Set the model name (e.g., vit_tiny_patch16_224)
```

Finally, run the script:

```bash
bash finetune.sh
```

By default, `main.py` in `finetune.sh` is configured for fine-tuning on CIFAR100. Customize the hyperparameters according to your dataset configuration.

- Example: Fine-tuning CIFAR100 from a pre-trained model, 8 GPUs (Batch Size = 96×8 = 768):

```bash
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
```

Please refer to the scripts and code files for details on each argument.
