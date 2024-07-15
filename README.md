# Scaling Backwards: Minimal Synthetic Pre-training? (ECCV'24)

## Summary
This repository contains the construction, pre-training, and fine-tuning of the 2D/3D-OFDB dataset in Python/PyTorch. <br>
The repository is based on the paper by Ryo Nakamura*, Ryu Tadokoro*, Ryosuke Yamada, Yuki M. Asano, Iro Laina, Christian Rupprech, Nakamasa Inoue, Rio Yokota and Hirokatsu Kataoka (These authors contributed equally), "Scaling Backwards: Minimal Synthetic Pre-training?", presented at the European Conference on Computer Vision (ECCV) 2024.
<!-- [[Project](https://masora1030.github.io/Visual-Atoms-Pre-training-Vision-Transformers-with-Sinusoidal-Waves/)]  -->
[[arXiv](https://arxiv.org/abs/2307.14710)] 
[[Dataset](https://drive.google.com/drive/folders/1KZfmu1OJKQZhwFKgiJx2mx6VFqq-pFTB?usp=share_link)] 
[[Supp](https://drive.google.com/file/d/1x7eV8jvZOrpdrQFP4tgfiH7HS9wIg53d/view?usp=share_link)]

<p align="center"> <img src="figure/main_image.png" width="90%"/> <p align="center">ComparisonofImageNet-1k,FractalDBand1p-frac(ours).1p-fracconsists of only a single fractal for pre-training. With 1p-frac, neural networks learn to clas- sify perturbations applied to the fractal. In our study “single” means a very narrow distribution over parameters that leads to images that are roughly equivalent from a human visual perspective. While the shape differences of perturbed images can be indistinguishable to humans, models pre-trained on 1p-frac achieve comparable per- formance with those pre-trained on ImageNet-1k or FractalDB.</p>

## 1-parameter Fractal as Data (1p-frac)
If you want to use 1p-frac (1k or 21k version), please download them from the link below. We provide files that compress png files into zip for both 1p-frac.

| ![alt text](figure/1p-frac.png)  |
|:---:|
| Download link : [[1p-frac (1k version)](https://drive.google.com/file/d/1tVsOJlju5ATXj8GT0Qt9TzdBopuxGm6D/view?usp=share_link)]  [[1p-frac (21k version)](https://drive.google.com/file/d/1LV5_hBrDCB4_WyhSh65NA7TvinXMuWRh/view?usp=share_link)]   

## 1p-frac ([README](1p-frac_generater/README.md))
If you want to generate data for 1p-frac, please refer to the 1p-frac README and execute the commands below.

```
$ cd 1p-frac_generater
$ bash generate_1p-frac.sh
```

## Requirements
This section introduces the environment required to pre-train the generated 2D/3D-OFDB or fine-tune the pre-trained model.

* Python 3.x (worked at 3.7.9)
* CUDA (worked at 10.2)
* CuDNN (worked at 7.6)
* NCCL (worked at 2.7)
* OpenMPI (worked at 4.1.3)
* Graphic board (worked at single/four NVIDIA V100)




Please install packages with the following command.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```



## Pre-training

We used almost the same scripts as in [Kataoka_2022_CVPR](https://github.com/masora1030/CVPR2022-Pretrained-ViT-PyTorch) and [Nakamura_20223_ICCV](https://github.com/ryoo-nakamura/OFDB) for our pre-training.

Run the shell script ```pre-traning.sh```, you can pre-train with 1p-frac.

Basically, you can run the python script ```pretrain.py``` with the following command.

- Example : with deit_base, pre-train 2d-ofdb-1k, 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpiexec -npernode 4 -np 8 python -B main.py data=grayimagefolder \
                data.baseinfo.name=2d-ofdb-1k \
                data.baseinfo.train_imgs=1000 data.baseinfo.num_classes=1000 \
                data.trainset.root=PATH/TO/TRAINDATA \
                data.loader.batch_size=32 data.transform.no_aug=False \
                data.transform.auto_augment=rand-m9-mstd0.5-inc1 \
                data.transform.re_prob=0.5 data.transform.color_jitter=0.4 \
                data.transform.hflip=0.5 data.transform.vflip=0.5 \
                data.transform.scale=[0.08,1.0] data.transform.ratio=[0.75,1.3333] \
                data.mixup.prob=1.0 data.mixup.mixup_alpha=0.8 \
                data.mixup.cutmix_alpha=1.0 data.mixup.switch_prob=0.5 model=vit \
                model.arch.model_name=vit_tiny_patch16_224 model.optim.learning_rate=0.001 \
                model.scheduler.args.warmup_epochs=5 logger.group=cpmpare_instance_augmentation \
                logger.save_epoch_freq=10000 epochs=80000 mode=pretrain \
                output_dir=PATH/TO/OUTPUT
    ```

    > **Note**
    > 
    > - ```--batch-size``` means batch size per process. In the above script, for example, you use 4 GPUs (2 process), so overall batch size is 8×32(=256).
    > 
    > - In our paper research, for datasets with more than 1k categories, we basically pre-trained with overall batch size of 256 (8×32).
    > 
    > - If you wish to distribute pre-train across multiple nodes, the following must be done.
    >   - Set the `MASTER_ADDR` environment variable which is the IP address of the machine in rank 0.
    >   - Set the ```-npernode``` and ```-np``` arguments of ```mpirun``` command.
    >     - ```-npernode``` means GPUs (processes) per node and ```-np``` means overall the number of GPUs (processes).

Or you can run the job script ```scripts/pretrain.sh``` (support multiple nodes training with OpenMPI). 
Note, the setup is multiple nodes and using a large number of GPUs (2 nodes and 8 GPUs for pre-train).

When running with the script above, please make your dataset structure as following.

```misc
/PATH/dataset/2d-ofdb-1k/
    image/
        00000/
          xxxxx.png
        00001/
          xxxxx.png
        ...
        ...
    ...
```


Please see the script and code files for details on each arguments.


### Pre-trained models

Pre-trained models will be available soon.
<!-- Our pre-trained models are available in this [[Link](https://drive.google.com/drive/folders/1GUlRQwRPw0qx56L1Voez6RulYXGLgmuw?usp=share_link)]. -->
<!-- Our pre-trained models are available in this Link -->

<!-- We have mainly prepared three different pre-trained models. 
These pre-trained models are ViT-Tiny/Base (patch size of 16, input size of 224) pre-trained on 2D/3D-OFDB-1k/21k and Swin-Base (patch size of 7, window size of 7, input size of 224) pre-trained on 2D/3D-OFDB-21k. -->

<!-- ```misc
pretrain_deit_tiny_2d-ofdb1k_patch_lr1.0e-3_epochs80000_bs256-last.pth #2D-OFDB-1k pre-training for ViT-T

pretrain_deit_tiny_3d-ofdb1k_patch_lr1.0e-3_epochs80000_bs256-last.pth #2D-OFDB-1k pre-training for ViT-T

pretrain_deit_tiny_2d-ofdb21k_patch_lr5.0e-4_epochs15238_bs1024_amp-last.pth.tar #2D-OFDB-21k pre-training for ViT-T

pretrain_deit_tiny_3d-ofdb21k_edgar_lr5.0e-4_epochs15238_bs1024_amp-last.pth.tar #3D-OFDB-21k pre-training for ViT-T


Other pre-trained models will be available on another day
``` -->

## Fine-tuning

We used fine-tuning scripts based on [Nakashima_2022_AAAI](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch).

Run the python script ```finetune.py```, you additionally train other datasets from your pre-trained model.

In order to use the fine-tuning code, you must prepare a fine-tuning dataset (e.g., CIFAR-10/100, ImageNet-1k, Pascal VOC 2012). 
You should set the dataset as the following structure.

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

Basically, you can run the python script ```finetune.sh``` with the following command.

- Example : with deit_base, fine-tune CIFAR10 from pre-trained model (with 2D-OFDB-1k), 8 GPUs (Batch Size = 96×8 = 768)

    ```bash
    $ mpiexec -npernode 4 -np 8 \
        python -B finetune.py data=colorimagefolder \
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
        logger.experiment=finetune_deit_base_CIFAR10_batch768_from_2d-ofdb-21000_1.0e-3 \
        logger.save_epoch_freq=100 \
        output_dir=./output/finetune
    ```

Or you can run the job script ```scripts/finetune.sh``` (support multiple nodes training with OpenMPI).

Please see the script and code files for details on each arguments.


## Citation
If you use our work in your research, please cite our paper:
```bibtex
@InProceedings{xxxxxx,
    title={Scaling Backwards: Minimal Synthetic Pre-training?},
    author={Ryo Nakamura, Ryu Tadokoro, Ryosuke Yamada, Yuki M. Asano, Iro Laina, Christian Rupprech, Nakamasa Inoue, Rio Yokota and Hirokatsu Kataoka},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2024},
}
``` 


## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST) and Tokyo Institute of Technology (TITech) are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the images and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.

[def]: #summary

## Acknowledgements
In this repository is based on [2D/3D-OFDB](https://github.com/ryoo-nakamura/OFDB).
