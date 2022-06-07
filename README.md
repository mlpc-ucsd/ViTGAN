# ViTGAN: Training GANs with Vision Transformers
This repository contains the code for the paper:
<br>
[**ViTGAN: Training GANs with Vision Transformers**](https://openreview.net/forum?id=dwg5rXg1WS_)
<br>
Kwonjoon Lee, Huiwen Chang, [Lu Jiang](http://www.lujiang.info), [Han Zhang](https://sites.google.com/view/hanzhang), [Zhuowen Tu](https://pages.ucsd.edu/~ztu/), [Ce Liu](https://people.csail.mit.edu/celiu/)   
ICLR 2022 (**Spotlight**)

<p align='center'>
  <img src='algorithm.png' width="800px">
</p>


### Abstract

Recently, Vision Transformers (ViTs) have shown competitive performance on image recognition while requiring less vision-specific inductive biases. In this paper, we investigate if such performance can be extended to image generation. To this end, we integrate the ViT architecture into generative adversarial networks (GANs). For ViT discriminators, we observe that existing regularization methods for GANs interact poorly with self-attention, causing serious instability during training. To resolve this issue, we introduce several novel regularization techniques for training GANs with ViTs. For ViT generators, we examine architectural choices for latent and pixel mapping layers to faciliate convergence. Empirically, our approach, named ViTGAN, achieves comparable performance to the leading CNN- based GAN models on three datasets: CIFAR-10, CelebA, and LSUN bedroom.

### Current Status

This is a **PyTorch reproduction** of the orignal ViTGAN code (which was orignally written in Tensorflow 2 to run on Google Cloud TPUs) by the authors. Due to the subtle differences between Tensorflow 2 and PyTorch implementations, we had to make modifications to hyperparameters such as learning rate and coefficient for R1 and bCR penalties. Currently, we only implement StyleGAN2-D+ViTGAN-G, which is the most performant variant. In addition to the original generator architecture which used Implicit Neural Representation for patch generation, we provide the generator architecture that employs convolutional blocks for patch generation. **In practice, we recommend using convolutional patch generation as it brings faster convergence (in terms of both wall clock time and the number of iterations).**

### Running Training
1. To train StyleGAN2-D+StyleGAN2-G (**Convolutional Network for Patch Generation**) on CIFAR-10 benchmark:
    ```bash
    python train_stylegan2.py configs/gan/stylegan2/c10_style64.gin stylegan2 --mode=aug_both --aug=diffaug --lbd_r1=0.1 --no_lazy --halflife_k=1000 --penalty=bcr --use_warmup
    ```
3. To train StyleGAN2-D+ViTGAN-G (**Convolutional Network for Patch Generation**) on CIFAR-10 benchmark:
    ```bash
    python train_stylegan2.py configs/gan/stylegan2/c10_style64.gin vitgan --mode=aug_both --aug=diffaug --lbd_r1=0.1 --no_lazy --halflife_k=1000 --penalty=bcr --use_warmup
    ```
3. To train StyleGAN2-D+ViTGAN-G (**Implicit Neural Representation for Patch Generation**) on CIFAR-10 benchmark:
    ```bash
    python train_stylegan2.py configs/gan/stylegan2/c10_style64.gin vitgan --mode=aug_both --aug=diffaug --lbd_r1=0.1 --no_lazy --halflife_k=1000 --penalty=bcr --use_warmup --use_nerf_proj
    ```

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{lee2022vitgan,
  title={ViTGAN: Training GANs with Vision Transformers},
  author={Kwonjoon Lee, Huiwen Chang, Lu Jiang, Han Zhang, Zhuowen Tu, Ce Liu},
  booktitle={ICLR},
  year={2022}
}
```

## Acknowledgments

This code is based on the implementations of [**Training GANs with Stronger Augmentations via Contrastive Discriminator**](https://github.com/jh-jeong/ContraD), and [**StyleGAN2-pytorch**](https://github.com/rosinality/stylegan2-pytorch).
