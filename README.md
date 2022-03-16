# ViTGAN: Training GANs with Vision Transformers
This repository contains the code for the paper:
<br>
[**ViTGAN: Training GANs with Vision Transformers**](https://arxiv.org/pdf/2107.04589)
<br>
Kwonjoon Lee, Huiwen Chang, [Lu Jiang](http://www.lujiang.info), [Han Zhang](https://sites.google.com/view/hanzhang), [Zhuowen Tu](https://pages.ucsd.edu/~ztu/), [Ce Liu](https://people.csail.mit.edu/celiu/)   
ICLR 2022 (**Spotlight**)

### Abstract

Recently, Vision Transformers (ViTs) have shown competitive performance on image recognition while requiring less vision-specific inductive biases. In this paper, we investigate if such performance can be extended to image generation. To this end, we integrate the ViT architecture into generative adversarial networks (GANs). For ViT discriminators, we observe that existing regularization methods for GANs interact poorly with self-attention, causing serious instability during training. To resolve this issue, we introduce several novel regularization techniques for training GANs with ViTs. For ViT generators, we examine architectural choices for latent and pixel mapping layers to faciliate convergence. Empirically, our approach, named ViTGAN, achieves comparable performance to the leading CNN- based GAN models on three datasets: CIFAR-10, CelebA, and LSUN bedroom.

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
