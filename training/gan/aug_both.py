import torch
import torch.nn.functional as F
from penalty import compute_penalty
from torch import autograd


def loss_D_fn(P, D, options, images, gen_images):
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)
    images.requires_grad = True

    all_images = torch.cat([images, gen_images], dim=0)
    d_all = D(P.augment_fn(all_images))
    d_real, d_gen = d_all[:N], d_all[N:]

    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    else:
        raise NotImplementedError()

    grad_real, = autograd.grad(outputs=d_real.sum(), inputs=images,#images_aug,
                               create_graph=True, retain_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    penalty = compute_penalty(P.penalty, P=P, D=D, all_images=all_images,
                              images=images, gen_images=gen_images,
                              d_real=d_real, d_gen=d_gen,
                              lbd=options['lbd'], lbd2=options['lbd2'])
    penalty += grad_penalty * (0.5 * P.lbd_r1)

    return d_loss, {
        "penalty": penalty,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss
