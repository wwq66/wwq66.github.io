---
title: ganloss
date: 2019-12-05 15:19:39
tags:
  - PyTorch
  - GANs
categories: 
  - Paper Reading
---
梳理下GAN中的loss。

## KL散度和JS散度
KL散度和JS散度是用来描述两个分布之间的相似程度的。
$$\mathbf{KL}(p|q) = \int_{x}p(x)\log\frac{p(x)}{q(x)}\mathrm{d}x$$
$$\mathbf{JS}(P_r,P_g) =0.5\mathbf{KL}(P_r|P_m) +0.5\mathbf{KL}(P_g|P_m)$$
其中$P_m = \frac{P_r + P_g}{2}$。
JS散度的性质是当两概率都为0时，$\mathbf{JS}=0$，当一个为0一个不为0时，$\mathbf{JS}=\log2$

## GAN的损失函数
V(G,D) = \underset{G}{min}\underset{D}{max} E_{xp_{data}(x)}[\log D(x)] + E_{zp_{z}(z)}[1-D(G(z))]