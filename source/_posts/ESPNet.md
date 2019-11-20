---
title: ESPNet,ESPNetv2
date: 2019-11-13 20:45:52
tags: 
  - Segmentation
categories: 
  - Paper Reading
---

前段时间随手跑了下ESPNetv2的代码，发现在自己目前的项目上mIoU能达到0.95，惊呆了，于是打算看下文章记录下。

# ESPNet

## 摘要
ESPNet是基于ESP模块提出的一种小型、快速、低功耗、低延时的轻量级分割网络。和PSPNet相比，它在速度上快22倍，在体积上小180倍。此外，ESPNet比MobileNet，ShuffleNet，ENet效果都要好，在一块标注GPU上能达到9帧每秒的速度。

## 方法
图1是ESPNet的结构图。![图1](/img/espnet1.jpg)
其主要思路有两步：
1. Point-wise卷积；
2. 空洞卷积的空间金字塔结构。
在ESP模块中，首先会进行一个point-wise卷积进行降维。然后使用$K$个$n \times n$的空洞卷积核进行空间金字塔的空洞卷积。每个卷积核的dilation rate是$2^{k-1}$，$k=\{1,...,K\}$。这样做能降低参数量，同时也能保留一个很大的感受野$[(n-1)2^{K-1}+1]^2$。

**Width divider K**
假设某层的输入通道数和输出通道数分别为M，N，卷积核大小为$n \times n$，则该层的需学习的参数量为$n^2MN$。为了减小计算量，这里引入一个超参数K，K的作用是均匀地缩小网络中每个ESP模块的特征映射的维数。给定一个K，ESP模块首先用point-wise卷积将feature map从M维空间映射到$N/K$维空间。然后，这些低纬度的feature map会被分成K个平行的分支。每个分支均使用不同dilation rate（大小为$2^{k-1}$，$k=\{1,...,K-1\}$）的$n \times n$大小的空洞卷积核进行空洞卷积。最后将这些feature map concat起来，得到最终的N维的feature map。如图1所示。

一个ESP模块的参数量是$\frac{MN}{K}+\frac{(nN)^2}{K}$，其感受野是$[(n-1)2^{K-1}+1]^2$。相比于标准卷积，使用ESP模块后参数量减小了$\frac{n^2MK}{M+n^2N}$，同时，感受野增大了$[2^{K-1}]^2$倍。

**分层特征融合（解决de-gridding）**
在concat由空洞卷积产生的各个feature map时，会产生棋盘效应（gridding artifact），如图2所示。![图2](/img/espnet2.jpg)
为了解决ESP模块中的棋盘效应，做法是将这些用不同空洞卷积核产生的feature map分层添加在一起。这种做法简单高效，并且不会增加ESP模块的复杂度，而现有的减小棋盘效应的方法通常是使用较小dilation rate的空洞卷积核。

## 实验
实验结果如图3所示。![图3](/img/espnet3.jpg)
可以看到精度相比于ResNet下降不多，但速度很快。