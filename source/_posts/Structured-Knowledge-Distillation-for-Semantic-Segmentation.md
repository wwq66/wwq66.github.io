---
title: Structured Knowledge Distillation for Semantic Segmentation
date: 2019-11-13 20:45:52
tags: 
  - Segmentation
categories: 
  - Paper Reading
---


## 摘要
在本文中，为了进行稠密预测任务（Dense predicition tasks）,作者希望把结构信息从大模型迁移到小模型中去。先前关于稠密预测任务的模型蒸馏策略通常都是直接借鉴图像分类的模型蒸馏技术，并且都是针对单个像素单独地做知识蒸馏，这样做一般达不到最优表现。考虑到稠密预测是一种结构化预测问题，本文提出了一种从大网络向小网络进行知识蒸馏的方法。具体来说，本文研究了两种结构化蒸馏方法：1，成对蒸馏（pair-wise distillation），通过构建静态图来蒸馏成对相似性。2，整体蒸馏（holistic distillation），利用对抗性训练来提取整体知识。通过语义分割、深度估计和目标检测三类稠密预测任务的实验，证明了本文的知识蒸馏方法是有效果的。

## 本文贡献
1. 本文研究了用于训练稠密预测任务的知识蒸馏策略。
2. 本文提出了两种结构化知识蒸馏方法，pair-wise蒸馏和holistic蒸馏。利用这两种方法在简单网络和复杂网络之间实现了成对和高阶的一致性。
3. 通过改进最新的简单网络在分割、检测、深度估计三类任务中的效果，本文证明了所提出方法的有效性。

## 方法
:分割网络的输入图像通常是RGB的三通道图，这里记为$I$,大小为$W \times H \times 3$，网络计算的feature map $F$大小为$W' \times H' \times N$，$N$表示通道数。接下来用一个分类模块来计算分割图$Q$，大小为$W' \times H' \times C$。

### Pixel-wise distillation
为了更好地训练简单网络，知识蒸馏可以将复杂网络$T$的知识迁移到简单网络$S$中。本文将分割任务看成是单独像素分类任务的集合，并且直接使用知识蒸馏来调整由简单网络产生的每个像素的分类概率。本文采取的做法是使用复杂网络的分类概率作为用来训练简单网络的soft target。损失函数为
$$\ell\_{pi}(S) = \frac{1}{W' \times H'}\sum_{i\in R}\mathbf{KL}(\mathbf{q}{\_i}{^s}||\mathbf{q}{\_i}{^t})$$
这里$\mathbf{q\_{i}{^s}}$表示简单网络$S$中第$i$个像素的类别概率，$\mathbf{q\_{i}{^t}}$表示复杂网络$T$的第$i$个像素的类别概率，$R = \{1,2,...,W' \times H'\}$为所有像素的集合。

### 结构知识蒸馏
除了使用pixel-wise的知识蒸馏的方法外，本文还提出了pair-wise蒸馏和holistic蒸馏，结构图如图1所示。![图1](/img/pic1.jpg)

### Pair-wise distillation
