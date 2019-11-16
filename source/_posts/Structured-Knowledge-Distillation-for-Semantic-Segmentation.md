---
title: 【2019 CVPR】Structured Knowledge Distillation for Semantic Segmentation
date: 2019-11-13 20:45:52
tags: 
  - Segmentation
categories: 
  - Paper Reading
---


# 摘要
在本文中，为了进行稠密预测任务（Dense predicition tasks）,作者希望把结构信息从大模型迁移到小模型中去。先前关于稠密预测任务的模型蒸馏策略通常都是直接借鉴图像分类的模型蒸馏技术，并且都是针对单个像素单独地做知识蒸馏，这样做一般达不到最优表现。考虑到稠密预测是一种结构化预测问题，本文提出了一种从大网络向小网络进行知识蒸馏的方法。具体来说，本文研究了两种结构化蒸馏方法：1，成对蒸馏（pair-wise distillation），通过构建静态图来蒸馏成对相似性。2，整体蒸馏（holistic distillation），利用对抗性训练来提取整体知识。通过语义分割、深度估计和目标检测三类稠密预测任务的实验，证明了本文的知识蒸馏方法是有效果的。

# 本文贡献
1. 本文研究了用于训练稠密预测任务的知识蒸馏策略。
2. 本文提出了两种结构化知识蒸馏方法，pair-wise蒸馏和holistic蒸馏。利用这两种方法在简单网络和复杂网络之间实现了成对和高阶的一致性。
3. 通过改进最新的简单网络在分割、检测、深度估计三类任务中的效果，本文证明了所提出方法的有效性。

# 方法
分割网络的输入图像通常是RGB的三通道图，这里记为$I$,大小为$W \times H \times 3$，网络计算的feature map $F$大小为$W' \times H' \times N$，$N$表示通道数。接下来用一个分类模块来计算分割图$Q$，大小为$W' \times H' \times C$。

**Pixel-wise distillation**
为了更好地训练简单网络，知识蒸馏可以将复杂网络$T$的知识迁移到简单网络$S$中。本文将分割任务看成是单独像素分类任务的集合，并且直接使用知识蒸馏来调整由简单网络产生的每个像素的分类概率。本文采取的做法是使用复杂网络的分类概率作为用来训练简单网络的soft target。损失函数为
$$\ell\_{pi}(S) = \frac{1}{W' \times H'}\sum_{i\in R}\mathbf{KL}(\mathbf{q}{\_i}{^s}||\mathbf{q}{\_i}{^t})$$
这里$\mathbf{q\_{i}{^s}}$表示简单网络$S$中第$i$个像素的类别概率，$\mathbf{q\_{i}{^t}}$表示复杂网络$T$的第$i$个像素的类别概率，$R = \{1,2,...,W' \times H'\}$为所有像素的集合。

## 结构知识蒸馏
除了使用pixel-wise的知识蒸馏的方法外，本文还提出了pair-wise蒸馏和holistic蒸馏，结构图如图1所示。![图1](/img/pic1.jpg)

**Pair-wise distillation**
首先构建一个静态关系图来表示空间成对关系，如图2所示。![图2](/img/pic2.jpg)
我们用每一个点的联系范围$\alpha$和间隔大小$\beta$来控制静态关系图的大小。对于每一个节点，我们仅通过空间距离（此处使用的是切比雪夫距离）来考虑与最近的 $\alpha$个节点的相似性，并聚合局部的$\beta$个像素来表示该节点的特征。对于大小为$W' \times H' \times C$的特征图，共有$W' \times H'$个像素，静态关系图中共包含$\frac{W' \times H'}{\beta}$个节点和$\frac{W' \times H'}{\beta} \times \alpha$个关系。
用$a^{t}\_{ij}$表示$T$网络生成的第$i$个节点和第$j$个节点的相似性，$a^{s}\_{ij}$表示$S$网络生成的第$i$个节点和第$j$个节点的相似性。采用平方损失来计算pair-wise相似性蒸馏损失
$$\ell\_{pa}(S) = \frac{W' \times H' \times \alpha}{\beta}\sum_{i \in R'}\sum_{j \in \alpha}(a^{s}\_{ij} - a^{t}\_{ij})^2$$
其中$R' = \{ 1,2,...,\frac{W' \times H' \times \alpha}{\beta} \}$为所有节点的集合。在实际操作中，我们使用平均池化将一个节点的$\beta \times C$个特征聚合成$1 \times C$，这样两个节点的相似性可以简化为计算聚合特征$\mathbf{f}\_{i}$和$\mathbf{f}\_{j}$，
$$a\_{ij} = \frac{\mathbf{f}^{\top}\_{i}\mathbf{f}\_{j}}{||\mathbf{f}\_{i}||\_{2}||\mathbf{f}\_{j}||\_{2}}$$

**Holistic distillation**
我们计算分割图的holistic embedding来将复杂网络和简单网络生成的分割图之间的高阶关系对齐。这里采用了传统的GAN来解决holistic distillation问题。简单网络可以看成生成器，其预测出的分割图$\mathbf{Q^s}$被视为假样本，复杂网络预测出的分割图$\mathbf{Q^t}$被视为真样本。我们希望$\mathbf{Q^s}$能尽可能地接近$\mathbf{Q^t}$。
Wasserstein distance用来衡量两个不同分布的差异，其定义是将模型分布$p\_{s}(\mathbf{Q}^s)$向真实分布$p\_{t}(\mathbf{Q}^t)$靠拢时所花的最小代价。它可以用来解决梯度消失或梯度爆炸的问题。公式如下
$$\ell\_{ho}(S,D) = \mathbb{E}\_{\mathbf{Q}^t \sim p\_{s}(\mathbf{Q}^s)}[D(\mathbf{Q}^s|\mathbf{I})] - \mathbb{E}\_{\mathbf{Q}^t \sim p\_{t}(\mathbf{Q}^t)}[D(\mathbf{Q}^t|\mathbf{I})]$$
其中$、mathbb{E}[\cdot]$为求均值，$D(\cdot)$是一个嵌入式网络，即GAN中的判别器，它将$\mathbf{Q}$和$\mathbf{I}$一起投影到一个整体的嵌入分数中。

分割图和RGB图一起concat后送入判别网络D，D是一个全卷积网络，共五个卷积层。在最后三个卷积层之间，我们插入了两个注意力模块用来捕获结构信息。在concat层前面我们添加了一个BN层用来处理RGB图像和logits之间的不同尺度问题。

这样的判别器可以产生一个整体嵌入来表征输入图像和分割图像的匹配程度。我们进一步加入了一个池化层来将整体嵌入池化为一个分数。在对抗训练中我们加入wasserstein distance来训练判别器，使其给予教师网络输出的分割图一个更高的分数，给予学生网络输出的分割图一个更低的分数。在此过程中，我们将评估分割图质量的知识提取到判别器中。同时我们对学生网络进行训练，使其在判别器下获得更高的分数。

## 优化
整个目标函数由传统的pixel-wise多分类交叉熵损失$\ell\_mc(S)$和结构蒸馏项组成。
$$\ell(S,D) = \ell\_{mc}(S) + \lambda\_{1}() - \lambda\_{2}\ell\_{}$$