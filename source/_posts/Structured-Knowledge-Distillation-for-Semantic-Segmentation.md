<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
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
2. 本文提出了两种结构化知识蒸馏方法，pair-wise蒸馏和holistic蒸馏。利用这两种方法在紧凑和繁琐的网络之间实现了成对和高阶的一致性。
3. 通过改进最新的紧凑型网络在分割、检测、深度估计三类任务中的效果，本文证明了所提出方法的有效性。

## 方法
分割网络的输入图像通常是RGB的三通道图，这里记为I,大小为$f(x)=ax+b$

