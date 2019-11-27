---
title: pytorchRecord
date: 2019-11-26 11:59:06
tags:
  - PyTorch
---

记录一些pytorch中的容易被忽略的小知识

## 1. Tensor和Variable
tensor是pytorch中基本的数据类型，类似于numpy中的array。而variable是对tensor的封装，variable有三个属性。1. 包含的tensor，2. tensor的梯度 .grad，3. 得到梯度的方式 .grad_fn。
