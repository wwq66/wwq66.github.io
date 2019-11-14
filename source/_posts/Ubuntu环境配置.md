---
title: Ubuntu环境配置
date: 2019-11-13 12:12:57
categories: 
- 开发工具
tags:
- Ubuntu
---


### 1.修改tmux prefix快捷键

tmux默认的prefix是Ctrl + B，使用起来不太方便，修改成Ctrl + A。
在~/.tmux.conf中输入下面命令
```bash
set -g prefix C-x
unbind C-b
bind C-x send-prefix
```
重进tmux的plane就ok啦。
### 2.conda虚拟环境管理
#### 1. 创建虚拟环境
```bash
conda create -n train python=3.6
```
进入虚拟环境
```bash
conda activate train
```
#### 2. 删除虚拟环境
```bash
conda remove -n train --all
```
#### 3. 重命名（克隆）虚拟环境
```bash
conda create -n train_new --clone train
```
再删除原环境train即可，这样便实现将虚拟环境train更名为train_new。
#### 4. 导出虚拟环境的配置
首先进入到需要导出的环境中
```bash
conda activate train
```
导出为yml文件
```bash
conda env export --file ~/home/ccc/train.yml
```
#### 5. 导入虚拟环境的配置
先进入需要导入配置的环境中
```bash
conda activate train_new
```
导入yml文件
```bash
conda env create -f ~/home/ccc/train.yml
```