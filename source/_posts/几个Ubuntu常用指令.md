---
title: 几个Ubuntu常用指令
date: 2019-11-12 12:12:57
categories: 
- 开发工具
tags:
- Ubuntu
---

在复制大量图片时，直接用cp命令会报错Argument list too long，可以采用

``` bash
$ find sourcePath/ -name "*.jpg" -exec cp {} targetPath/ \;
```

注意最后的分号不能省略。

有时需要保存文件夹下所有jpg的绝对路径，可以采用
``` bash
$ ls | sed "s:^:`pwd`/:" > image.txt
```
统计某一目录下所有子目录的图片数量，
```bash
$ ls -lR | grep '^-' | wc -l
```
