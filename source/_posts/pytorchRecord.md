---
title: 关于PGGAN PyTorch代码中的一些问题
date: 2019-11-26 11:59:06
tags:
  - PyTorch
  - GANs
categories: 
  - Paper Reading
---

## He initialization
关于权值的初始化方式，pggan中采用的是kaiming初始化，代码如下
``` python
def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with 'relu' or 'leaky_relu' (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)
```
可以看到kaiming初始化是将权值按正态分布$N(0,std)$初始化。标准差的计算公式为$std = \frac{gain}{\sqrt{fan}}$。即需要知道gain和fan的值。fan的求法：
``` python
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
```
其中mode为“ fan_in”表示保留前向传递中权重方差的大小。 mode为“ fan_out”会保留反向传递的幅度。fan_in和fan_out的求法：
``` python
def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()	# 向量中元素总个数
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
```
dimensions为tensor的shape，如[512,512,4,4]。
gain的求法：
``` python 
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
```
可以看到在nonlinearity为Leaky ReLU时，std的值为$std = \sqrt{\frac{2}{(1+a^2) \times fan\_{in}}}$

## named_children()和named_modules()
**named_children()**
Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
返回子模块的迭代器，同时产生模块名称以及模块本身。
**named_modules()**
Returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
返回网络中所有模块的迭代器，同时产生模块的名称以及模块本身。
引用一个网上的例子
``` python
import torch
import torch.nn as nn
 
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(16,32,3,1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32,10)
        )
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
 
model = TestModule()
 
for name, module in model.named_children():
    print('children module:', name)
 
for name, module in model.named_modules():
    print('modules:', name)
```
结果
``` python
children module: layer1
children module: layer2
modules: 
modules: layer1
modules: layer1.0
modules: layer1.1
modules: layer2
modules: layer2.0
```

## pixel-wise normalization 和local response normalization(LRN)
LRN用来进行局部对比度增强，以便使局部特征在下一层表述。公式如下：
$$b_{xy}^i = \frac{a_{xy}^i}{[k + \alpha\sum_{j=max(0,i-\frac{n}{2})}^{min(N-1,i+\frac{n}{2})}(a_{xy}^j)^2]^\beta}$$
其中$a$为卷积层后的输出，大小为$[N,H,W,C]$，$a_{xy}^i$表示在这个输出结构中的一个位置$[a,b,c,d]$，即第$a$张图的第$d$个通道的高度为$b$，宽度为$c$的点。公式中$N$代表channel数，$a$为输出，$\frac{n}{2}$表示深度半径，$k$为偏置，$\alpha$和$\beta$为自定义参数。该公式可以理解为对某点按通道邻域进行归一化。当这个通道邻域为这个通道时，即为pixel-wise normalization。公式如下：
$$b_{x,y} = \frac{a_{x,y}}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}{(a_{x,y}^j)^2 + \epsilon}}}$$
其中$\epsilon = 10^{-8}$，$N$是feature map的个数，$a_{x,y}$和$b_{x,y}$分别是$pixel(x,y)$原始向量和归一化向量。文章指出这种约束并没有损害生成器，并且在大多数数据集上它也没有太大地改变效果。但这种归一化可以非常有效地防止信号幅度的上升。

## minibatch standard deviation
在Improved GAN中作者指出，出现mode collapse后，没有任何信息能引导网络走出这一困境。生成器的每个minibatch输出都应该具有足够的分辩能力，即每个minibatch不应该太相似。惩罚minibatch的相似性能够在mode collapse发生时，产生足够的梯度信息，引导它走出。具体做法如下。
假设样本$x_i$在D网络中某一层的特征向量为$f(x_i) \in R^A$，然后将$f(x_i)$乘以一个张量$T \in R^{A \times B \times C}$得到张量$M_i \in R^B \times C$。基于不同的样本生成的矩阵$M_i,i=1,2,...,n$的行之间计算$L_1$距离，得到$c_b(x_i,x_j) = e^{-||M_{ib}-M_{jb}|| \_{L1}} \in R$，然后将所有的$c_b(x_i,x_j)$相加得到$o(x_i)\_b$，最后将$B$个$o(x_i)\_b$并起来得到一个大小为$B$的向量$o(x_i)$。最后，将$o(x_i)$和$f(x_i)$合并成一个向量作为D网络下一层的输入。如图1所示。
$$o(x_i)\_b = \sum_{j=1}^{n}{c_b(x_i,x_j)} \in \mathbb{R}$$
$$o(x_i) = [o(x_i)\_1, o(x_i)\_2, ..., o(x_i)\_B] \in \mathbb{B}$$
$$o(X) \in \mathbb{R}^{n \times B}$$
![图1](/img/pggan3.jpg)
PGGAN中提出的minibatch standard deviation也是基于这种思路。看下代码：
``` python
class minibatch_std_concat_layer(nn.Module):
    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):   # [32,512,4,4]
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)    # [1,512,4,4]
        if self.averaging == 'all':
            target_shape[1] = 1     
            vals = torch.mean(vals, dim=1, keepdim=True)    # [1,1,4,4]
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)   # target_shape:[32,1,4,4]-> vals shape:[32,1,4,4]
        return torch.cat([x, vals], 1)  # [32,513,4,4]
```
输入x为feature map。其中adjusted_std函数用来根据指定维度计算标准差。在调用时，首先在batch方向上计算标准差vals,然后在channel方向上计算这些标准差的均值。将最终的均值concat在feature map之后。

## 有针对性地给样本加噪声
在文章Towards Principled Methods for Training Generative Adversarial Networks中，作者分析了GAN损失函数存在的问题，并提出了两种解决方法。其中一种是对两个分布加噪声，使其能重叠，以此来缓解mode collapse。当生成样本在判别器中的输出接近1时，此时的loss会接近常数，梯度接近消失。不能再来引导生成器。因此，可以用添加噪声的方式来避免这种问题。代码如下：
``` python
    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        if hasattr(self, '_d_'):    # 有_d_属性，d*0.9 + fx_tilde*0.1，fx_tilde为生成样本经过判别器后的结果
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:        # 给输入向量添加_d_属性，并赋值为0
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2      
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z
```
其中strength用来控制噪声大小，其值与变量_d_有关。而_d_的值与真实样本在判别器的输出有关。也就是说 ，判别器输出越大，所加的噪声越大。而当判别器输出小于0.5时，不添加噪声。原因是网络收敛时，判别器对于样本的判别概率应该是0.5，那么当输出小于0.5时表明网络的分辨能力较弱。