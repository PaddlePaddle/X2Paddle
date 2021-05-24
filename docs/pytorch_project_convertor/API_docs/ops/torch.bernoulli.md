## torch.bernoulli
### [torch.bernoulli](https://pytorch.org/docs/stable/generated/torch.bernoulli.html?highlight=bernoulli#torch.bernoulli)
```python
torch.bernoulli(input, *, generator=None, out=None)
```
### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bernoulli_cn.html#bernoulli)
```python
paddle.bernoulli(x, name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入Tensor。                                     |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle无此参数。                   |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |  

***【注意】*** 这类生成器的用法如下：
```python
G = torch.Generator()
G.manual_seed(1)
# 生成指定分布Tensor
torch.randperm(5, generator=G)
```
