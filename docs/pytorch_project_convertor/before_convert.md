# 转换前代码预处理
1. 去除TensorBoard相关的操作。

2. 将PyTorch中Tensor逐位逻辑与、或、异或运算操作符替换为对应的API的操作：
> | 替换为 torch.bitwise_or  
> & 替换为 torch.bitwise_and  
> ^ 替换为 torch.bitwise_xor  

``` python
# 原始代码：
pos_mask | neg_mask
# 替换后代码
torch.bitwise_or(pos_mask, neg_mask)
```

3. 若自定义的`DataSet`（用于加载数据模块，作为`torch.utils.data.DataLoader`的参数）未继承`torch.utils.data.Dataset`，则需要添加该继承关系。

```
# 原始代码
class VocDataset:
# 替换后代码
import torch
class VocDataset(torch.utils.data.Dataset):
```

4. 若预训练模型需要下载，去除下载预训练模型相关代码，在转换前将预训练模型下载至本地，并修改加载预训练模型参数相关代码的路径为预训练模型本地保存路径。

5. 若在数据预处理中出现Tensor与float型/int型对比大小，则需要将float型/int型修改为Tensor，例如下面代码为一段未数据预处理中一段代码，修改如下：
``` python
# 原始代码：
mask = best_target_per_prior < 0.5
# 替换后代码
threshold_tensor = torch.full_like(best_target_per_prior, 0.5)
mask = best_target_per_prior < threshold_tensor
```
