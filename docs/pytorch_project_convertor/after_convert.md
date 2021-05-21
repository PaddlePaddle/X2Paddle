# 转换后代码后处理
1. 若需要使用GPU，且预处理中使用了Tensor，`x2paddle.torch2paddle.DataLoader`中的`num_workers`必须设置为0。

2. 修改自定义Dataset（继承自`paddle.io.Dataset`）中的`__getitem__`的返回值，若返回值中存在Tensor，需添加相应代码将Tensor修改为numpy。

```
# 原始代码
class VocDataset(paddle.io.Dataset):
    ...
    def __getitem__(self):
        ...
        return out1, out2
    ...
# 替换后代码
class VocDataset(paddle.io.Dataset):
    ...
    def __getitem__(self):
        ...
        if isinstance(out1, paddle.Tensor):
            out1 = out1.numpy()
        if isinstance(out2, paddle.Tensor):
            out2 = out2.numpy()
        return out1, out2
    ...
```

3. 若存在Tensor对比操作（包含==、!=、<、<=、>、>=操作符）,在对比操作符前添加对Tensor类型的判断，如果为非bool型强转为bool型，并在对比后转换回bool型。

```
# 原始代码（其中c_trg是非bool型的Tensor）
c_trg = c_trg == 0
# 替换后代码
c_trg = c_trg.cast("int32")
c_trg_tmp = paddle.zeros_like(c_trg)
paddle.assign(c_trg, c_trg_tmp)
c_trg_tmp = c_trg_tmp.cast("bool")
c_trg_tmp[:, i] = c_trg[:, i] == 0
c_trg = c_trg_tmp 
```

4. 如若转换后的运行代码的入口为sh脚本文件，且其中有预训练模型路径，应将其中的预训练模型的路径字符串中的“.pth”、“.pt”、“.ckpt”替换为“.pdiparams”。
