## torch.nn.Embedding
### [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding)
```python
torch.nn.Embedding(num_embeddings,
                   embedding_dim,
                   padding_idx=None,
                   max_norm=None,
                   norm_type=2.0,
                   scale_grad_by_freq=False,
                   sparse=False)
```
### [paddle.nn.Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/common/Embedding_cn.html#embedding)
```python
paddle.nn.Embedding(num_embeddings,
                    embedding_dim,
                    padding_idx=None,
                    sparse=False,
                    weight_attr=None,
                    name=None)
```

### 功能差异
#### 归一化设置
***PyTorch***：当max_norm不为`None`时，如果Embeddding向量的范数（范数的计算方式由norm_type决定）超过了max_norm这个界限，就要再进行归一化。  
***PaddlePaddle***：PaddlePaddle无此要求，因此不需要归一化。

#### 梯度缩放设置
***PyTorch***：若scale_grad_by_freq设置为`True`，会根据单词在mini-batch中出现的频率，对梯度进行放缩。  
***PaddlePaddle***：PaddlePaddle无此功能。
