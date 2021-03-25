# PyTorch项目转换示例
## [StarGAN](https://github.com/yunjey/stargan)
### 准备工作
``` shell
# 下载项目
git clone https://github.com/yunjey/stargan.git
# 下载预训练模型
cd stargan
bash download.sh pretrained-celeba-128x128
# 下载数据集
bash download.sh celeba
```
### 转换前修改代码
1. 规避使用TensorBoard，在[config处](https://github.com/yunjey/stargan/blob/master/main.py#L109)设置不使用tensorboard，具体添加代码如下：
``` python
...
parser.add_argument('--lr_update_step', type=int, default=1000)
config = parser.parse_args()
# 第5行为添加不使用tensorboard的相关代码
config.use_tensorboard = False
print(config)
main(config)
```
### 转换
``` shell
cd ../
x2paddle --framework=pytorch_project --project_dir=stargan --save_dir=paddle_project --pretrain_model=stargan/stargan_celeba_128/models/
```
### 转换后代码修改
1. 若需要使用GPU，DataLoader的`num_workers`设置为0，在[config处](https://github.com/yunjey/stargan/blob/master/main.py#L109)设置强制设置`num_workers`，具体添加代码如下：
``` python
...
parser.add_argument('--lr_update_step', type=int, default=1000)
config = parser.parse_args()
config.use_tensorboard = False
# 第6行添加设置num_workers为0
config.num_workers = 0
print(config)
main(config)
```

2. 修改自定义Dataset中的[\_\_getitem\_\_的返回值](https://github.com/yunjey/stargan/blob/master/data_loader.py#L64)，将Tensor修改为numpy，修改代码如下：
``` python
...
class CelebA(data.Dataset):
    ...
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = (self.train_dataset if self.mode == 'train' else self.
                test_dataset)
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        # return self.transform(image), torch.FloatTensor(label)
        # 将原来的return替换为如下12-17行
        out1 = self.transform(image)
        if isinstance(out1, paddle.Tensor):
            out1 = out1.numpy()
        out2 = paddle.Tensor(label)
        if isinstance(out2, paddle.Tensor):
            out2 = out2.numpy()
        return out1, out2
    ...
```

3. 在[Tensor对比操作](https://github.com/yunjey/stargan/blob/master/solver.py#L168)中对Tensor进行判断，判断是否为bool型，如果为bool类型需要强制转换，修改代码如下：
``` python
...
class Solver(object):
    ...
    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        ...
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
            else:
                # 如果为bool型，需要强转为int32，
                # 在17-20行实现
                is_bool = False
                    if str(c_trg.dtype) == "VarType.BOOL":
                        c_trg = c_trg.cast("int32")
                        is_bool = True
                c_trg[:, i] = (c_trg[:, i] == 0) 
                # 如果为bool类型转换为原类型
                # 在23-24行实现
                if is_bool:
                    c_trg = c_trg.cast("bool")
            ...
        ...
    ...
...
```

### 运行训练代码
``` shell
cd paddle_project/stargan
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --celeba_image_dir ./data/celeba/images --attr_path ./data/celeba/list_attr_celeba.txt
```


## [Ultra-Light-Fast-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

### 准备工作
1. 
``` shell
# 下载项目
git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB.git
```
2. 根据Generate VOC format training data set and training process的README.md所示下载数据集，并存放于Ultra-Light-Fast-Generic-Face-Detector-1MB/data/文件夹下。
### 转换前修改代码
1. 将代码中的[或操作符](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py#L153)替换为如下代码：
``` python
...
def hard_negative_mining(loss, labels, neg_pos_ratio):
    ...
    # return pos_mask | neg_mask
    return torch.bitwise_or(pos_mask, neg_mask)
...
```

2. 使自定义的[`DataSet`](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/datasets/voc_dataset.py#L10)继承`torch.utils.data.Dataset`，修改为如下代码：
``` python
...
# class VOCDataset
class VOCDataset(torch.utils.data.Dataset):
    ...
...
```
3. 若需要使用GPU，DataLoader的`num_workers`设置为0，在[train-version-RFB.sh处](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/train-version-RFB.sh#L27)设置强制设置`num_workers`，具体添加代码如下：
```shell
...
  --num_workers \
  #4 \
  0 \
...
```
### 转换
```shell
x2paddle --framework=pytorch_project --project_dir=Ultra-Light-Fast-Generic-Face-Detector-1MB --save_dir=paddle_project 
```
### 运行训练代码
``` shell
cd paddle_project/Ultra-Light-Fast-Generic-Face-Detector-1MB
sh train-version-RFB.sh
```


