## [StarGAN](https://github.com/yunjey/stargan)
### 准备工作
``` shell
# 下载项目
git clone https://github.com/yunjey/stargan.git
cd stargan
git checkout 30867d6f85a3bb99c38ae075de651004747c42d4
# 下载预训练模型
bash download.sh pretrained-celeba-128x128
# 下载数据集
bash download.sh celeba
```
### 第一步：转换前代码预处理
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
###  第二步：转换
``` shell
cd ../
x2paddle --convert_torch_project --project_dir=stargan --save_dir=paddle_project --pretrain_model=stargan/stargan_celeba_128/models/
```
【注意】此示例中的`pretrain_model`是训练后的PyTorch模型，转换后则为PaddlePaddle训练后的模型，用户可修改转换后代码将其作为预训练模型，也可直接用于预测。
### 第三步：转换后代码后处理
**需要修改的文件位于paddle_project文件夹中，其中文件命名与原始stargan文件夹中文件命名一致。**
1. DataLoader的`num_workers`设置为0，在[config处](https://github.com/SunAhong1993/stargan/blob/paddle/main.py#L116)设置强制设置`num_workers`，具体添加代码如下：
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

2. 修改自定义Dataset中的[\_\_getitem\_\_的返回值](https://github.com/SunAhong1993/stargan/blob/paddle/data_loader.py#L63)，将Tensor修改为numpy，修改代码如下：
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
        # return self.transform(image), torch2paddle.create_float32_tensor(label)
        # 将原来的return替换为如下12-17行
        out1 = self.transform(image)
        if isinstance(out1, paddle.Tensor):
            out1 = out1.numpy()
        out2 = torch2paddle.create_float32_tensor(label)
        if isinstance(out2, paddle.Tensor):
            out2 = out2.numpy()
        return out1, out2
    ...
```

3. 在[Tensor对比操作](https://github.com/SunAhong1993/stargan/blob/paddle/solver.py#L156)中对Tensor进行判断，判断是否为bool型，如果为bool类型需要强制转换，修改代码如下：
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
                # 如果为非int型，需要强转为int32，
                # 在18-22行实现
                # c_trg[:, i] = (c_trg[:, i] == 0)
                c_trg = c_trg.cast("int32")
                c_trg_tmp = paddle.zeros_like(c_trg)
                paddle.assign(c_trg, c_trg_tmp)
                c_trg_tmp = c_trg_tmp.cast("bool")
                c_trg_tmp[:, i] = c_trg[:, i] == 0
                c_trg = c_trg_tmp
            ...
        ...
    ...
...
```

### 运行训练代码
``` shell
cd paddle_project
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --celeba_image_dir ./data/celeba/images --attr_path ./data/celeba/list_attr_celeba.txt
```

***转换后的代码可在[这里](https://github.com/SunAhong1993/stargan/tree/paddle)进行查看。***
