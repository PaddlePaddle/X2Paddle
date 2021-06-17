## [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
### 准备工作
``` shell
# 下载项目
git clone https://github.com/clovaai/CRAFT-pytorch.git
cd CRAFT-pytorch
git checkout e332dd8b718e291f51b66ff8f9ef2c98ee4474c8
```
模型与数据可根据[原repo](https://github.com/clovaai/CRAFT-pytorch#test-instruction-using-pretrained-model)相关信息进行下载，可将模型存放于新建文件夹`./weight`，将数据存放于新建文件夹`./data`。

###  转换
``` shell
cd ../
x2paddle --convert_torch_project --project_dir=CRAFT-pytorch --save_dir=paddle_project --pretrain_model=CRAFT-pytorch/weights/
```

### 运行训练代码
``` shell
cd paddle_project
python test.py --trained_model=weights/craft_mlt_25k.pdiparams --test_folder=data
```

***转换后的代码可在[这里](https://github.com/SunAhong1993/CRAFT-pytorch/tree/paddle)进行查看。***
