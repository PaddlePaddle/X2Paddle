## [Ultra-Light-Fast-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

### 准备工作
1. 下载项目
``` shell
# 下载项目
git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB.git
cd Ultra-Light-Fast-Generic-Face-Detector-1MB
git checkout 492a02471671b49c56be8d90cda54c94749d2980
```
2. 根据[Generate VOC format training data set and training process的README.md](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB#generate-voc-format-training-data-set-and-training-process)所示下载数据集，并存放于Ultra-Light-Fast-Generic-Face-Detector-1MB/data/文件夹下。
### 第一步：转换前代码预处理
1. 将代码中的[或操作符](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py#L153)替换为如下代码：
``` python
...
def hard_negative_mining(loss, labels, neg_pos_ratio):
    ...
    # return pos_mask | neg_mask
    return torch.bitwise_or(pos_mask, neg_mask)
...
```

2. 使自定义的[`DataSet`](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/datasets/voc_dataset.py#L10)继承`torch.utils.data.Dataset`，同时由于代码未导入torch，要添加相关导入的包，修改为如下代码：
``` python
...
# 导入torch
import torch
...
# class VOCDataset
class VOCDataset(torch.utils.data.Dataset):
    ...
...
```
3. 将[数据预处理](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py#L126)Tensor与int型对比，修改为Tensor与Tensor对比，修改如下：
``` python
...
def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    ...
    # labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    # 将原来的赋值修改为7-8行
    iou_threshold_tensor = torch.full_like(best_target_per_prior, iou_threshold)
    labels[best_target_per_prior < iou_threshold_tensor] = 0
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels
...
```

### 第二步：转换
```shell
x2paddle --convert_torch_project --project_dir=Ultra-Light-Fast-Generic-Face-Detector-1MB --save_dir=paddle_project
```
### 第三步：转换后代码后处理
**需要修改的文件位于paddle_project文件夹中，其中文件命名与原始Ultra-Light-Fast-Generic-Face-Detector-1MB文件夹中文件命名一致。**
1. DataLoader的`num_workers`设置为0，在转换后的[train-version-RFB.sh处](https://github.com/SunAhong1993/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/paddle/train-version-RFB.sh#L27)设置强制设置`num_workers`，具体添加代码如下：
```shell
...
  --num_workers \
  #4 \
  0 \
...
```
2.修改自定义Dataset中的[\_\_getitem\_\_的返回值](https://github.com/SunAhong1993/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/paddle/vision/datasets/voc_dataset.py#L56)，将Tensor修改为numpy，修改代码如下：
``` python
...
class VOCDataset(data.Dataset):
    ...
    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # return image, boxes, labels
        # 将原来的return替换为如下17行
        return image.numpy(), boxes.numpy(), labels.numpy()
    ...
```

### 运行训练代码
``` shell
cd paddle_projec
sh train-version-RFB.sh
```
***转换后的代码可在[这里](https://github.com/SunAhong1993/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/paddle)进行查看。***
