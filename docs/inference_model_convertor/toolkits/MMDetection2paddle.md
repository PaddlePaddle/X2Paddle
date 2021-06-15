# MMDetection模型导出为Paddle模型教程

X2Paddle新增对MMDetection模型支持，目前测试支持的列表如下
| 模型  | 来源 | OP版本 | 备注 |
| :---- | :---- | :----- | :--- |
| FCOS | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py) | 11 | 仅支持batch=1推理；模型导出需固定shape |
| FSAF | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py) | 11 | 仅支持batch=1推理；模型导出需固定shape |
| RetinaNet | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | 11 | 仅支持batch=1推理；模型导出需固定shape |
| SSD | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssd300_coco.py) | 11 | 仅支持batch=1推理；模型导出需固定shape |
| YOLOv3 | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py) |11 | 仅支持batch=1；推理模型导出需固定shape |
| Faster R-CNN | [pytorch(mmdetection)](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |11 | 仅支持batch=1；推理模型导出需固定shape |

## 导出教程

### 步骤一、通过MMDetection导出ONNX模型
导出步骤参考文档[MMDetection导出ONNX](https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html)，以COCO数据集训练的YOLOv3为例，导出示例如下
```bash
python tools/deployment/pytorch2onnx.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    --output-file checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 608 608 \
    --mean 0 0 0 \
    --std 255 255 255 \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
```

### 步骤二、通过X2Paddle将ONNX模型转换为Paddle格式
安装X2Paddle最新版本
```
pip install x2paddle
```
使用如下命令转换
```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```
转换后的模型为paddle inference格式，保存在pd_model当中

## 结果测试

<table border="1" class="docutils">
	<tr>
	    <th align="center">Model</th>
	    <th align="center">Config</th>
	    <th align="center">Metric</th>
	    <th align="center">ONNX Runtime</th>
	    <th align="center">Paddle</th>
	</tr >
  <tr >
	    <td align="center">FCOS</td>
	    <td align="center"><code>configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">34</td>
	    <td align="center">33.8</td>
	</tr>
  <tr >
	    <td align="center">FSAF</td>
	    <td align="center"><code>configs/fsaf/fsaf_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">33.7</td>
	    <td align="center">33.7</td>
	</tr>
  <tr >
	    <td align="center">RetinaNet</td>
	    <td align="center"><code>configs/retinanet/retinanet_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">34.1</td>
	    <td align="center">34.1</td>
	</tr>
	<tr >
	    <td align="center" align="center" >SSD</td>
	    <td align="center" align="center"><code>configs/ssd/ssd300_coco.py</code></td>
	    <td align="center" align="center">Box AP</td>
	    <td align="center" align="center">25.6</td>
	    <td align="center" align="center">25.6</td>
	</tr>
  <tr >
	    <td align="center">YOLOv3</td>
	    <td align="center"><code>configs/yolo/yolov3_d53_mstrain-608_273e_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">31.1</td>
	    <td align="center">31.1</td>
	</tr>
  <tr >
	    <td align="center">Faster R-CNN</td>
	    <td align="center"><code>configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">34.8</td>
	    <td align="center">34.8</td>
	</tr>
</table>

Notes:

- 上述AP均为固定shape进行测试，除SSD的shape为300x300、YOLOv3为608x608之外，其他shape均为800x1216
