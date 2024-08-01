#!/bin/bash

#test onnx
bash run_autoscan_onnx.sh

#test torch
bash start_autoscan_torch.sh

result_onnx=`cat onnx/result.txt | grep 'FAILED' | wc -l`
result_torch=`cat torch/result.txt | grep 'FAILED' | wc -l`

if [ ${result_onnx} -ne 0 ];then
  exit 1
fi

if [ ${result_torch} -ne 0 ];then
  exit 1
fi
