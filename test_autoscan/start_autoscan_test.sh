#!/bin/bash

#test onnx
bash run_autoscan_onnx.sh

#test torch
bash start_autoscan_torch.sh

#cat result
cd ..
path=`pwd`

if [ -d "${path}/X2Paddle/tests/onnx" ];then
   cd ${path}/X2Paddle/tests/onnx
   cat result.txt
   result_onnx=`cat result.txt | grep 'FAILED' | wc -l`
else
   result_onnx=0
fi 

if [ -d "${path}/X2Paddle/tests/torch" ];then
   cd ${path}/X2Paddle/tests/torch
   cat result.txt
   result_torch=`cat result.txt | grep 'FAILED' | wc -l`
else
   result_torch=0
fi

if [ ${result_onnx} -ne 0 ];then
  exit 1
fi

if [ ${result_torch} -ne 0 ];then
  exit 1
fi

