#!/bin/bash

set -x

cd ..
home_path=`pwd`
echo ${home_path}
test_path=${home_path}/X2Paddle/tests/torch
if [ ! -d "${test_path}" ];then
   echo "the torch_autoscan file not exist"
   exit 0
fi

python -m pip uninstall torch -y
python -m pip uninstall torchvision -y

for i in {7..12}
do
  ver=$[`expr ${i}+1`]
  torchvision_version="0.${ver}.0"
  #echo ${torchvision_version}
  torch_version="1.${i}.0"
  python -m pip install torch==${torch_version}
  python -m pip install torchvision==${torchvision_version}
  echo "successfully install torch ${torch_version}"

  bash run_autoscan_torch.sh ${torch_version}
done
