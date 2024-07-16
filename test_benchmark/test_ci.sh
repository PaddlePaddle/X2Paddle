#docker 命令参考: docker/run.sh

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH};
# build x2paddle
cd ..;
python setup.py install;
 if [ "$?" != "0" ]; then
    echo -e "\033[33m build x2-paddle failed! \033[0m";
    exit -1;
fi;

cd test_benchmark/Caffe;
#bash convert.sh;

#cd ../PyTorch;
#bash convert.sh;

cd ../ONNX;
bash convert.sh;
#
#cd ../TensorFlow;
#mv KerasBert ..; #暂时取消，会hang
#bash convert.sh;
