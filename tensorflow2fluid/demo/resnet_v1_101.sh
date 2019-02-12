export CUDA_VISIBLE_DEVICES=-1

wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar xzvf resnet_v1_101_2016_08_28.tar.gz
python export_to_checkpoint.py --model resnet_v1_101 --ckpt_file resnet_v1_101.ckpt --save_dir resnet_v1_101_checkpoint
rm resnet_v1_101_2016_08_28.tar.gz resnet_v1_101.ckpt

tf2fluid --meta_file resnet_v1_101_checkpoint/model.meta \
         --ckpt_dir resnet_v1_101_checkpoint \
         --in_nodes inputs \
         --input_shape None,224,224,3 \
         --output_nodes resnet_v1_101/predictions/Softmax \
         --save_dir paddle_resnet_v1_101
