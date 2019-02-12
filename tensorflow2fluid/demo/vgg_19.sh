export CUDA_VISIBLE_DEVICES=-1

wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar xzvf vgg_19_2016_08_28.tar.gz
python export_to_checkpoint.py --model vgg_19 --ckpt_file vgg_19.ckpt --save_dir vgg_19_checkpoint
rm vgg_19_2016_08_28.tar.gz vgg_19.ckpt

tf2fluid --meta_file vgg_19_checkpoint/model.meta \
         --ckpt_dir vgg_19_checkpoint \
         --in_nodes inputs \
         --input_shape None,224,224,3 \
         --output_nodes vgg_19/fc8/squeezed \
         --save_dir paddle_vgg_19
