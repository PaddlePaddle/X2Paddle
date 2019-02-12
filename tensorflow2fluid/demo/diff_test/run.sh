rm -rf fluid_vgg_19.result
python vgg_19_infer.py
echo "paddle fluid vgg_19 model"
python diff.py fluid_vgg_19.result tf_vgg_19.result

rm -rf fluid_vgg_16.result
python vgg_16_infer.py
echo "paddle fluid vgg_16 model"
python diff.py fluid_vgg_16.result tf_vgg_16.result

rm -rf fluid_resnet_v1_50.result
python resnet_v1_50_infer.py
echo "paddle fluid resnet_v1_50 model"
python diff.py fluid_resnet_v1_50.result tf_resnet_v1_50.result

rm -rf fluid_resnet_v1_101.result
python resnet_v1_101_infer.py
echo "paddle fluid resnet_v1_101 model"
python diff.py fluid_resnet_v1_101.result tf_resnet_v1_101.result
