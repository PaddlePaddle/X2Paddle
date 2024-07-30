echo "[X2Paddle]    Running PyTorch model converting test..."
if [ ! -d "dataset/" ]; then
  wget -nc https://x2paddle.bj.bcebos.com/test_benchmark/PyTorch/dataset.tar.gz
  tar xzvf dataset.tar.gz
fi
# 后续打包进 dataset.tar.gz
if [ ! -d "dataset/SwinTransformer" ]; then
  cd dataset
  wget -nc https://x2paddle.bj.bcebos.com/test_benchmark/PyTorch/SwinTransformer.tar.gz
  tar xzvf SwinTransformer.tar.gz
  cd ..
fi
find . -name "result.txt" | xargs rm -rf
find . -name "pd_model" | xargs rm -rf
find . -name "pd_model_trace" | xargs rm -rf
find . -name "pd_model_script" | xargs rm -rf
find . -name "run.log" | xargs rm -rf
find . -name "run.err" | xargs rm -rf

models=$(ls -d */ | grep -v 'tools' | grep -v 'output' | grep -v 'dataset' | grep -v 'MockingBird')
num_of_models=$(ls -d */ | grep -v 'tools' | grep -v 'output' | grep -v 'dataset' | grep -v 'MockingBird' | wc -l)
num_of_pb_files=`expr $(find . -name "convert.py" | wc -l) + $(find . -name "convert_trace.py" | grep -v 'MockingBird' | wc -l)`

if [ $num_of_pb_files -ne $num_of_models ]
then
    echo "[ERROR] num_of_pb_files != num_of_models"
    exit -1
fi

counter=1
for model in $models
do
    echo "[X2Paddle-PyTorch] ${counter}/${num_of_models} $model ..."
    cd $model
    sh run_convert.sh 1>run.log 2>run.err &
    cd ..
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait
rm -rf result.txt
touch result.txt
for model in $models
do
    cat ${model}/result.txt >> ./result.txt
done

cd tools
sh log_summary.sh
cd ..

number_lines=$(cat result.txt | wc -l)
failed_line=$(grep -o "Failed"  result.txt|wc -l)
zero=0
if [ $failed_line -ne $zero ]
then
    echo "[ERROR] There are $number_lines results in result.txt, but failed number of models is $failed_line."
    cat ./result.txt
    exit -1
else
    echo "All Succeed!"
fi
