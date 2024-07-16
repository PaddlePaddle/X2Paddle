echo "[X2Paddle]    Running ONNX model converting test..."
if [ ! -d "dataset/" ]; then	
  wget -nc https://x2paddle.bj.bcebos.com/test_benchmark/ONNX/dataset.tar.gz
  tar xzvf dataset.tar.gz
fi
find . -name "result.txt" | xargs rm -rf
find . -name "pd_model_dygraph" | xargs rm -rf
find . -name "run.log" | xargs rm -rf
find . -name "run.err" | xargs rm -rf

num_of_models=$(ls -d */ | grep -v 'tools' | grep -v 'output' | wc -l)
num_of_onnx_files=$(find . -name "*.onnx" | wc -l)

#if [ $num_of_onnx_files -ne $num_of_models ]
#then 
#    echo $num_of_onnx_files
#    echo $num_of_models
#    echo "[ERROR] num_of_onnx_files != num_of_models"
#    exit -1
#fi

counter=1
for model in $(ls -d */ | grep -v 'tools' | grep -v 'output')
do
    echo "[X2Paddle-ONNX] ${counter}/${num_of_models} $model ..."
    cd $model
    sh run_convert.sh $model 1>run.log 2>run.err &
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
for model in $(ls -d */ | grep -v 'tools' | grep -v 'output')
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
