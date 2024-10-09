echo "[X2Paddle]    Running TensorFlow model converting test..."
find . -name "result.txt" | xargs rm -rf
find . -name "pd_model_dygraph" | xargs rm -rf
find . -name "run.log" | xargs rm -rf
find . -name "run.err" | xargs rm -rf

# use black.list to control CI tests
filename="black.list"
models=$(ls -d */ | grep -v -F -f "$filename")
num_of_models=$(ls -d */ | grep -v -F -f "$filename" | wc -l)

counter=1
for model in $models
do
    echo "[X2Paddle-TensorFlow] ${counter}/${num_of_models} $model ..."
    cd $model
    sh run_benchmark.sh $model 1>run.log 2>run.err &
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

number_lines=$(cat result.txt | wc -l)
failed_line=$(grep -o "Failed"  result.txt|wc -l)
zero=0
if [ $failed_line -ne $zero ]
then
    echo "[ERROR] There are $number_lines results in result.txt, but failed number of models is $failed_line."
    cat result.txt
    #exit -1
else
    echo "All Succeed!"
fi

cd tools
python benchmark_summary.py
cd ..
tar zcvf output.tar.gz output

if [ $failed_line -ne $zero ]
then
    exit -1
fi
