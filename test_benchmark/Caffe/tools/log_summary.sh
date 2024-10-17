#/bin/bash
set -x

if [ -d '../output' ];then
    rm -rf ../output
fi
mkdir ../output

cd ..

# use black.list to control CI tests
filename="black.list"
models=$(ls -d */ | grep -v -F -f "$filename")

for model in $models
do
    cp ${model}/run.log output/${model}_run.log
    cp ${model}/run.err output/${model}_run.err
done

tar zcvf output.tar.gz output
