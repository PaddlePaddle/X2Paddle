#/bin/bash
set -x

if [ -d '../output' ];then
    rm -rf ../output
fi
mkdir ../output

cd ..
for model in `ls -d */ | grep -v 'tools' | grep -v 'output' | awk -F '/' '{print $1}'`
do
    cp ${model}/run.log output/${model}_run.log
    cp ${model}/run.err output/${model}_run.err
done

tar zcvf output.tar.gz output
