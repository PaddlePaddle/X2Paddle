#!/bin/bash
set -x

cd onnx

find . -name "result.txt" | xargs rm -rf
touch result.txt
echo "===================onnx==================" >> result.txt
logs_path=logs
mkdir -p ${logs_path}
#rm -rf logs_path/*

ls | grep test_auto_scan > test_file.txt
test_num=`ls | grep test_auto_scan | wc -l`
echo "The num of test_file is: ${test_num}" >> result.txt

while read line;
do
   file_str=$file_str,$line;
done < test_file.txt
file_arr=(${file_str//,/ })

for var in ${file_arr[@]}
do
  log_name=${logs_path}/${var}.log
  echo " Now start test: ${var}"
  python ${var} > ${log_name} 2>&1

done

for var in ${file_arr[@]}
do
  log_name=${logs_path}/${var}.log
  success=`cat ${log_name} | grep 'Run Successfully!' | wc -l`
  echo ${success}
  fail=`cat ${log_name} | grep 'FAILED' | wc -l`
  echo ${fail}
  if [ ${success} -ne 0 -a ${fail} -eq 0 ];then
        echo "${var}:Run Successfully!" >> result.txt
  fi

  if [ ${fail} -ne 0 ];then
        echo "${var}:FAILED" >> result.txt
        cat ${log_name}
  fi

done

tar zcvf logs.tar.gz logs
cd ..
