#!/bin/bash
set -x

torch_version=${1}
cd torch

find . -name "result.txt" | xargs rm -rf
if [! -f ${result.txt}];then
   touch result.txt
fi
echo "===================torch-${torch_version}===================" >> result.txt

logs_path=logs/${torch_version}
mkdir -p ${logs_path}
 

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
  fail=`cat ${log_name} | grep 'FAILED' | wc -l`
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
