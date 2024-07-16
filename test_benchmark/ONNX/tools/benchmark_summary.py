import os
import logging

def benchmark_info(file_name, model_name, inference_success, inference_failed, compare_success, compare_failed):
    if os.path.exists(file_name):
        inference_success.append(model_name)
        with open(file_name, 'r') as f:
            for line in f.readlines():
                if "Compare Successed" in line:
                    compare_success.append(model_name)
                    break
                elif "Compare Failed" in line:
                    compare_failed.append(model_name)
                    break
    else:
        inference_failed.append(model_name)

def main():
    all_model_names = list()
    with open('../result.txt','r') as f:
        for line in f.readlines():
            if "======" in line:
                all_model_names.append(line.strip().split(':')[0][6:])
    # trt
    inference_success_trt = list()
    inference_failed_trt = list()
    compare_success_trt = list()
    compare_failed_trt = list()
    # gpu
    inference_success_gpu = list()
    inference_failed_gpu = list()
    compare_success_gpu = list()
    compare_failed_gpu = list()
    # mkldnn
    inference_success_mkldnn = list()
    inference_failed_mkldnn = list()
    compare_success_mkldnn = list()
    compare_failed_mkldnn = list()
    # cpu
    inference_success_cpu = list()
    inference_failed_cpu = list()
    compare_success_cpu = list()
    compare_failed_cpu = list()
    for model_name in all_model_names:
        # gpu + trt
        file_name = '../output/' + model_name + '_gpu_tensorrt.log'
        benchmark_info(file_name, model_name, inference_success_trt, inference_failed_trt, compare_success_trt, compare_failed_trt)
        # gpu
        file_name = '../output/' + model_name + '_gpu.log'
        benchmark_info(file_name, model_name, inference_success_gpu, inference_failed_gpu, compare_success_gpu, compare_failed_gpu)
        # cpu_mkldnn
        file_name = '../output/' + model_name + '_cpu_mkldnn.log'
        benchmark_info(file_name, model_name, inference_success_mkldnn, inference_failed_mkldnn, compare_success_mkldnn, compare_failed_mkldnn)
        # cpu
        file_name = '../output/' + model_name + '_cpu.log'
        benchmark_info(file_name, model_name, inference_success_cpu, inference_failed_cpu, compare_success_cpu, compare_failed_cpu)
    # Init logger
    log_output = '../output/benchmark_summary.log'
    FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('benchmark_summary')
    fh = logging.FileHandler(log_output, encoding='UTF-8')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(FORMAT)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info(
        f"Paddle Inference benchmark log will be saved to {log_output}")
    logger.info("\n")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        "                     trt     gpu     mkldnn      cpu    ")
    logger.info(
        f"inference success    {len(inference_success_trt)}      {len(inference_success_gpu)}       {len(inference_success_mkldnn)}         {len(inference_success_cpu)}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"inference failed     {len(inference_failed_trt)}      {len(inference_failed_gpu)}        {len(inference_failed_mkldnn)}           {len(inference_failed_cpu)}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"compare success      {len(compare_success_trt)}      {len(compare_success_gpu)}       {len(compare_success_mkldnn)}         {len(compare_success_cpu)}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"compare failed       {len(compare_failed_trt)}       {len(compare_failed_gpu)}        {len(compare_failed_mkldnn)}           {len(compare_failed_cpu)}")
    logger.info(
        "--------------------------------------------------------")
    logger.info("\n")
    logger.info(
        "*****************detail information*************************")
    logger.info(
        f"Models list of run trt failed: {inference_failed_trt}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of trt accu compare failed: {compare_failed_trt}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of run gpu failed: {inference_failed_gpu}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of gpu accu compare failed: {compare_failed_gpu}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of run mkldnn failed: {inference_failed_mkldnn}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of mkldnn accu compare failed: {compare_failed_mkldnn}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of run cpu failed: {inference_failed_cpu}")
    logger.info(
        "--------------------------------------------------------")
    logger.info(
        f"models list of cpu accu compare failed: {compare_failed_cpu}")
    logger.info(
        "--------------------------------------------------------")
    
    
    
if __name__ == '__main__':
    main()
