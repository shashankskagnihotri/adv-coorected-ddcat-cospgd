#!/bin/sh
PYTHON=/work/sagnihot/miniconda3/envs/py39/bin/python

gpu_id=3

dataset=voc2012
exp_name=pspnet50_ddcat
attack_name=cospgd
attack_iterations="4"
attack_epsilon="8"
attack_alpha="0.01"
exp_dir=exp/${dataset}/${exp_name}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
cp tool_test/voc2012/psp_test_ddcat.sh tool_test/voc2012/test_voc_psp_ddcat.py ${config} ${exp_dir}

export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=${gpu_id} $PYTHON -u tool_test/voc2012/test_voc_psp_ddcat.py \
  --config=${config} --attack --gpu_id ${gpu_id} --attack_name ${attack_name} --attack_iterations ${attack_iterations} --attack_epsilon ${attack_epsilon} --attack_alpha ${attack_alpha} \
  2>&1 | tee ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}/test-$now.log


  dataset=voc2012
exp_name=pspnet50_ddcat
attack_name=segpgd
attack_iterations="4"
attack_epsilon="8"
attack_alpha="0.01"
exp_dir=exp/${dataset}/${exp_name}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
cp tool_test/voc2012/psp_test_ddcat.sh tool_test/voc2012/test_voc_psp_ddcat.py ${config} ${exp_dir}

export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=${gpu_id} $PYTHON -u tool_test/voc2012/test_voc_psp_ddcat.py \
  --config=${config} --attack --gpu_id ${gpu_id} --attack_name ${attack_name} --attack_iterations ${attack_iterations} --attack_epsilon ${attack_epsilon} --attack_alpha ${attack_alpha} \
  2>&1 | tee ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}/test-$now.log



dataset=voc2012
exp_name=pspnet50_ddcat
attack_name=BIM
attack_iterations="4"
attack_epsilon="8"
attack_alpha="0.01"
gpu_id=3
exp_dir=exp/${dataset}/${exp_name}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}/testing_no_attack
mkdir -p ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}
cp tool_test/voc2012/psp_test_ddcat.sh tool_test/voc2012/test_voc_psp_ddcat.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=python \
CUDA_VISIBLE_DEVICES=${gpu_id} $PYTHON -u tool_test/voc2012/test_voc_psp_ddcat.py \
  --config=${config}  --gpu_id ${gpu_id} \
  2>&1 | tee ${result_dir}/testing_no_attack/test-$now.log


CUDA_VISIBLE_DEVICES=${gpu_id} $PYTHON -u tool_test/voc2012/test_voc_psp_ddcat.py \
  --config=${config} --attack --gpu_id ${gpu_id} --attack_name ${attack_name} --attack_iterations ${attack_iterations} --attack_epsilon ${attack_epsilon} --attack_alpha ${attack_alpha} \
  2>&1 | tee ${result_dir}/testing_${attack_name}_itrs_${attack_iterations}_eps_${attack_epsilon}_alpha_${attack_iterations}/test-$now.log

