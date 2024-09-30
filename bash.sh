#!/bin/bash

net_arch='sar_v1'
seed=42
gpu=0
config='configs/diffuser_inv_halfcheetah/diffuser_inv_halfcheetah_mdexpert.py'
run_name='haha'
split_rate=0.01
start_time=$(date)
save_path=''
echo "脚本开始时间: ${start_time}"


# 运行第一个程序
ckpt_path=$(python train.py --gpu $gpu --config $config --run_name $run_name --split_rate $split_rate --use_condition --net_arch $net_arch --seperate_encoding --seed $seed | grep -oP '(?<=ckpt_path: ).*')
# 等待第一个程序结束
wait

echo "${ckpt_path}"
# # 运行第二个程序
# python train.py --opt 'gen' --gen_trajs --gpu $gpu --config $config --run_name $run_name --split_rate $split_rate --use_condition --net_arch $net_arch --seperate_encoding --seed $seed --gen_source 'sim' --ckpt_path
# # 等待第二个程序结束
# wait

# # 切换目录
# cd D_2_C/example/benchmark/
# # 等待目录切换完成
# wait

# # 运行第三个程序
# python program3.py
# # 等待第三个程序结束
# wait
