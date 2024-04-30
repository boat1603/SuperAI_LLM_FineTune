#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script


module restore
module load Mamba
module load cudatoolkit/22.7_11.7
module load gcc/10.3.0

conda deactivate
conda activate ./env

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0


accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision fp16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    scripts/train.py \
        --model_name_or_path /project/lt900048-ai24tn/models/new5558/tinyllama_1b-200000 \
        --train_data_path /project/lt900048-ai24tn/datasets/Thaweewat/instruct-qa-thai-combined-jsonl/train_instruct_qa.jsonl \
        --eval_data_path /project/lt900048-ai24tn/datasets/Thaweewat/instruct-qa-thai-combined-jsonl/val_instruct_qa.jsonl \
        --data_seed 42 \
        --model_max_length 2048 \
        --bf16 True \
        --output_dir ./checkpoint/ \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --save_strategy "steps" \
        --save_steps 700 \
        --save_total_limit 5 \
        --logging_strategy 'steps' \
        --logging_steps 1 \
        --logging_first_step True \
        --learning_rate 8e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --deepspeed ./deepspeed_config/deepspeed_3.json \
        --gradient_checkpointing True \
        --tf32 True \
        # --checkpoint ...
        