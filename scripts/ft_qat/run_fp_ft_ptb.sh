# Model 
model_name_sub=opt-$2
model_name=facebook/$model_name_sub

# HF
cache_dir="/home/ms/hf_cache"

# Data
dataset_name=ptb_text_only

# Training config
train_batch_size=4
gradient_accumulation_steps=1
eval_batch_size=4
learning_rate=$3
torch_dtype=bfloat16
num_train_epochs=$4
exp_name=None
block_size=512

evaluation_strategy=epoch
save_strategy=no

track_eval_ppl=true
save_model_weight=true

# Quantization config
quantizer=None
quant_type=int
n_bits_w=16

full_finetune=true
name=fullFT

output_dir=${name}_ptb_epoch_${num_train_epochs}_FP_lr_${learning_rate}
echo $model_name
echo $output_dir
CUDA_VISIBLE_DEVICES=$1 python run_qclm_full.py \
    --model_name_or_path $model_name --cache_dir $cache_dir \
    --dataset_name $dataset_name \
    --gradient_accumulation_steps $gradient_accumulation_steps --per_device_train_batch_size $train_batch_size --per_device_eval_batch_size $eval_batch_size \
    --evaluation_strategy $evaluation_strategy --save_strategy $save_strategy --block_size 512  --learning_rate $learning_rate --save_model_weight $save_model_weight \
    --do_train --bf16 --torch_dtype $torch_dtype --low_cpu_mem_usage True --output_dir outputs/ptb/qat_kd/${model_name_sub}/$output_dir --overwrite_output_dir \
    --full_finetune $full_finetune --num_train_epochs $num_train_epochs --track_eval_ppl $track_eval_ppl \
    --quantizer $quantizer --quant_type $quant_type --n_bits_w $n_bits_w 