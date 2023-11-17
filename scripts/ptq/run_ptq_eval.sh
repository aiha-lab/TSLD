# HF
cache_dir="/home/ms/hf_cache"

# Data
dataset_name=wikitext
dataset_config_name=wikitext-2-v1

# Eval config
eval_batch_size=4
torch_dtype=float16
exp_name=None
block_size=512
track_eval_ppl=true
save_model_weight=false

# Quantization config
quantizer=gptq
quant_type=int
name=$3
n_bits_w=$4
learning_rate=$5 
apply_awq=false
group_size=128

output_dir=${name}_wikitext_epoch_5_${n_bits_w}bit_None_lr_${learning_rate}_qerr_init_true_saving


model_name=outputs/opt-$2/${name}_wikitext_epoch_5_${n_bits_w}bit_None_lr_${learning_rate}_qerr_init_true_saving
model_name_sub=opt-$2

echo $model_name
echo $model_name_sub
# model_name=facebook/opt-1.3b
CUDA_VISIBLE_DEVICES=$1 python run_qclm_lora.py \
    --model_name_or_path $model_name --cache_dir $cache_dir \
    --dataset_name $dataset_name --dataset_config_name $dataset_config_name \
    --full_finetune true --quantizer $quantizer --output_dir outputs/${model_name_sub}/$output_dir \
    --per_device_eval_batch_size $eval_batch_size --block_size $block_size \
    --do_eval --bf16 --torch_dtype $torch_dtype --low_cpu_mem_usage True --save_model_weight $save_model_weight \
    --track_eval_ppl $track_eval_ppl --quant_type $quant_type --n_bits_w $n_bits_w --apply_awq $apply_awq --group_size $group_size
    