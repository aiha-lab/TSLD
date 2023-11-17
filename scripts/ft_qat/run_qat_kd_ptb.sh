# Model 
model_name_sub=opt-$2
model_name=outputs/ptb/qat_kd/${model_name_sub}/$3

# HF
cache_dir="/home/ms/hf_cache"

# Data
dataset_name=ptb_text_only

# Training config
train_batch_size=4
gradient_accumulation_steps=1
eval_batch_size=4
learning_rate=$4
torch_dtype=bfloat16
num_train_epochs=$5
block_size=512

evaluation_strategy=epoch
save_strategy=no

track_eval_ppl=true
save_model_weight=false

# Quantization config
quantizer=fake
quant_type=int
n_bits_w=2
per_tensor=false # per_channel
learned_scale=false # For Advanced PACT option
init_scale=1.0

# KD setting
full_finetune=true
kd_option=$6
kd_qat_full=true
kd_gt=false
kd_pred=false
kd_l2l=false
kd_tsld=false

name=fullQAT

if [ $kd_option == tsld ] ; then
    kd_gt=true
    kd_tsld=true
    name=fullQAT_tsld
fi
if [ $kd_option == logit ] ; then
    kd_pred=true
    name=fullQAT_logit
fi
if [ $kd_option == l2l ] ; then
    kd_pred=true
    kd_l2l=true
    name=fullQAT_l2l
fi
if [ $kd_option == gt ] ; then
    kd_gt=true
    name=fullQAT_gt
fi
if [ $kd_option == logit_gt ] ; then
    kd_gt=true
    kd_pred=true
    name=fullQAT_logit_gt
fi

output_dir=${name}_KD_ptb_epoch_${num_train_epochs}_${n_bits_w}bit_lr_${learning_rate}_per_tensor_${per_tensor}
echo $model_name
echo $output_dir
CUDA_VISIBLE_DEVICES=$1 python run_qclm_full.py \
    --model_name_or_path $model_name --cache_dir $cache_dir \
    --dataset_name $dataset_name \
    --gradient_accumulation_steps $gradient_accumulation_steps --per_device_train_batch_size $train_batch_size --per_device_eval_batch_size $eval_batch_size \
    --evaluation_strategy $evaluation_strategy --save_strategy $save_strategy --block_size 512  --learning_rate $learning_rate --save_model_weight $save_model_weight \
    --do_train --do_eval --bf16 --torch_dtype $torch_dtype --low_cpu_mem_usage True --output_dir outputs/ptb/qat_kd/${model_name_sub}/$output_dir --overwrite_output_dir \
    --full_finetune $full_finetune --num_train_epochs $num_train_epochs --track_eval_ppl $track_eval_ppl --logging_steps 100 --learned_scale $learned_scale --init_scale $init_scale \
    --quantizer $quantizer --quant_type $quant_type --n_bits_w $n_bits_w --per_tensor $per_tensor --kd_qat_full $kd_qat_full --kd_gt $kd_gt --kd_pred $kd_pred --kd_l2l $kd_l2l --kd_tsld $kd_tsld