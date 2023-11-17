dataset=ptb
lr=1E-4
epoch=3
model_size=$2

# FP Task-Specific Fine-Tuning
bash scripts/ft_qat/run_fp_ft_ptb.sh $1 $model_size $lr $epoch

ft_name=fullFT_${dataset}_epoch_${epoch}_FP_lr_${lr}

# QAT with KD (initialized with Fine-Tuned FP model)
lr=1E-4
epoch=30

kd_option=$3 # [gt, logit, l2l, logit_gt, tsld]
bash scripts/ft_qat/run_qat_kd_ptb.sh $1 ${model_size} ${ft_name} ${lr} ${epoch} ${kd_option}

 