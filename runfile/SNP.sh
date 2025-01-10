export HF_HOME=/home/work/DBLM/huggingface_cache


model_name=DBLM2
train_epochs=100
learning_rate=0.01
llama_layers=128

master_port=00092
num_process=4
batch_size=16

comment='SNP'


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path SNP100_ret_1129.csv \
  --model $model_name \
  --data SNP_Multi \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value B \
  --num_enc_l 1 \
  --num_hidden 128 \
  --d_ff 12 \
  --enc_in 50 \
  --dec_in 50 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait


