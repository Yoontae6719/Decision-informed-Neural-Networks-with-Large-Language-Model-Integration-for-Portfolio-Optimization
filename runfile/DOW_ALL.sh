export HF_HOME=/home/work/DBLM/huggingface_cache


model_name=DBLM2
train_epochs=100
learning_rate=0.01
llama_layers=256

master_port=00091
num_process=4
batch_size=16

comment='DOW'


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path DOW30_ret.csv \
  --model $model_name \
  --data DOW30 \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value A \
  --num_enc_l 1 \
  --num_hidden 256 \
  --d_ff 12 \
  --enc_in 30 \
  --dec_in 30 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait




accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path DOW30_ret.csv \
  --model $model_name \
  --data DOW30 \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value HC \
  --num_enc_l 1 \
  --num_hidden 256 \
  --d_ff 12 \
  --enc_in 30 \
  --dec_in 30 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path DOW30_ret.csv \
  --model $model_name \
  --data DOW30 \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value C \
  --num_enc_l 1 \
  --num_hidden 256 \
  --d_ff 12 \
  --enc_in 30 \
  --dec_in 30 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait



accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path DOW30_ret.csv \
  --model $model_name \
  --data DOW30 \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value B \
  --num_enc_l 1 \
  --num_hidden 256 \
  --d_ff 12 \
  --enc_in 30 \
  --dec_in 30 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait



accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
  --is_training 1 \
  --root_path ./preprocessing/ \
  --data_path DOW30_ret.csv \
  --model $model_name \
  --data DOW30 \
  --seq_len 252 \
  --pred_len 22 \
  --loss_alpha 0.4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_heads 2 \
  --lambda_value HA \
  --num_enc_l 1 \
  --num_hidden 256 \
  --d_ff 12 \
  --enc_in 30 \
  --dec_in 30 \
  --des 'Exp' \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

wait
