#export CUDA_VISIBLE_DEVICES=4

model_name=TimesNet

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/nepse/ \
  --data_path nepse.csv \
  --model_id nepse \
  --model $model_name \
  --data NEPSE \
  --features M \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1