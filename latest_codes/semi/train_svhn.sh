seed=$RANDOM
python main.py --dataset 'svhn' \
               --train_batch_size 100 \
               --dev_batch_size 100 \
               --eval_period 730 \
               --size_labeled_data 1000 \
               --max_epochs 1000 \
               --visual_period 5000 \
               --dis_lr 4e-3 \
               --gen_lr 1e-3 \
               --classifier_lr 1e-3 \
               --min_lr 1e-4 \
               --image_size 32 \
               --n_channels 3 \
               --alpha 0.8 \
               --beta_2 0.1 \
               --beta_1 0.0 \
               --lambda_e 0.3 \
               --lambda_p 0 \
               --p_d_bar_type 'inter' \
               --use_gpu '0' \
               --seed $seed \
               --task_id $i \
               --affix '' \
               --iter_c 1 \
               --record_file_affix '_0' --update_freq 10


