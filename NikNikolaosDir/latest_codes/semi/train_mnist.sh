
seed=$RANDOM
python main.py --dataset 'mnist' \
            --train_batch_size 100 \
            --dev_batch_size 100 \
            --eval_period 600 \
            --size_labeled_data 100 \
            --max_epochs 1000 \
            --visual_period 20000 \
            --dis_lr 3e-3 \
            --gen_lr 3e-3 \
            --classifier_lr 3e-3 \
            --image_size 28 \
            --n_channels 1 \
            --alpha 0.8 \
            --beta_2 0.3 \
            --beta_1 0.0 \
            --lambda_e 0.0 \
            --lambda_p 0.0 \
            --p_d_bar_type 'inter' \
            --use_gpu '0' \
            --seed $seed \
            --task_id 0 \
            --affix '' \
            --iter_c 1 \
            --record_file_affix '_0' --update_freq 1

