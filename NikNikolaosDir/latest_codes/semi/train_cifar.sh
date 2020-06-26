
seed=$RANDOM
python main.py --dataset 'cifar' \
            --train_batch_size 100 \
            --dev_batch_size 100 \
            --eval_period 500 \
            --size_labeled_data 4000 \
            --max_epochs 1200 \
            --visual_period 5000 \
            --dis_lr 3e-4 \
            --gen_lr 3e-4 \
            --classifier_lr 3e-4 \
            --image_size 32 \
            --n_channels 3 \
            --alpha 0.5 \
            --beta_2 0.1 \
            --beta_1 0.0 \
            --lambda_e 0.1 \
            --lambda_p 0 \
            --p_d_bar_type 'inter' \
            --use_gpu '0' \
            --seed $seed \
            --task_id 0 \
            --affix '' \
            --iter_c 1 \
            --record_file_affix '_0' \
            --save_classifier --update_freq 3
