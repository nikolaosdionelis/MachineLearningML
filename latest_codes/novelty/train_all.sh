# train vae

for((i=0;i<10;i=i+1))
do
    python main.py --todo 'vae' --dataset 'cifar' --save_vae --max_epochs 500 --use_gpu '0' --LAMBDA 1e-4 \
    --train_class $i --task_id 'vae' --feature_extractor 'vae'
done

# train gan

for((i=0;i<10;i=i+1))
do
    python main.py --todo 'gan' --dataset 'cifar' --save_gan --max_epochs 500 --use_gpu '0' \
    --vae_checkpoint 'model/cifar_vae/'$i'/vae_checkpoint.pt' --train_class $i --task_id 'gan_vae' \
    --p_d_bar_type 'normal' --beta_2 1.0 --visual_period 2000 --eval_period 500 --iter_c 5  \
    --train_batch_size 100 --feature_extractor 'vae' --alpha 0.9 --lambda_feat 0.8
done


# tune decoder
dataset='cifar'
task_id='finetune_vae'

for((i=0;i<10;i=i+1))
do
    python main.py --todo 'finetune' --dataset $dataset --save_vae --max_epochs 600 --use_gpu '0' \
    --vae_checkpoint 'model/cifar_vae/'$i'/vae_checkpoint.pt' --train_class $i --task_id $task_id \
    --gan_checkpoint 'model/cifar_gan_vae/'$i'/gan_checkpoint.pt' --dis_lr 1e-2 --feature_extractor 'vae' \
    --feature_size 128 --lambda_out 0.5 --threshold 1.5
done
