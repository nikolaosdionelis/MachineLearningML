# tune decoder

dataset='cifar'
task_id='finetune_vae'

start=0
end=10
for((i=$start;i<$end;i=i+1))
do
    python main.py --todo 'test' --dataset $dataset --train_class $i --use_gpu '0' \
    --classifier_checkpoint 'model/'$dataset'_'$task_id'/'$i'/classifier_checkpoint.pt' \
    --vae_checkpoint 'model/'$dataset'_'$task_id'/'$i'/vae_checkpoint.pt' --task_id $task_id --feature_extractor 'vae' --feature_size 128
done

test_file='log/'$dataset'_'$task_id'/record.txt'

rm $test_file

for((i=$start;i<$end;i=i+1))
do
    cat 'log/'$dataset'_'$task_id'/'$i'/auc.txt' >> $test_file

done