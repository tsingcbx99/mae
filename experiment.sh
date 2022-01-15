# ======================================================================================================================
# baseline
# ======================================================================================================================

# 1% data 128 batch size
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 16 \
    --model vit_base_patch16 \
    --finetune pretrain/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path data/imagenet \
    --dataset_split dataset_split/1percent.pth \
    --output_dir labeled_only/1percent \
    --log_dir labeled_only/1percent

# 10% data 128 batch size
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 16 \
    --model vit_base_patch16 \
    --finetune pretrain/mae_pretrain_vit_base.pth \
    --epochs 30 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path data/imagenet \
    --dataset_split dataset_split/10percent.pth \
    --output_dir labeled_only/10percent \
    --log_dir labeled_only/10percent
