EVAL_ARGS=(
    paths.data_dir="/data/tao/wilds/data" # change to your own data path
    ckpt_path="/data/chichi/NeurIPS_submission/L2C_clean/model_ckpts/poverty_B16.ckpt" # change for ViT-L -> iwildcam_L14.ckpt
    model=prompt_tta_ViT_B16_CLIP.yaml # change for ViT-L -> prompt_tta_ViT_L14_px336_CLIP_default.yaml
    data.input_resolution=224          # 336 for ViT-L
    model.model.side_layers=1          
    data=poverty.yaml
    model.model.pool_length=10
    model.model.pool_size=10
    model.model.learnable_scaling=True
    trainer.devices=[3]                # set GPU number 
    trainer.precision=16                 # faster inference with FP16
)

python eval.py "${EVAL_ARGS[@]}"
