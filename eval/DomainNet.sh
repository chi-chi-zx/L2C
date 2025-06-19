EVAL_ARGS=(
    paths.data_dir="/data/tao/wilds/data" # change to your own data path
    ckpt_path="/data/chichi/NeurIPS_submission/L2C_clean/model_ckpts/DomainNet_part_B16.ckpt" # change the ckpt
    model=prompt_tta_ViT_B16_CLIP.yaml # change for ViT-L -> prompt_tta_ViT_L14_px336_CLIP_default.yaml
    data.input_resolution=224          # 336 for ViT-L
    model.model.side_layers=3          
    data=domainnet_part.yaml      # change to part, painting, infograph, quick, real, sketch
    model.model.pool_length=1
    model.model.pool_size=10
    trainer.devices=[0]                # set GPU number 
    trainer.precision=16                 # faster inference with FP16
)

python eval.py "${EVAL_ARGS[@]}"
