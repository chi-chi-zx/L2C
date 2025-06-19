EVAL_ARGS=(
    paths.data_dir="/data/tao/wilds/data" # change to your own data path
    ckpt_path="/data/chichi/NeurIPS_submission/L2C_clean/model_ckpts/camelyon17_B16.ckpt" 
    model=prompt_tta_ViT_B16_CLIP.yaml 
    data.input_resolution=224          
    model.model.side_layers=1          
    data=camelyon17.yaml
    model.model.pool_length=5
    model.model.pool_size=5
    trainer.devices=[2]                # set GPU number 
    trainer.precision=16               # faster inference with FP16
)

python eval.py "${EVAL_ARGS[@]}"
