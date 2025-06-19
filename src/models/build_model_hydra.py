import torch
import os
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from typing import List, Optional, Tuple
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import src.utils as utils
from src.models.prompt_tta import PromptTTA
import open_clip

from src.lightning.utils import PROJECT_ROOT
from src.clip.clip import load

from src.models.decoder_zoo.decoder_model import DomainFusionModule
from src.models.domain_cache import DomainCache
from src.clip.model import define_SideNet


_M2FEATDIM = {
    "sup_vitb16_224": 768,
    "sup_vitb16": 768,
    "sup_vitl16_224": 1024,
    "sup_vitl16": 1024,
    "sup_vitb8_imagenet21k": 768,
    "sup_vitb16_imagenet21k": 768,
    "sup_vitb32_imagenet21k": 768,
    "sup_vitl16_imagenet21k": 1024,
    "sup_vitl32_imagenet21k": 1024,
    "sup_vith14_imagenet21k": 1280,
    "clip_vit_L14": 1024,
    "clip_vit_b16": 768,
}

_M2TEXTDIM = {
    "clip_vit_b16": 512,
    "clip_vit_L14": 768,
}


def _freeze_components(model: nn.Module, components: List[str] = "CLIP_model"):
    for component_name in components:
        counter = 0
        if component_name == "CLIP_model":
            for _, p in model.CLIP_model.named_parameters():
                p.requires_grad = False
                counter += 1
        if component_name == "SideNet":
            for _, p in model.SideNet.named_parameters():
                p.requires_grad = False
                counter += 1
        print('Freezing {}, {} params'.format(component_name, counter))

    return model


def _build_clip_model(
        model_name: str,
        pretrained_source: str,
        pretrained_weights_dir: str,
        freeze: bool = True,
):
    pretrained_weights_dir = os.path.join(PROJECT_ROOT, pretrained_weights_dir)
    CLIP_model = open_clip.create_model(
        model_name=model_name,
        pretrained=pretrained_source,
        cache_dir=pretrained_weights_dir,
    )
    CLIP_proj = CLIP_model.visual.proj.detach().clone()

    CLIP_model.visual.output_tokens = True

    return CLIP_model, CLIP_proj

def _build_side_net(
        pretrained_weights_dir: str,
        side_base_model_name: str,
        side_layers: int,
        custmized_SideNet,
):  

    print(f'::::::::: status of custmized_SideNet: {custmized_SideNet}')
    if custmized_SideNet == False:
        if '336' in side_base_model_name:
            paths = [PROJECT_ROOT, pretrained_weights_dir, side_base_model_name + 'px.pt']
        else:
            paths = [PROJECT_ROOT, pretrained_weights_dir, side_base_model_name + '.pt']
        pretrained_weights_dir = os.path.join('', *paths)


        side_model, _ = load(pretrain_ckpt_path=pretrained_weights_dir,
                             base_model=side_base_model_name, jit=False, side_visual_layers=side_layers)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = os.path.join('src/clip/model_config', custmized_SideNet)
        side_model = define_SideNet(base_model, side_layers)
        side_model.float()
        side_model.to(device=device)
        print('Custmized side net successful')

    side_model.visual.output_tokens = True

    return side_model.visual

def build_prompt_tta(
        CLIP_model_name: str,
        CLIP_checkpoints_source: str,
        CLIP_checkpoints_dir: str,
        ViT: str = "clip_vit_L14",
        decoder_mlp_dim: int = 2048,
        decoder_num_heads: int = 8,
        decoder_attention_downsample_rate: int = 2,
        num_class: int = 182,
        freeze_components: List[str] = ["CLIP_model"],
        side_layers: int = 3,
        pool_length=1,
        pool_size=10,
        learnable_scaling=False,
        custmized_SideNet=False,

):
    embed_dim = _M2FEATDIM[ViT]
    text_dim =_M2TEXTDIM[ViT]
    model = PromptTTA(
        CLIP_model=_build_clip_model(
            model_name=CLIP_model_name,
            pretrained_source=CLIP_checkpoints_source,
            pretrained_weights_dir=CLIP_checkpoints_dir,
            freeze=True,
        ),
        SideNet=_build_side_net(
            pretrained_weights_dir=CLIP_checkpoints_dir,
            side_base_model_name=CLIP_model_name,
            side_layers=side_layers,
            custmized_SideNet=custmized_SideNet,
        ),
        DomainCache=DomainCache(
            embed_dim=text_dim,
            length=pool_length,
            pool_size=pool_size,
        ),
        Fusion_image=DomainFusionModule(
                embedding_dim=text_dim,
                mlp_dim=decoder_mlp_dim,
                num_heads=decoder_num_heads,
                attention_downsample_rate=decoder_attention_downsample_rate,
        ),
        Fusion_text=DomainFusionModule(
            embedding_dim=text_dim,
            mlp_dim=decoder_mlp_dim,
            num_heads=decoder_num_heads,
            attention_downsample_rate=decoder_attention_downsample_rate,
        ),
        num_class=num_class,
        embed_dim=embed_dim,
        text_dim=text_dim,
        learnable_scaling=learnable_scaling,
    )

    model = _freeze_components(model, components=freeze_components)

    return model



@hydra.main(version_base="1.3", config_path="/data/chichi/NeurIPS_submission/NeurIPS/configs",
            config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    import pytorch_lightning as pl
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(42, workers=True)

    module: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model = module.model

    model.CLIP_model.to(device)
    query_imgs = torch.ones(4, 3, 224, 224).to(device)
    print('check d types of ims', query_imgs.dtype)

    logits = model(query_imgs)

    print(logits.shape)


if __name__ == "__main__":
    main()
