import copy
import math
import numpy as np
import torch
import torch.nn as nn

from typing import Any, Optional, Tuple, Type, List
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from src.clip import clip as clip
from typing import Any, Union
from src.clip.model import LayerNorm
import time

def _sanity_check_domain_ids(domain_ids):
    """
    During training, the first several images must come from the same domain, 
    while the rest of images from the other random domains
    """
    positive_domian_id = domain_ids[0]
    
    # Find the point where the first element changes
    first_change_index = next((i for i, x in enumerate(domain_ids) if x != positive_domian_id), len(domain_ids))
    if first_change_index < 1:
        raise ValueError(f"In a support set {domain_ids}, there should be at least one data \
                         examples as positives from a same domain")

    # Check the rest of the list for distinct elements
    rest_of_list = domain_ids[first_change_index:]
    if len(set(rest_of_list)) != len(rest_of_list):
        raise ValueError("In a support set, the rest of data examples as the negatives \
                         must be sampled from other distinct domains")

class PromptTTA(nn.Module):
    def __init__(self, 
                 CLIP_model,
                 SideNet,
                 DomainCache,
                 Fusion_image,
                 Fusion_text,
                 embed_dim,
                 text_dim,
                 num_class,
                 learnable_scaling=False,
                 # RevertedCrossAtten,
                 # device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                 ) -> None:
        super().__init__()

        # CLIP returns cls token without projection + image imbedding, [:, 0, :] and [:. 1:, :]
        # Side returns projected cls + and whole embedding

        self.CLIP_model = CLIP_model[0]
        self.CLIP_proj = CLIP_model[1]
        self.Fusion_image = Fusion_image
        self.Fusion_text = Fusion_text
        self.cache = DomainCache
        self.CLIP_model.visual.proj = None
        
        self.SideNet = SideNet
        self.num_class = num_class
        # self.device = device
        self.text_refine = nn.Parameter(torch.zeros(self.num_class, self.num_class))
        self.text_refine_post = nn.Parameter(torch.zeros(text_dim, text_dim))


        if learnable_scaling:
            self.scaling = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.learnable_scaling = True
        else:
            self.learnable_scaling = False


        scale = embed_dim ** -0.5

    def forward(
        self,
        query_images,
        ):

        '''
        query_images: input
        '''

        batch_size, c, h, w = query_images.shape
        prompt = repeat(self.prompt, 'l c -> b l c', b=batch_size)

        ''' CLIP Image Encoder '''
        CLIP_cls, CLIP_feature, CLIP_feature_proj = self.CLIP_features(query_images)
        if self.learnable_scaling:
            log_scale = self.scaling.exp()
        else:
            log_scale = self.CLIP_model.logit_scale.exp()

        ''' SideNet feature '''
        Side_cls, Side_feature = self.Side_feature(query_images)

        ''' Revert '''
        revert_cls_norm, Side_feature_proj = self.revert_crossattn(CLIP_feature=CLIP_feature, Side_feature=Side_feature)

        cls_token = CLIP_cls + revert_cls_norm

        im_feature_proj = CLIP_feature_proj + Side_feature_proj

        ''' Text refinement '''
        text_refined_1 = torch.matmul(self.text_refine, self.text_embed) + self.text_embed
        text_refined = torch.matmul(text_refined_1, self.text_refine_post) + text_refined_1

        middle_txt = self.normalize(text_refined)

        text_batch = repeat(text_refined, 'l c -> b l c', b=batch_size)

        ''' Fusion '''
        fusion_im, fusion_tx = self.compute_fusion(prompt, im_feature_proj, self.CLIP_model.visual.positional_embedding,text_batch, None)


        protopypes = reduce(fusion_tx, "b l c -> l c", "mean")

        protopypes_norm = self.normalize(protopypes)

        cls_token = fusion_im[:, 0, :]
        cls_token_norm = self.normalize(cls_token)

        logits = log_scale * cls_token_norm @ protopypes_norm.T


        return cls_token_norm, protopypes_norm, log_scale, logits



    def compute_text_embedding(self, input_prompts, device):
        text_tokens = clip.tokenize(input_prompts).to(device=device)
        self.text_embed = self.CLIP_model.encode_text(text_tokens)
        self.text_embed_norm = self.normalize(self.text_embed)

        self.CLIP_proj = self.CLIP_proj.to(device=self.text_embed_norm.device)

        text_loss = torch.pdist(self.text_embed_norm.to(torch.float), p=2).pow(2.0).mul(
            -2.0).exp().mean()

    def text_embedding(self, class_name, templates, device, single_temp=True, temp_ID=74):
        self.CLIP_proj = self.CLIP_proj.to(device=device)

        if single_temp:
            template = templates[temp_ID]
            prompt = [template.format(x) for x in class_name]
            text_tokens = clip.tokenize(prompt).to(device=device)
            self.text_embed = self.CLIP_model.encode_text(text_tokens)
            self.text_embed_norm = self.normalize(self.text_embed)
            text_loss = torch.pdist(self.text_embed_norm.to(torch.float), p=2).pow(2.0).mul(
                -2.0).exp().mean()
        else:
            # greedy emsenble
            embedding_list = []
            loss_list = []
            for template in templates:
                prompt = [template.format(x) for x in class_name]
                text_tokens = clip.tokenize(prompt).to(device=device)
                text_embed = self.CLIP_model.encode_text(text_tokens)
                text_embed_norm = self.normalize(text_embed)
                text_loss = self.uniform_loss(text_embed_norm)
                loss_list.append(text_loss)
                embedding_list.append(text_embed)

            loss_list = torch.stack(loss_list, dim=0)
            embedding_list = torch.stack(embedding_list, dim=0)
            templates = np.array(templates)

            sort_idx = torch.argsort(loss_list, descending=True).cpu().numpy()

            embed_sorted = embedding_list[sort_idx]
            template_sorted = templates[sort_idx]

            for i, temp_ref in enumerate(template_sorted):
                if i == len(template_sorted) - 1:
                    current_list = [embed_sorted[i]]
                    low_score = self.uniform_loss(self.normalize(embed_sorted[i]))
                    count = 0
                    for j, temp_comp in enumerate(template_sorted):
                        if i != j:
                            current_list.append(embed_sorted[j])
                            current_embed = self.normalize(torch.stack(current_list, dim=0))
                            current_embed = current_embed.mean(dim=0)
                            current_embed = self.normalize(current_embed)
                            curr_score = self.uniform_loss(current_embed)
                            if curr_score > low_score:
                                current_list.pop()
                            else:
                                low_score = curr_score
                                count += 1

                    current_stack = torch.stack(current_list, dim=0)
                    self.text_embed = current_stack.mean(dim=0)
                    current_embed = self.normalize(current_stack)
                    current_embed = current_embed.mean(dim=0)
                    self.text_embed_norm = self.normalize(current_embed)
                    text_loss = self.uniform_loss(self.text_embed_norm)




    def CLIP_features(self, query_im):

        cls_token_CLIP, image_embeds_CLIP = self.CLIP_model.encode_image(query_im)
        CLIP_feature = torch.cat([cls_token_CLIP[:, None, :], image_embeds_CLIP], dim=1)

        CLIP_cls = self.normalize(cls_token_CLIP @ self.CLIP_proj)

        CLIP_feature_proj = CLIP_feature @ self.CLIP_proj

        return CLIP_cls, CLIP_feature, CLIP_feature_proj

    def normalize(self, tensor):
        return tensor / tensor.norm(dim=-1, keepdim=True)


    def Side_feature(self, query_im):

        cls_token_Side, image_embeds_Side = self.SideNet(query_im)
        cls_token_Side_norm = self.normalize(cls_token_Side)

        return cls_token_Side_norm, image_embeds_Side


    def revert_crossattn(self, CLIP_feature, Side_feature):


        CLIP_shape = CLIP_feature.shape
        Side_shape = Side_feature.shape

        if CLIP_shape[2] == Side_shape[2]:

            cross_mult = torch.mul(CLIP_feature, Side_feature)
            atten = 1 - cross_mult.softmax(dim=-1)
            Side_attented = Side_feature * atten

            Side_attented_proj = (self.SideNet.ln_post(Side_attented) @ self.SideNet.proj)

        else:
            CLIP_proj = CLIP_feature @ self.CLIP_proj
            Side_proj = (self.SideNet.ln_post(Side_feature) @ self.SideNet.proj)

            cross_mult = torch.mul(CLIP_proj, Side_proj)
            atten = 1 - cross_mult.softmax(dim=-1)
            Side_attented_proj = Side_proj * atten

        revert_cls = Side_attented_proj[:, 0, :]
        revert_cls_norm = self.normalize(revert_cls)

        return revert_cls_norm, Side_attented_proj

    def encode_prompt(self, support_im, domain_ids):

        _sanity_check_domain_ids(domain_ids)
        self.prompt = self.cache(support_im, self.SideNet)
        return self.prompt

    def compute_fusion(self, d_prompt, im, im_pe, aux, aux_pe):
        fusion_im = self.Fusion_image(d_prompt, im, im_pe, aux, aux_pe)
        fusion_tx = self.Fusion_text(d_prompt, aux, aux_pe, im, im_pe)


        text_output = self.Fusion_text.final_fusion(condition=fusion_im)
        im_output = self.Fusion_image.final_fusion(condition=fusion_tx)

        return im_output, text_output

    def uniform_loss(self, text_embed):

        return torch.pdist(text_embed.to(torch.float), p=2).pow(2.0).mul(
                        -2.0).exp().mean()
