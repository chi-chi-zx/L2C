import torch
import torch.nn as nn
# import flax.linen as nn
from einops import rearrange, repeat, reduce

class DomainCache(nn.Module):
    def __init__(self,
                 embed_dim,
                 length=1,
                 pool_size=10) -> None:
        super().__init__()
        key_shape = (pool_size, embed_dim)
        self.prompt = nn.Parameter(torch.randn(pool_size, length, embed_dim))
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        
    def forward(self, support_im, SideNet):
        
        single_domain = self.compute_DomainPrompt(support_im, SideNet)
        single_domain_norm = self.normalize(single_domain)
        key_norm = self.normalize(self.prompt_key)
        score = (key_norm @ single_domain_norm.T).softmax(dim=0)[:, :, None]
        prompt_scale = (self.prompt * score).flatten(0, 1)
        domain_prompt = torch.cat((prompt_scale, single_domain), dim=0)


        return domain_prompt

    def compute_DomainPrompt(self, support_im, SideNet):

        domain_prompt = SideNet(support_im, domain_branch=True)

        return domain_prompt

    def normalize(self, tensor):
        return tensor / tensor.norm(dim=-1, keepdim=True)

