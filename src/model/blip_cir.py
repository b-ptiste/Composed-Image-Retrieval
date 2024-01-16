from typing import Any

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertLMHeadModel
from src.model.vit import VisionTransformer, interpolate_pos_embed
from lavis.models import load_model_and_preprocess

class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",
        drop_path_rate=0,
        med_config="configs/med_config.json",
        vit_precision="fp16",
        use_grad_checkpoint=False,
        freeze_vit=True,
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
        num_query_token=32,
        cross_attention_freq=2,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        blip2_model, _, _ = load_model_and_preprocess(
            name="blip2_image_text_matching", model_type="coco", is_eval=False, device='cuda'
        )

        self.visual_encoder = blip2_model.visual_encoder
        self.ln_vision = blip2_model.ln_vision

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            # logging.info("freeze vision encoder")

        self.Qformer = blip2_model.Qformer
        
        self.query_tokens = blip2_model.query_tokens

        self.tokenizer = init_tokenizer()

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        
        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch
        device = ref_img.device

        # Define the resize transform
        resize_transform = transforms.Resize((364, 364))

        # Apply the transform to each image in the batch
        ref_img_resize = torch.stack([resize_transform(image) for image in ref_img])

        image_embeds = self.ln_vision(self.visual_encoder(ref_img_resize))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        text_tokens = self.tokenizer(
              caption,
              padding="max_length",
              truncation=True,
              max_length=35,
              return_tensors="pt",
          ).to(device)


        # Image Text Matching
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                ref_img.device
            )
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output_itm = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        query_feat = self.text_proj(output_itm.last_hidden_state[:, : query_tokens.size(1), :])

        query_feat = torch.mean(query_feat, dim=1)


        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            tar_feat = fabric.all_gather(tar_feat, sync_grads=True)

            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")
            tar_feat = einops.rearrange(tar_feat, "d b e -> (d b) e")

        return self.loss(query_feat, tar_feat, self.temp)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
