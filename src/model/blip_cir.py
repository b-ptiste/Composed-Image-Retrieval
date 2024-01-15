from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist


class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.text_encoder_only = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.text_only_proj = nn.Linear(embed_dim * 3, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07
        # own model
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )
        self.training_type = "text_embds_outside"

    # own function
    def freeze(self, epoch):
        if epoch == 1:
            print("We don't freeze :")
            for name, param in self.named_parameters():
                if (
                    "mlp" in name or "text_encoder_only" in name
                ) and "visual_encoder" not in name:
                    print(name, end=" ")
                    pass
                else:
                    param.requires_grad = False
        if epoch == 2:
            print("We don't freeze :")
            for name, param in self.named_parameters():
                if "mlp" in name and "visual_encoder" not in name:
                    print(name, end=" ")
                    pass
                else:
                    param.requires_grad = False

        print("\n\nNumber of Trainable parameters")
        t_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(t_params)
        print("Ratio trainable / non trainable")
        n_t_params = sum(p.numel() for p in self.parameters())
        print(t_params / n_t_params)

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch

        device = ref_img.device

        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img)

        # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_embs, text_feat = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )

        if self.training_type == "text_embds_outside":
            encoder_input_ids = text.input_ids.clone()
            text_feat = self.text_encoder_only(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )

            text_feat = text_feat.last_hidden_state[:, 0, :]
            text_feat = F.normalize(self.text_proj(text_feat), dim=-1)
        if self.training_type == "text_embds_inside":
            text_feat = F.normalize(self.text_proj(text_feat.mean(dim=1)), dim=-1)
        # print(text_embs.shape)
        query_feat = query_embs.last_hidden_state[:, 0, :]
        query_feat = F.normalize(self.text_proj(query_feat), dim=-1)
        img_feat_2d = F.normalize(self.vision_proj(ref_img_embs.mean(dim=1)), dim=-1)
        # print(img_feat_2d.shape)

        concatenated_feats = torch.cat(
            (query_feat.unsqueeze(1), img_feat_2d.unsqueeze(1), text_feat.unsqueeze(1)),
            dim=1,
        )
        combined_query_feat = concatenated_feats.view(concatenated_feats.size(0), -1)
        # print(query_feat.shape)
        # Get weights from the MLP
        weights = self.mlp(combined_query_feat)
        query_feat = (
            weights[:, 0].unsqueeze(1) * query_feat
            + weights[:, 1].unsqueeze(1) * img_feat_2d
            + weights[:, 2].unsqueeze(1) * text_feat
        )

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        return self.loss(query_feat, tar_img_feat, self.temp)


def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
