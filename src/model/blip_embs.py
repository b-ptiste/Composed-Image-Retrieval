import torch
from torch import nn

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertConfig, BertLMHeadModel
from src.model.vit import VisionTransformer, interpolate_pos_embed
from lavis.models import load_model_and_preprocess

class BLIPEmbs(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        vit_model="eva_clip_g",
        use_grad_checkpoint=False,
        drop_path_rate=0,
        vit_precision="fp16",
        image_size=384,
        freeze_vit=True,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        negative_all_rank=False,
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

        self.queue_size = queue_size
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.negative_all_rank = negative_all_rank
    
    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            # visual_encoder = create_eva_vit_g(
            #     img_size, drop_path_rate, use_grad_checkpoint, precision
            # )
            visual_encoder, vision_width = create_vit(
                "base", img_size, False, 0
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.is_decoder = True
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

def create_vit(
    vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0
):
    assert vit in ["base", "large"], "vit parameter must be base or large"
    if vit == "base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate,
        )
    elif vit == "large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate,
        )
    else:
        raise NotImplementedError
    return visual_encoder, vision_width




def blip_embs(pretrained="", **kwargs):
    model = BLIPEmbs(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
        # assert len(msg.missing_keys) == 0, "Missing keys!"
    return model
