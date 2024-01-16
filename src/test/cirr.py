import datetime
import time
from collections import OrderedDict
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.tools.files import json_dump

class TestCirr:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []
        pair_ids = []
        for ref_img, tar_feat, caption, pair_id, *_ in data_loader:
            pair_ids.extend(pair_id.cpu().numpy().tolist())

            device = ref_img.device

            # Define the resize transform
            resize_transform = transforms.Resize((364, 364))

            ref_img_resize = torch.stack([resize_transform(image) for image in ref_img])
            
            image_embeds = model.ln_vision(model.visual_encoder(ref_img_resize))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            text_tokens = model.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(device)

            # Image Text Matching
            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    ref_img.device
                )
                
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            output_itm = model.Qformer.bert(
                    text_tokens.input_ids,
                    query_embeds=query_tokens,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            query_feat = model.text_proj(output_itm.last_hidden_state[:, : query_tokens.size(1), :])
            query_feat = torch.mean(query_feat, dim=1)

            query_feats.append(query_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            pair_ids = fabric.all_gather(pair_ids)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
            pair_ids = einops.rearrange(pair_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            pair_ids = pair_ids.cpu().numpy().tolist()

            assert len(query_feats) == len(pair_ids)
            img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
            assert len(img_ids) == len(pair_ids)

            id2emb = OrderedDict()
            for img_id, tar_img_feat in zip(img_ids, tar_img_feats):
                if img_id not in id2emb:
                    id2emb[img_id] = tar_img_feat

            tar_feats = torch.stack(list(id2emb.values()), dim=0)
            sims_q2t = query_feats @ tar_feats.T

            # Create a mapping from pair_id to row index for faster lookup
            pairid2index = {pair_id: i for i, pair_id in enumerate(pair_ids)}

            # Create a mapping from target_id to column index for faster lookup
            tarid2index = {tar_id: j for j, tar_id in enumerate(id2emb.keys())}

            # Update the similarity matrix based on the condition
            for pair_id, query_feat in zip(pair_ids, query_feats):
                que_id = data_loader.dataset.pairid2ref[pair_id]
                if que_id in tarid2index:
                    sims_q2t[pairid2index[pair_id], tarid2index[que_id]] = -100
            sims_q2t = sims_q2t.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = {}
            recalls["version"] = "rc2"
            recalls["metric"] = "recall"

            recalls_subset = {}
            recalls_subset["version"] = "rc2"
            recalls_subset["metric"] = "recall_subset"

            target_imgs = np.array(list(id2emb.keys()))

            assert len(sims_q2t) == len(pair_ids)
            for pair_id, query_sims in zip(pair_ids, sims_q2t):
                sorted_indices = np.argsort(query_sims)[::-1]

                query_id_recalls = list(target_imgs[sorted_indices][:50])
                query_id_recalls = [
                    str(data_loader.dataset.int2id[x]) for x in query_id_recalls
                ]
                recalls[str(pair_id)] = query_id_recalls

                members = data_loader.dataset.pairid2members[pair_id]
                query_id_recalls_subset = [
                    target
                    for target in target_imgs[sorted_indices]
                    if target in members
                ]
                query_id_recalls_subset = [
                    data_loader.dataset.int2id[x] for x in query_id_recalls_subset
                ][:3]
                recalls_subset[str(pair_id)] = query_id_recalls_subset

            json_dump(recalls, "recalls_cirr.json")
            json_dump(recalls_subset, "recalls_cirr_subset.json")

            print(f"Recalls saved in {Path.cwd()} as recalls_cirr.json")

        fabric.barrier()
