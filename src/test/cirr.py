import datetime
import time
from collections import OrderedDict
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump


class TestCirr:
    def __init__(self):
        # text_embds_inside
        # text_embds_outside
        self.training_type = "text_embds_outside"
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []

        # embd_feat
        query_feats_before = []
        img_feat_2ds = []
        text_feats = []

        weights_feats = []
        pair_ids = []
        for ref_img, tar_feat, caption, pair_id, *_ in data_loader:
            pair_ids.extend(pair_id.cpu().numpy().tolist())

            device = ref_img.device

            ref_img_embs = model.visual_encoder(ref_img)
            ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            text = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
            query_embs, text_feat = model.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
            )

            if self.training_type == "text_embds_outside":
                encoder_input_ids = text.input_ids.clone()
                text_feat = model.text_encoder_only(
                    encoder_input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="text",
                )

                text_feat = text_feat.last_hidden_state[:, 0, :]
                text_feat = F.normalize(model.text_proj(text_feat), dim=-1)
            if self.training_type == "text_embds_inside":
                text_feat = F.normalize(model.text_proj(text_feat.mean(dim=1)), dim=-1)
            # print(text_embs.shape)

            query_feat = query_embs.last_hidden_state[:, 0, :]
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1)

            img_feat_2d = F.normalize(
                model.vision_proj(ref_img_embs.mean(dim=1)), dim=-1
            )
            # print(img_feat_2d.shape)
            # avg_feat = (query_feat + img_feat + text_feat) / 3
            concatenated_feats = torch.cat(
                (
                    query_feat.unsqueeze(1),
                    img_feat_2d.unsqueeze(1),
                    text_feat.unsqueeze(1),
                ),
                dim=1,
            )

            combined_query_feat = concatenated_feats.view(
                concatenated_feats.size(0), -1
            )
            # print(query_feat.shape)
            # Get weights from the MLP

            weights = model.mlp(combined_query_feat)
            query_feat_multi = (
                weights[:, 0].unsqueeze(1) * query_feat
                + weights[:, 1].unsqueeze(1) * img_feat_2d
                + weights[:, 2].unsqueeze(1) * text_feat
            )
            weights_feats.append(weights.detach().cpu())
            query_feats.append(query_feat_multi.cpu())
            query_feats_before.append(query_feat.cpu())
            img_feat_2ds.append(img_feat_2d.cpu())
            text_feats.append(text_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        query_feats = torch.cat(query_feats, dim=0)
        weights_feats = torch.cat(weights_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)
        query_feats_before = torch.cat(query_feats_before, dim=0)
        img_feat_2ds = torch.cat(img_feat_2ds, dim=0)
        text_feats = torch.cat(text_feats, dim=0)

        np.save(".Composed-Image-Retrieval/outputs/weight.npy", weights_feats.numpy())

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            pair_ids = fabric.all_gather(pair_ids)
            query_feats_before = fabric.all_gather(query_feats_before)
            img_feat_2ds = fabric.all_gather(img_feat_2ds)
            text_feats = fabric.all_gather(text_feats)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
            query_feats_before = einops.rearrange(
                query_feats_before, "d b e -> (d b) e"
            )
            img_feat_2ds = einops.rearrange(img_feat_2ds, "d b e -> (d b) e")
            text_feats = einops.rearrange(text_feats, "d b e -> (d b) e")
            pair_ids = einops.rearrange(pair_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            pair_ids_np = pair_ids.cpu().numpy()
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
            sims_query_feats_before = query_feats_before @ tar_feats.T
            sims_img_feat_2ds = img_feat_2ds @ tar_feats.T
            sims_text_feats = text_feats @ tar_feats.T
            np.save(".Composed-Image-Retrieval/outputs/pair_ids_np.npy", pair_ids_np)
            np.save(
                ".Composed-Image-Retrieval/outputs/tar_feats.npy", tar_feats.numpy()
            )
            np.save(".Composed-Image-Retrieval/outputs/sims_q2t.npy", sims_q2t.numpy())
            np.save(
                ".Composed-Image-Retrieval/outputs/sims_query_feats_before.npy",
                sims_query_feats_before.numpy(),
            )
            np.save(
                ".Composed-Image-Retrieval/outputs/sims_img_feat_2ds.npy",
                sims_img_feat_2ds.numpy(),
            )
            np.save(
                ".Composed-Image-Retrieval/outputs/sims_text_feats.npy",
                sims_text_feats.numpy(),
            )
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

            # print(eval_recall(sims_q2t))

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


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        match_index = np.where(inds == index)[0]

        if match_index.size > 0:
            ranks[index] = match_index[0]
        else:
            ranks[index] = len(score)  # Or some other default value indicating no match

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result
