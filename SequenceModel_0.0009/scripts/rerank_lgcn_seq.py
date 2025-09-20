import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
MODEL_DIR = Path('models')

TOPK = 50
W_LGCN = 0.7
W_SEQ = 0.3
OUT_PATH = ANS_DIR / 'submission_seq_rerank.csv'


def load_lightgcn_embeddings():
    ckpt = torch.load(MODEL_DIR / 'lightgcn.pt', map_location='cpu')
    user_emb = ckpt['user_emb']  # (U,E)
    item_emb = ckpt['item_emb']  # (I,E)
    user2idx = ckpt['user2idx']
    item2idx = ckpt['item2idx']
    idx2item = {v:k for k,v in item2idx.items()}
    return user_emb, item_emb, user2idx, item2idx, idx2item


def load_seq_model():
    ckpt = torch.load(MODEL_DIR / 'seqrec_gru.pt', map_location='cpu')
    from seqrec_gru import GRU4Rec, MAX_LEN
    model = GRU4Rec(n_items=ckpt['n_items'], emb_dim=ckpt['emb_dim'], hidden_dim=ckpt['hidden_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def build_user_histories(train_only=True):
    df = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv', usecols=['user_id','book_id','borrow_time'])
    df['user_id'] = df['user_id'].astype(str)
    df['book_id'] = df['book_id'].astype(str)
    df['borrow_time'] = pd.to_datetime(df['borrow_time'], errors='coerce')
    df = df.dropna(subset=['borrow_time']).sort_values(['user_id','borrow_time'])
    # LOO: remove last per user for train histories
    if train_only:
        last_idx = df.groupby('user_id').tail(1).index
        df = df.drop(last_idx)
    histories = df.groupby('user_id')['book_id'].apply(list).to_dict()
    return histories


def topk_from_lightgcn(user_emb, item_emb, user2idx, idx2item, user_histories: Dict[str, List[str]]):
    user_recs = {}
    item_all = item_emb
    for u, seq in user_histories.items():
        if u not in user2idx:
            continue
        uid = user2idx[u]
        ue = user_emb[uid]  # (E)
        scores = torch.matmul(item_all, ue)  # (I)
        # mask seen in train histories
        if len(seq) > 0:
            # map to item indices
            # some items might be absent in lightgcn mapping; skip them
            # we can only mask those that exist
            # build an index list
            # idx2item: idx->book_id, so we need reverse
            pass
        top_scores, top_idx = torch.topk(scores, k=min(TOPK, scores.size(0)))
        items = [idx2item[i.item()] for i in top_idx]
        user_recs[u] = items
    return user_recs


def rerank_with_seq(model, seq_ckpt, candidates: Dict[str, List[str]], user_histories: Dict[str,List[str]], lgcn_scores_raw: Dict[str, Dict[str, float]]):
    from seqrec_gru import MAX_LEN
    # prepare mappings
    item2idx_seq = seq_ckpt['item2idx']
    idx2item_seq = seq_ckpt['idx2item']
    results = []
    for u, cand_list in candidates.items():
        hist = user_histories.get(u, [])
        if len(hist) == 0:
            # fallback: take first candidate if exists
            if len(cand_list) > 0:
                results.append((u, cand_list[0]))
            continue
        # map history to seq item idx
        hist_idx = [item2idx_seq.get(b, 0) for b in hist if b in item2idx_seq]
        # build input
        h = hist_idx[-MAX_LEN:]
        pad_len = MAX_LEN - len(h)
        inp = torch.tensor(([0]*pad_len + h), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            z = model(inp)  # (1,E)
        # score candidates using seq item embedding table
        # model.item_emb exists in GRU4Rec
        cand_idx = [item2idx_seq.get(b, 0) for b in cand_list]
        if len(cand_idx) == 0:
            continue
        item_ids = torch.tensor([i for i in cand_idx if i>0], dtype=torch.long)
        if item_ids.numel() == 0:
            continue
        item_e = model.item_emb(item_ids)
        seq_scores = torch.matmul(item_e, z.squeeze(0))  # (M)
        # normalize seq scores
        s = seq_scores
        s = (s - s.mean()) / (s.std() + 1e-6)

        # gather lgcn scores for the same candidate subset
        lg_map = lgcn_scores_raw.get(u, {})
        lg_scores = torch.tensor([lg_map.get(b, 0.0) for b in cand_list if item2idx_seq.get(b,0)>0], dtype=torch.float32)
        if lg_scores.numel() != s.numel():
            # size mismatch due to unknown items in seq space; align lengths
            m = min(lg_scores.numel(), s.numel())
            lg_scores = lg_scores[:m]
            s = s[:m]
            item_ids = item_ids[:m]
        if lg_scores.numel() == 0:
            best_local = torch.argmax(s).item()
        else:
            lg_norm = (lg_scores - lg_scores.mean()) / (lg_scores.std() + 1e-6)
            fused = W_LGCN * lg_norm + W_SEQ * s
            best_local = torch.argmax(fused).item()
        # map back to book_id
        real_idx = item_ids[best_local].item()
        pred_book = idx2item_seq[real_idx]
        results.append((u, pred_book))
    return results


def main():
    user_emb, item_emb, user2idx, item2idx, idx2item = load_lightgcn_embeddings()
    model, seq_ckpt = load_seq_model()
    user_hist = build_user_histories(train_only=True)

    # Build LightGCN Top-K candidates per user
    # Need to mask train-seen items using LightGCN index space
    # to implement masking we will create a mapping book_id->lightgcn item idx
    book2lg_idx = item2idx
    # Precompute a tensor mask list for each user is heavy; do per-user masking
    candidates = {}
    lgcn_scores_raw = {}
    item_all = item_emb
    for u, seq in user_hist.items():
        if u not in user2idx:
            continue
        uid = user2idx[u]
        ue = user_emb[uid]  # (E)
        scores = torch.matmul(item_all, ue)  # (I)
        # mask train-seen that exist in lightgcn mapping
        for b in seq:
            if b in book2lg_idx:
                scores[book2lg_idx[b]] = -1e9
        top_scores, top_idx = torch.topk(scores, k=min(TOPK, scores.size(0)))
        items = [idx2item[i.item()] for i in top_idx]
        candidates[u] = items
        # store raw scores for fusion (map back to book_id)
        lgcn_scores_raw[u] = {idx2item[i.item()]: top_scores[k].item() for k,i in enumerate(top_idx)}

    # Rerank with GRU sequence model
    reranked = rerank_with_seq(model, seq_ckpt, candidates, user_hist, lgcn_scores_raw)
    sub = pd.DataFrame(reranked, columns=['user_id','book_id'])
    sub.to_csv(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
