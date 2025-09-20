import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple
import random

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
ANS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
EMB_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 25
BATCH_SIZE = 512
MAX_LEN = 50
NEG_PER_POS = 2
L2_REG = 1e-6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_interactions():
    df = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv', usecols=['user_id','book_id','borrow_time'])
    df['user_id'] = df['user_id'].astype(str)
    df['book_id'] = df['book_id'].astype(str)
    df['borrow_time'] = pd.to_datetime(df['borrow_time'], errors='coerce')
    df = df.dropna(subset=['borrow_time'])
    df = df.sort_values(['user_id','borrow_time'])
    return df


def leave_one_out_split(inter: pd.DataFrame):
    # returns train_df, test_df (last interaction per user)
    inter = inter.sort_values(['user_id','borrow_time'])
    last_idx = inter.groupby('user_id').tail(1).index
    test = inter.loc[last_idx]
    train = inter.drop(last_idx)
    return train, test


def build_mappings(train: pd.DataFrame, all_items: List[str]):
    users = sorted(train['user_id'].unique())
    # include all items across the dataset to allow scoring for test-only items (untrained embeds remain near-init)
    items = sorted(all_items)
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {b:i+1 for i,b in enumerate(items)}  # reserve 0 for PAD
    idx2item = {i+1:b for i,b in enumerate(items)}
    return users, items, user2idx, item2idx, idx2item


def build_user_sequences(train: pd.DataFrame, user2idx: Dict[str,int], item2idx: Dict[str,int]) -> List[List[int]]:
    seqs = [[] for _ in range(len(user2idx))]
    for uid, g in train.groupby('user_id'):
        if uid not in user2idx:
            continue
        uidx = user2idx[uid]
        items = [item2idx.get(b, 0) for b in g['book_id'].tolist() if b in item2idx]
        seqs[uidx] = items
    return seqs


class SeqDataset(Dataset):
    def __init__(self, user_seqs: List[List[int]], n_items: int, max_len: int, neg_per_pos: int):
        self.samples = []
        self.n_items = n_items
        self.max_len = max_len
        self.neg_per_pos = neg_per_pos
        for u, seq in enumerate(user_seqs):
            # need at least 2 interactions to form (context->target)
            if len(seq) < 2:
                continue
            # we train on positions 1..len(seq)-1
            for t in range(1, len(seq)):
                tgt = seq[t]
                hist = seq[max(0, t - max_len):t]
                self.samples.append((u, hist, tgt))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        u, hist, tgt = self.samples[idx]
        # negative sampling: avoid items in user's full sequence
        user_all = set()
        # recover user's full sequence from samples would be expensive; approximate with current hist
        user_all.update(hist)
        negs = []
        while len(negs) < self.neg_per_pos:
            j = random.randint(1, self.n_items)  # 1..n_items
            if j not in user_all and j != tgt:
                negs.append(j)
        return u, torch.tensor(hist, dtype=torch.long), torch.tensor(tgt, dtype=torch.long), torch.tensor(negs, dtype=torch.long)


def collate_fn(batch):
    # pad histories to MAX_LEN with 0 (PAD)
    us, hists, tgts, negs = zip(*batch)
    lengths = [len(h) for h in hists]
    maxL = max(lengths) if len(lengths)>0 else 1
    maxL = min(maxL, MAX_LEN)
    padded = []
    for h in hists:
        # ensure python list
        if isinstance(h, torch.Tensor):
            h = h.tolist()
        h = h[-MAX_LEN:]
        pad_len = MAX_LEN - len(h)
        if pad_len > 0:
            padded.append([0]*pad_len + h)
        else:
            padded.append(h)
    return (
        torch.tensor(us, dtype=torch.long),
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(tgts, dtype=torch.long),
        torch.stack(negs, dim=0)
    )


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)  # +1 for PAD
        self.pos_emb = nn.Embedding(MAX_LEN, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.hidden2emb = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
    def forward(self, seq_batch):
        # seq_batch: (B, L)
        B, L = seq_batch.size()
        item_e = self.item_emb(seq_batch)  # (B,L,E)
        positions = torch.arange(L, dtype=torch.long, device=seq_batch.device).unsqueeze(0).expand(B, L)
        pos_e = self.pos_emb(positions)
        x = item_e + pos_e
        x = self.dropout(x)
        out, h_n = self.gru(x)  # out: (B,L,H), h_n: (1,B,H)
        h = h_n[-1]  # (B,H)
        z = self.hidden2emb(h)  # (B,E)
        return z  # sequence representation in item-embedding space


def bpr_loss(user_z, pos_items, neg_items, item_emb):
    # user_z: (B,E); pos_items: (B,); neg_items: (B, K)
    pos_e = item_emb(pos_items)  # (B,E)
    neg_e = item_emb(neg_items)  # (B,K,E)
    pos_scores = (user_z * pos_e).sum(dim=-1, keepdim=True)  # (B,1)
    # expand z to (B,K,E)
    z_exp = user_z.unsqueeze(1).expand_as(neg_e)
    neg_scores = (z_exp * neg_e).sum(dim=-1)  # (B,K)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    reg = (user_z.pow(2).mean() + pos_e.pow(2).mean() + neg_e.pow(2).mean())
    return loss + L2_REG * reg


def train_and_infer():
    inter = load_interactions()
    train_df, test_df = leave_one_out_split(inter)

    all_items = inter['book_id'].unique().tolist()
    users, items, user2idx, item2idx, idx2item = build_mappings(train_df, all_items)

    # build sequences from train only (to avoid leaking the last test item)
    user_seqs = build_user_sequences(train_df, user2idx, item2idx)

    # dataset & loader
    ds = SeqDataset(user_seqs, n_items=len(items), max_len=MAX_LEN, neg_per_pos=NEG_PER_POS)
    if len(ds) == 0:
        raise RuntimeError('No training samples for GRU4Rec. Check data preprocessing.')
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)

    model = GRU4Rec(n_items=len(items), emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS+1):
        total = 0.0
        for us, seqs, tgts, negs in loader:
            seqs = seqs.to(DEVICE)
            tgts = tgts.to(DEVICE)
            negs = negs.to(DEVICE)
            optimizer.zero_grad()
            z = model(seqs)
            loss = bpr_loss(z, tgts, negs, model.item_emb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch}/{EPOCHS} loss={total/len(loader):.4f}')

    # inference: for users seen in train, score all items and mask TRAIN-seen
    model.eval()
    subs = []
    with torch.no_grad():
        for uidx, seq in enumerate(user_seqs):
            if len(seq) == 0:
                continue
            # prepare input sequence (pad left)
            h = seq[-MAX_LEN:]
            pad_len = MAX_LEN - len(h)
            inp = torch.tensor(([0]*pad_len + h), dtype=torch.long, device=DEVICE).unsqueeze(0)
            z = model(inp)  # (1,E)
            # scores against all items 1..n_items
            item_ids = torch.arange(1, len(items)+1, dtype=torch.long, device=DEVICE)
            item_e = model.item_emb(item_ids)  # (N,E)
            scores = torch.matmul(item_e, z.squeeze(0))  # (N)
            # mask train-seen items
            if len(seq) > 0:
                seen = torch.tensor(sorted(set(seq)), dtype=torch.long, device=DEVICE)
                scores[seen-1] = -1e9  # shift because item ids start at 1
            topk = torch.topk(scores, k=1).indices.tolist()
            pred_item_idx = topk[0] + 1  # back to 1-based
            subs.append((users[uidx], idx2item[pred_item_idx]))

    sub = pd.DataFrame(subs, columns=['user_id','book_id'])
    out_path = ANS_DIR / 'submission_seq.csv'
    sub.to_csv(out_path, index=False)
    print(f'Saved {out_path}')

    # save checkpoint for reranking reuse
    ckpt = {
        'state_dict': model.state_dict(),
        'n_items': len(items),
        'emb_dim': EMB_DIM,
        'hidden_dim': HIDDEN_DIM,
        'max_len': MAX_LEN,
        'users': users,
        'items': items,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2item': idx2item,
    }
    torch.save(ckpt, MODEL_DIR / 'seqrec_gru.pt')
    print('Saved models/seqrec_gru.pt')


if __name__ == '__main__':
    train_and_infer()
