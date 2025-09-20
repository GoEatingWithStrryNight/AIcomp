import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Tuple

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
ANS_DIR.mkdir(exist_ok=True)

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
LR = 1e-3
EPOCHS = 25
BATCH_SIZE = 4096
NEG_PER_POS = 2
L2_REG = 1e-5
SEED = 42
TOPK_CAND = 500  # candidate breadth when scoring (all items anyway small enough)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class BPRDataset(Dataset):
    def __init__(self, user_pos: List[List[int]], n_items: int, neg_per_pos: int):
        self.user_pos = user_pos
        self.n_items = n_items
        self.neg_per_pos = neg_per_pos
        self.samples = []
        for u, items in enumerate(user_pos):
            for i in items:
                self.samples.append((u, i))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        u, i = self.samples[idx]
        negs = []
        pos_set = set(self.user_pos[u])
        for _ in range(self.neg_per_pos):
            while True:
                j = random.randint(0, self.n_items - 1)
                if j not in pos_set:
                    negs.append(j)
                    break
        return u, i, torch.tensor(negs, dtype=torch.long)

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, edge_index):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        # edge_index: (2, E) with user nodes [0..n_users-1], item nodes [n_users..n_users+n_items-1]
        self.edge_index = edge_index  # torch.LongTensor
        self.adj = None
        self._build_adj()
    def _build_adj(self):
        n_total = self.n_users + self.n_items
        # build sparse adjacency with symmetric edges
        rows = self.edge_index[0]
        cols = self.edge_index[1]
        indices = torch.stack([rows, cols], dim=0)
        data = torch.ones(indices.size(1))
        adj = torch.sparse_coo_tensor(indices, data, (n_total, n_total))
        # add transpose (already assumed edges both directions passed) ensure symmetric
        adj = adj.coalesce()
        # compute degree
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        d_i = deg_inv_sqrt[rows]
        d_j = deg_inv_sqrt[cols]
        norm_data = d_i * data * d_j
        norm_adj = torch.sparse_coo_tensor(indices, norm_data, (n_total, n_total))
        self.adj = norm_adj.coalesce().to(DEVICE)
    def propagate(self, x):
        out = [x]
        h = x
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.adj, h)
            out.append(h)
        return torch.mean(torch.stack(out, dim=0), dim=0)
    def forward(self):
        user_e = self.user_emb.weight
        item_e = self.item_emb.weight
        x = torch.cat([user_e, item_e], dim=0)
        all_out = self.propagate(x)
        return all_out[:self.n_users], all_out[self.n_users:]
    def bpr_loss(self, users, pos_items, neg_items):
        user_all, item_all = self.forward()
        u_e = user_all[users]
        i_e = item_all[pos_items]
        j_e = item_all[neg_items]  # shape (batch, neg_per_pos)
        # Expand u_e / i_e for broadcasting
        u_e_exp = u_e.unsqueeze(1)
        i_e_exp = i_e.unsqueeze(1)
        pos_scores = (u_e_exp * i_e_exp).sum(-1)  # (batch,1)
        neg_scores = torch.einsum('bnd,bnd->bn', u_e_exp.expand_as(j_e), j_e)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        reg = (u_e.pow(2).sum() + i_e.pow(2).sum() + j_e.pow(2).sum())/users.size(0)
        return loss + L2_REG * reg

def build_edge_index(interactions, user2idx, item2idx):
    user_ids = interactions['user_id'].map(user2idx).values
    item_ids = interactions['book_id'].map(item2idx).values + len(user2idx)
    rows = np.concatenate([user_ids, item_ids])
    cols = np.concatenate([item_ids, user_ids])
    edge_index = torch.LongTensor(np.stack([rows, cols], axis=0))
    return edge_index

def load_interactions_with_time():
    df = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv', usecols=['user_id','book_id','borrow_time'])
    df['user_id'] = df['user_id'].astype(str)
    df['book_id'] = df['book_id'].astype(str)
    df['borrow_time'] = pd.to_datetime(df['borrow_time'], errors='coerce')
    df = df.dropna(subset=['borrow_time'])
    return df.sort_values('borrow_time')

def leave_one_out_split(inter: pd.DataFrame):
    last_idx = inter.groupby('user_id').tail(1).index
    test = inter.loc[last_idx]
    train = inter.drop(last_idx)
    return train, test


def train_lightgcn():
    # Use LOO split: train excludes each user's last interaction
    inter_all = load_interactions_with_time()[['user_id','book_id','borrow_time']]
    train_inter, test_inter = leave_one_out_split(inter_all)

    # Universe defined on train to avoid leaking test-only users/items into training graph
    users = sorted(train_inter['user_id'].unique())
    items = sorted(train_inter['book_id'].unique())
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {b:i for i,b in enumerate(items)}

    # Filter train interactions to users/items in mapping
    train_inter = train_inter[train_inter['user_id'].isin(user2idx.keys()) & train_inter['book_id'].isin(item2idx.keys())]
    edge_index = build_edge_index(train_inter, user2idx, item2idx)

    # build user positive lists
    user_pos_train = [[] for _ in range(len(users))]
    for row in train_inter.itertuples(index=False):
        u = user2idx[row.user_id]
        i = item2idx[row.book_id]
        user_pos_train[u].append(i)

    dataset = BPRDataset(user_pos_train, len(items), NEG_PER_POS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = LightGCN(len(users), len(items), EMB_DIM, N_LAYERS, edge_index).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            users_b, pos_b, negs_b = batch
            users_b = users_b.to(DEVICE)
            pos_b = pos_b.to(DEVICE)
            negs_b = negs_b.to(DEVICE)
            optimizer.zero_grad()
            loss = model.bpr_loss(users_b, pos_b, negs_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}/{EPOCHS} loss={total_loss/len(loader):.4f}')

    # inference
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()
    torch.save({'user_emb':user_emb.cpu(), 'item_emb':item_emb.cpu(), 'user2idx':user2idx, 'item2idx':item2idx}, MODEL_DIR / 'lightgcn.pt')

    # Generate submission: for each user pick highest score unseen (all seen actually all items minus positives) - but dataset requires one book
    user_recs = []
    # For submission, we score for all users seen in train; evaluation会自动只统计在测试中的用户
    for u, pos_list in enumerate(user_pos_train):
        ue = user_emb[u]
        scores = torch.matmul(item_emb, ue)
        # Only mask items seen in TRAIN (不要误把 LOO 的“最后一本”当已看)
        if len(pos_list) > 0:
            scores[pos_list] = -1e9
        top_item = torch.argmax(scores).item()
        user_recs.append((users[u], items[top_item]))

    sub = pd.DataFrame(user_recs, columns=['user_id','book_id'])
    sub.to_csv(ANS_DIR / 'submission_lightgcn.csv', index=False)
    print('Saved submission_lightgcn.csv')

if __name__ == '__main__':
    train_lightgcn()
