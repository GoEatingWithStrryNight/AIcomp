import pandas as pd
from pathlib import Path
from collections import defaultdict
import math

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
OUTPUT = ANS_DIR / 'submission_itemcf.csv'

# Hyperparameters
MAX_ITEMS_PER_USER_FOR_CF = 300  # limit to control pair explosion
TOPK_SIM_PER_ITEM = 100  # keep only top similar items per item
RECENT_WEIGHT_ALPHA = 0.7  # recency blending weight
RECENT_WINDOW = 180  # days


def load_interactions():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv')
    inter['borrow_time'] = pd.to_datetime(inter['borrow_time'], errors='coerce')
    inter = inter.dropna(subset=['borrow_time'])
    inter = inter.sort_values('borrow_time')
    return inter[['user_id','book_id','borrow_time']]


def build_item_cooccurrence(inter: pd.DataFrame):
    # user -> list of (book, time)
    user_items = inter.groupby('user_id').apply(lambda df: list(zip(df['book_id'], df['borrow_time'])) )
    co_counts = defaultdict(float)
    item_user_degree = defaultdict(int)

    for user, pairs in user_items.items():
        if len(pairs) == 0:
            continue
        # Deduplicate per user per item keep earliest (or latest)
        seen = {}
        for b, t in pairs:
            if b not in seen:
                seen[b] = t
        items_list = list(seen.items())
        # Limit length
        if len(items_list) > MAX_ITEMS_PER_USER_FOR_CF:
            items_list = items_list[-MAX_ITEMS_PER_USER_FOR_CF:]
        # Update item degree
        for b,_ in items_list:
            item_user_degree[b] += 1
        n = len(items_list)
        if n < 2:
            continue
        # Weight factor for large user histories (down-weight common users)
        user_weight = 1 / math.log(2 + n)
        # Generate pairs
        for i in range(n):
            bi, ti = items_list[i]
            for j in range(i+1, n):
                bj, tj = items_list[j]
                if bi == bj:
                    continue
                # Co-occurrence increment symmetrical
                co_counts[(bi,bj)] += user_weight
                co_counts[(bj,bi)] += user_weight

    # Convert to similarity: sim(i,j) = co(i,j) / sqrt(deg(i)*deg(j))
    sims = defaultdict(list)  # item -> list of (other, score)
    for (i,j), c in co_counts.items():
        denom = math.sqrt(item_user_degree[i] * item_user_degree[j]) or 1.0
        score = c / denom
        sims[i].append((j, score))

    # Keep topK per item
    for i in list(sims.keys()):
        sims[i] = sorted(sims[i], key=lambda x: x[1], reverse=True)[:TOPK_SIM_PER_ITEM]
    return sims


def build_recency_weight(inter: pd.DataFrame):
    max_t = inter['borrow_time'].max()
    rec_w = {}
    for row in inter.itertuples(index=False):
        delta_days = (max_t - row.borrow_time).days
        w = math.exp(-delta_days / RECENT_WINDOW)
        # Keep max weight per (user,item)
        key = (row.user_id, row.book_id)
        if key not in rec_w or rec_w[key] < w:
            rec_w[key] = w
    return rec_w


def recommend_for_user(user_id: str, user_hist, sims, rec_w, seen_set):
    scores = defaultdict(float)
    # Aggregate based on each historical item
    for b, t in user_hist:
        if b not in sims:
            continue
        base_weight = rec_w.get((user_id, b), 1.0)
        for nb, s in sims[b]:
            if nb in seen_set:
                continue
            scores[nb] += s * (RECENT_WEIGHT_ALPHA * base_weight + (1-RECENT_WEIGHT_ALPHA))
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def main():
    inter = load_interactions()
    sims = build_item_cooccurrence(inter)
    rec_w = build_recency_weight(inter)

    user_groups = inter.groupby('user_id').apply(lambda df: list(zip(df['book_id'], df['borrow_time'])) )

    recs = []
    for user_id, hist in user_groups.items():
        seen = {b for b,_ in hist}
        rec = recommend_for_user(user_id, hist, sims, rec_w, seen)
        if rec is None:
            # fallback: most recent global book not seen
            fallback = inter[~inter['book_id'].isin(list(seen))]['book_id'].value_counts().index
            if len(fallback) == 0:
                continue
            rec = fallback[0]
        recs.append((user_id, rec))

    out = pd.DataFrame(recs, columns=['user_id','book_id'])
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f'Saved {OUTPUT} shape={out.shape}')

if __name__ == '__main__':
    main()
