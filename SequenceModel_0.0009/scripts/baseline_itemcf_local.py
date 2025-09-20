import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import math

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
OUTPUT = ANS_DIR / 'submission_itemcf_local.csv'

# Hyperparameters
WINDOW_USER_HISTORY = 60   # Only use last N interactions per user to build co-occurrence
RECENT_REF = 5             # Aggregate only from the last R items when recommending
TIME_DECAY_TAU = 120       # days for time decay
TOP_SIM_PER_ITEM = 80
CATEGORY_BONUS = 0.10
MIN_CO_COUNT = 1


def load_data():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv')
    items = pd.read_csv(PROCESSED_DIR / 'items_step1.csv')
    inter['borrow_time'] = pd.to_datetime(inter['borrow_time'], errors='coerce')
    inter = inter.dropna(subset=['borrow_time'])
    inter = inter.sort_values('borrow_time')
    items = items[['book_id','cat_lvl1','cat_lvl2']]
    return inter[['user_id','book_id','borrow_time']], items


def build_co_matrix(inter: pd.DataFrame):
    max_time = inter['borrow_time'].max()
    user_groups = inter.groupby('user_id')
    co = defaultdict(float)
    deg = defaultdict(float)

    for user, df in user_groups:
        # Take last WINDOW_USER_HISTORY
        dfu = df.tail(WINDOW_USER_HISTORY)
        books = list(zip(dfu['book_id'], dfu['borrow_time']))
        if len(books) < 2:
            for b,_ in books:
                deg[b] += 1
            continue
        # Precompute per-item time weight
        time_w = {}
        for b,t in books:
            delta_days = (max_time - t).days
            time_w[b] = math.exp(- delta_days / TIME_DECAY_TAU)
        # user normalization
        user_norm = 1 / math.log(2 + len(books))
        # Build pairs
        n = len(books)
        for i in range(n):
            bi, ti = books[i]
            deg[bi] += 1
            wi = time_w[bi]
            for j in range(i+1, n):
                bj, tj = books[j]
                wj = time_w[bj]
                weight = user_norm * wi * wj
                co[(bi,bj)] += weight
                co[(bj,bi)] += weight
    # Build similarity list
    sims = defaultdict(list)
    for (i,j), val in co.items():
        if val < MIN_CO_COUNT:
            continue
        denom = math.sqrt(deg[i] * deg[j]) or 1.0
        sims[i].append((j, val / denom))
    # Keep top-K
    for i in list(sims.keys()):
        sims[i] = sorted(sims[i], key=lambda x: x[1], reverse=True)[:TOP_SIM_PER_ITEM]
    return sims


def build_user_recent(inter: pd.DataFrame):
    # store full ordered history per user
    return inter.groupby('user_id').apply(lambda df: list(zip(df['book_id'], df['borrow_time'])) )


def recommend_user(user_id, history, sims, items_df, max_time):
    if not history:
        return None
    # Take last RECENT_REF
    recent = history[-RECENT_REF:]
    seen = {b for b,_ in history}
    # Category preference from recent
    cat_map = items_df.set_index('book_id')
    cats = []
    for b,_ in recent:
        if b in cat_map.index:
            row = cat_map.loc[b]
            main_cat = row['cat_lvl1'] if not pd.isna(row['cat_lvl1']) else row['cat_lvl2']
            if pd.notna(main_cat):
                cats.append(main_cat)
    cat_pref = Counter(cats)
    if cat_pref:
        top_cats = {c for c,_ in cat_pref.most_common(2)}
    else:
        top_cats = set()

    scores = defaultdict(float)
    for b,t in recent:
        if b not in sims:
            continue
        # time decay for the anchor item itself
        delta_days = (max_time - t).days
        anchor_w = math.exp(- delta_days / TIME_DECAY_TAU)
        for nb, s in sims[b]:
            if nb in seen:
                continue
            score = s * (0.5 + 0.5 * anchor_w)  # blend anchor recency
            # category bonus
            if nb in cat_map.index:
                row = cat_map.loc[nb]
                main_cat = row['cat_lvl1'] if not pd.isna(row['cat_lvl1']) else row['cat_lvl2']
                if main_cat in top_cats:
                    score *= (1 + CATEGORY_BONUS)
            scores[nb] += score
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def main():
    inter, items = load_data()
    sims = build_co_matrix(inter)
    user_histories = build_user_recent(inter)
    max_time = inter['borrow_time'].max()

    recs = []
    for user_id, hist in user_histories.items():
        book = recommend_user(user_id, hist, sims, items, max_time)
        if book is None:
            # fallback: recent global time-decay popularity
            tmp = inter.tail(20000)  # recent slice
            decay = (max_time - tmp['borrow_time']).dt.days.apply(lambda d: math.exp(-d / TIME_DECAY_TAU))
            pop = tmp.assign(w=decay).groupby('book_id')['w'].sum().sort_values(ascending=False)
            seen = {b for b,_ in hist}
            for cand in pop.index:
                if cand not in seen:
                    book = cand
                    break
            if book is None:
                continue
        recs.append((user_id, book))

    out = pd.DataFrame(recs, columns=['user_id','book_id'])
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f'Saved {OUTPUT} shape={out.shape}')

if __name__ == '__main__':
    main()
