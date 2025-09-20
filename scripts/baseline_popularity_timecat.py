import pandas as pd
from pathlib import Path
import math
from collections import defaultdict

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')
OUTPUT = ANS_DIR / 'submission_popularity_timecat.csv'

# Hyperparameters
TAU_DAYS = 180  # time decay scale
CAT_ALPHA = 0.6  # weight strength for category preference


def load_data():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv')
    items = pd.read_csv(PROCESSED_DIR / 'items_step1.csv')
    inter['borrow_time'] = pd.to_datetime(inter['borrow_time'], errors='coerce')
    inter = inter.dropna(subset=['borrow_time'])
    # Keep cols
    items = items[['book_id','cat_lvl1','cat_lvl2']]
    return inter[['user_id','book_id','borrow_time']], items


def compute_time_decay_pop(inter: pd.DataFrame):
    max_t = inter['borrow_time'].max()
    # weight = exp(-delta_days / TAU)
    delta = (max_t - inter['borrow_time']).dt.days.clip(lower=0)
    weights = (-(delta / TAU_DAYS)).apply(math.exp)
    pop = inter.assign(weight=weights).groupby('book_id')['weight'].sum().sort_values(ascending=False)
    return pop


def build_user_category_pref(inter: pd.DataFrame, items: pd.DataFrame):
    merged = inter.merge(items, on='book_id', how='left')
    merged['main_cat'] = merged['cat_lvl1'].fillna(merged['cat_lvl2'])
    grp = merged.groupby(['user_id','main_cat'])['book_id'].count().reset_index(name='cnt')
    # Normalize per user
    grp['user_total'] = grp.groupby('user_id')['cnt'].transform('sum')
    grp['pref'] = grp['cnt'] / grp['user_total']
    user_cat_pref = defaultdict(dict)
    for row in grp.itertuples(index=False):
        user_cat_pref[str(row.user_id)][row.main_cat] = row.pref
    return user_cat_pref


def score_candidates(pop: pd.Series, cat_pref: dict, items: pd.DataFrame, user_id: str, seen: set, topK_pool=4000):
    # Build candidate pool from topK popularity
    candidates = []
    subset = pop.iloc[:topK_pool]
    for book_id, base_score in subset.items():
        if book_id in seen:
            continue
        candidates.append((book_id, base_score))
    if not candidates:
        # fallback first of pop
        return pop.index[0]
    df = pd.DataFrame(candidates, columns=['book_id','base'])
    df = df.merge(items, on='book_id', how='left')
    df['main_cat'] = df['cat_lvl1'].fillna(df['cat_lvl2'])
    user_pref = cat_pref.get(str(user_id), {})
    df['score'] = df.apply(lambda r: r['base'] * (1 + CAT_ALPHA * user_pref.get(r['main_cat'], 0.0)), axis=1)
    df = df.sort_values('score', ascending=False)
    return df.iloc[0]['book_id']


def main():
    inter, items = load_data()
    pop = compute_time_decay_pop(inter)
    cat_pref = build_user_category_pref(inter, items)

    user_groups = inter.groupby('user_id')['book_id'].apply(set)

    recs = []
    for user_id, seen in user_groups.items():
        book = score_candidates(pop, cat_pref, items, user_id, seen)
        recs.append((user_id, book))

    out = pd.DataFrame(recs, columns=['user_id','book_id'])
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f'Saved {OUTPUT} shape={out.shape}')

if __name__ == '__main__':
    main()
