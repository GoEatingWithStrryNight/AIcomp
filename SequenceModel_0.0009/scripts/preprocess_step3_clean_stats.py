import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('data/processed')

def main():
    inter_path = PROCESSED_DIR / 'interactions_step2_time.csv'
    interactions = pd.read_csv(inter_path)

    # Remove rows missing essential ids
    before = len(interactions)
    interactions = interactions.dropna(subset=['user_id','book_id'])

    # Renew count fill
    if 'renew_cnt' in interactions.columns:
        interactions['renew_cnt'] = pd.to_numeric(interactions['renew_cnt'], errors='coerce').fillna(0).astype(int)

    # Clip unreasonable borrow duration (e.g., > 365 days -> set NaN)
    if 'borrow_duration_days' in interactions.columns:
        interactions.loc[interactions['borrow_duration_days'] > 400, 'borrow_duration_days'] = pd.NA

    removed = before - len(interactions)

    # Basic per-user / per-item counts
    user_stats = interactions.groupby('user_id').agg(
        user_interactions=('book_id','count'),
        user_unique_books=('book_id','nunique')
    ).reset_index()
    item_stats = interactions.groupby('book_id').agg(
        item_interactions=('user_id','count'),
        item_unique_users=('user_id','nunique')
    ).reset_index()

    interactions.to_csv(PROCESSED_DIR / 'interactions_clean.csv', index=False)
    user_stats.to_csv(PROCESSED_DIR / 'user_stats.csv', index=False)
    item_stats.to_csv(PROCESSED_DIR / 'item_stats.csv', index=False)

    print(f'Cleaned interactions saved. Removed {removed} rows. Users:{len(user_stats)} Items:{len(item_stats)}')

if __name__ == '__main__':
    main()
