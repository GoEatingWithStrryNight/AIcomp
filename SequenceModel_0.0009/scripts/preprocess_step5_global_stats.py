import pandas as pd
from pathlib import Path
import json

PROCESSED_DIR = Path('data/processed')


def main():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_clean.csv')

    # Time span
    min_time = pd.to_datetime(inter['borrow_time']).min()
    max_time = pd.to_datetime(inter['borrow_time']).max()

    user_deg = inter.groupby('user_id')['book_id'].nunique()
    item_deg = inter.groupby('book_id')['user_id'].nunique()

    stats = {
        'n_rows': int(len(inter)),
        'n_users': int(inter['user_id'].nunique()),
        'n_items': int(inter['book_id'].nunique()),
        'time_span_days': None if pd.isna(min_time) or pd.isna(max_time) else int((max_time - min_time).days),
        'user_unique_books_quantiles': user_deg.quantile([0.25,0.5,0.75,0.9,0.95,0.99]).to_dict(),
        'item_unique_users_quantiles': item_deg.quantile([0.25,0.5,0.75,0.9,0.95,0.99]).to_dict(),
    }

    # Top popular items
    top_items = inter['book_id'].value_counts().head(20).to_dict()
    stats['top_items'] = top_items

    with open(PROCESSED_DIR / 'global_stats.json','w',encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('Global stats saved to global_stats.json')

if __name__ == '__main__':
    main()
