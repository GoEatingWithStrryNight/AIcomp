import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('data/processed')

def main():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_clean.csv')

    users = sorted(inter['user_id'].unique())
    items = sorted(inter['book_id'].unique())

    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {b:i for i,b in enumerate(items)}

    inter['user_idx'] = inter['user_id'].map(user2idx)
    inter['item_idx'] = inter['book_id'].map(item2idx)

    # Keep essential slim columns
    slim = inter[['user_idx','item_idx']].copy()
    slim.to_parquet(PROCESSED_DIR / 'interactions_slim.parquet', index=False)

    pd.DataFrame({'user_id':users,'user_idx':range(len(users))}).to_csv(PROCESSED_DIR / 'user_mapping.csv', index=False)
    pd.DataFrame({'book_id':items,'item_idx':range(len(items))}).to_csv(PROCESSED_DIR / 'item_mapping.csv', index=False)

    print(f'Mappings saved. Users:{len(users)} Items:{len(items)} Rows:{len(inter)}')

if __name__ == '__main__':
    main()
