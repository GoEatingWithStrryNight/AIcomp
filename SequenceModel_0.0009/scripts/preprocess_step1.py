import os
import pandas as pd
from pathlib import Path

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Column rename maps
INTER_RENAME = {
    'inter_id': 'inter_id',
    'user_id': 'user_id',
    'book_id': 'book_id',
    '借阅时间': 'borrow_time',
    '还书时间': 'return_time',
    '续借时间': 'renew_time',
    '续借次数': 'renew_cnt'
}
ITEM_RENAME = {
    'book_id': 'book_id',
    '题名': 'title',
    '作者': 'author',
    '出版社': 'publisher',
    '一级分类': 'cat_lvl1',
    '二级分类': 'cat_lvl2'
}
USER_RENAME = {
    '借阅人': 'user_id',
    '性别': 'gender',
    'DEPT': 'dept',
    '年级': 'grade',
    '类型': 'user_type'
}

def read_csv_auto(path: Path) -> pd.DataFrame:
    # Try utf-8, then gbk
    for enc in ('utf-8', 'gbk'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # fallback
    return pd.read_csv(path, encoding='latin1')


def load_and_standardize():
    inter_path = RAW_DIR / 'inter_preliminary.csv'
    item_path = RAW_DIR / 'item.csv'
    user_path = RAW_DIR / 'user.csv'

    interactions = read_csv_auto(inter_path)
    items = read_csv_auto(item_path)
    users = read_csv_auto(user_path)

    # Rename columns (only those existing)
    interactions = interactions.rename(columns={k: v for k, v in INTER_RENAME.items() if k in interactions.columns})
    items = items.rename(columns={k: v for k, v in ITEM_RENAME.items() if k in items.columns})
    users = users.rename(columns={k: v for k, v in USER_RENAME.items() if k in users.columns})

    # Basic trimming / dtype coercion
    for col in ['user_id', 'book_id']:
        if col in interactions.columns:
            interactions[col] = interactions[col].astype(str).str.strip()
    if 'user_id' in users.columns:
        users['user_id'] = users['user_id'].astype(str).str.strip()
    if 'book_id' in items.columns:
        items['book_id'] = items['book_id'].astype(str).str.strip()

    # Save initial cleaned versions
    interactions.to_csv(PROCESSED_DIR / 'interactions_step1.csv', index=False)
    items.to_csv(PROCESSED_DIR / 'items_step1.csv', index=False)
    users.to_csv(PROCESSED_DIR / 'users_step1.csv', index=False)

    print('Step1 saved: interactions_step1.csv, items_step1.csv, users_step1.csv')


if __name__ == '__main__':
    load_and_standardize()
