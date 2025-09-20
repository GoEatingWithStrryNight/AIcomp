import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('data/processed')
OUTPUT_FILE = Path('data/processed/submission_popularity.csv')

"""Baseline: For each user recommend the most popular book not yet borrowed.
Popularity = total interaction count (can be refined later).
"""

def build_popularity(interactions: pd.DataFrame) -> pd.Series:
    return interactions['book_id'].value_counts()

def recommend(popularity: pd.Series, user_items: set) -> str:
    for book_id in popularity.index:
        if book_id not in user_items:
            return book_id
    # fallback: if user has seen everything in popularity list
    return popularity.index[0]

def main():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_clean.csv', usecols=['user_id','book_id'])

    popularity = build_popularity(inter)
    user_groups = inter.groupby('user_id')['book_id'].apply(set)

    recs = []
    for user_id, items in user_groups.items():
        book = recommend(popularity, items)
        recs.append((user_id, book))

    sub = pd.DataFrame(recs, columns=['user_id','book_id'])
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f'Submission saved to {OUTPUT_FILE} shape={sub.shape}')

if __name__ == '__main__':
    main()
