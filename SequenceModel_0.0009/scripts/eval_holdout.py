import pandas as pd
from pathlib import Path
import argparse
import json

PROCESSED_DIR = Path('data/processed')
ANS_DIR = Path('data/ans')

"""Simple time-based holdout evaluation.
Steps:
1. Load interactions_clean.csv (must contain borrow_time or already parsed earlier steps). If missing borrow_time parsed, we reload step2 file.
2. Split by borrow_time: use the latest X percent (time-based) as test, the rest as train.
3. For each user in train that also appears in test, check if submission (one book per user) matches ANY of that user's test books.
Because only 1 recommendation per user: Precision = Recall = HitRate; F1 = same value.
"""

def load_interactions():
    inter = pd.read_csv(PROCESSED_DIR / 'interactions_step2_time.csv')
    # Ensure borrow_time parsed
    inter['borrow_time'] = pd.to_datetime(inter['borrow_time'], errors='coerce')
    inter = inter.dropna(subset=['borrow_time'])
    return inter[['user_id','book_id','borrow_time']]

def time_split(inter: pd.DataFrame, test_ratio: float = 0.1):
    # Sort by time
    inter = inter.sort_values('borrow_time')
    cutoff_index = int(len(inter) * (1 - test_ratio))
    cutoff_time = inter.iloc[cutoff_index]['borrow_time']
    train = inter[inter['borrow_time'] < cutoff_time]
    test = inter[inter['borrow_time'] >= cutoff_time]
    return train, test, cutoff_time

def evaluate(sub_path: Path, test: pd.DataFrame):
    if not sub_path.exists():
        raise FileNotFoundError(sub_path)
    sub = pd.read_csv(sub_path)
    # Normalize dtypes to string in case
    sub['user_id'] = sub['user_id'].astype(str)
    sub['book_id'] = sub['book_id'].astype(str)
    test = test.copy()
    test['user_id'] = test['user_id'].astype(str)
    test['book_id'] = test['book_id'].astype(str)

    # Build ground truth per user
    gt = test.groupby('user_id')['book_id'].apply(set)

    hits = 0
    total_users = 0
    for _, row in sub.iterrows():
        u = row['user_id']
        b = row['book_id']
        if u in gt:
            total_users += 1
            if b in gt[u]:
                hits += 1
    if total_users == 0:
        return {'users_in_test':0,'hits':0,'precision':0,'recall':0,'f1':0}
    precision = hits / total_users
    recall = precision  # single recommendation
    f1 = precision  # same
    return {
        'users_in_test': total_users,
        'hits': hits,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def leave_one_out_split(inter: pd.DataFrame):
    # For each user keep last interaction as test, rest as train
    inter = inter.sort_values('borrow_time')
    last_idx = inter.groupby('user_id').tail(1).index
    test = inter.loc[last_idx]
    train = inter.drop(last_idx)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str, required=True, help='Path to submission csv')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--mode', type=str, choices=['window','loo'], default='window', help='window: last time ratio; loo: leave-one-out')
    parser.add_argument('--json_out', type=str, default=None, help='Optional path to save metrics json')
    args = parser.parse_args()

    inter = load_interactions()
    if args.mode == 'window':
        train, test, cutoff_time = time_split(inter, test_ratio=args.test_ratio)
        metrics = evaluate(Path(args.submission), test)
        print(f'Cutoff time: {cutoff_time}')
    else:  # LOO
        train, test = leave_one_out_split(inter)
        metrics = evaluate(Path(args.submission), test)
        print('Mode: Leave-One-Out (last interaction per user as test)')
    # High precision formatting
    metrics_fmt = {k: (round(v,6) if isinstance(v, float) else v) for k,v in metrics.items()}
    print(f'Metrics: {metrics_fmt}')
    if args.json_out:
        with open(args.json_out,'w',encoding='utf-8') as f:
            json.dump(metrics_fmt,f,ensure_ascii=False,indent=2)

if __name__ == '__main__':
    main()
