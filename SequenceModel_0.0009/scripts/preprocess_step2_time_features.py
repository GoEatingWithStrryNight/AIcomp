import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('data/processed')

TIME_COLS = ['borrow_time','return_time','renew_time']

# Possible datetime formats observed (mixed)
DT_FORMATS = [
    '%Y/%m/%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y-%m-%d%H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d%H:%M',
]

def parse_mixed_datetime(series: pd.Series):
    def try_parse(x):
        if pd.isna(x) or str(x).strip()=='' or str(x).lower()=='nan':
            return pd.NaT
        s = str(x).strip()
        for fmt in DT_FORMATS:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        # last resort
        try:
            return pd.to_datetime(s, errors='coerce')
        except Exception:
            return pd.NaT
    return series.apply(try_parse)

def add_time_features(df: pd.DataFrame):
    df['borrow_year'] = df['borrow_time'].dt.year
    df['borrow_month'] = df['borrow_time'].dt.month
    df['borrow_dow'] = df['borrow_time'].dt.dayofweek
    df['borrow_hour'] = df['borrow_time'].dt.hour
    # duration features
    df['borrow_duration_days'] = (df['return_time'] - df['borrow_time']).dt.total_seconds()/86400
    df['renew_gap_days'] = (df['renew_time'] - df['borrow_time']).dt.total_seconds()/86400
    return df

def main():
    inter_path = PROCESSED_DIR / 'interactions_step1.csv'
    interactions = pd.read_csv(inter_path)

    # Parse times
    for c in TIME_COLS:
        if c in interactions.columns:
            interactions[c] = parse_mixed_datetime(interactions[c])

    interactions = add_time_features(interactions)

    # Basic sanity: negative durations -> NaN
    for col in ['borrow_duration_days','renew_gap_days']:
        interactions.loc[interactions[col] < 0, col] = pd.NA

    interactions.to_csv(PROCESSED_DIR / 'interactions_step2_time.csv', index=False)
    print('Saved interactions_step2_time.csv with time features')

if __name__ == '__main__':
    main()
