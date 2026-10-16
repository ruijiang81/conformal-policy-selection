import numpy as np
import pandas as pd

ANTIDIABETIC_COLS = ['med=150', 'med=178']
ONSET_COL = 'icd9=250'


def load_and_prepare(data_path, lookahead=5, min_age=0, min_bmi=0, min_hba1c=0):
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    dtypes = {c: 'float32' for c in header}
    for c in ('id', 'year'):
        if c in dtypes:
            dtypes[c] = 'int32'
    df = pd.read_csv(data_path, dtype=dtypes)

    T = (df[ANTIDIABETIC_COLS].fillna(0).sum(axis=1) > 0).astype(int).values
    df['_T'] = T
    df['_onset'] = df[ONSET_COL].fillna(0).astype(int)
    df = df.sort_values(['id', 'year']).reset_index(drop=True)

    onset_in_window = np.zeros(len(df), dtype=int)
    full_followup = np.ones(len(df), dtype=bool)
    onset_table = df[['id', 'year', '_onset']].rename(columns={'year': '_y_future'})
    base = df[['id', 'year']].copy()
    for h in range(1, lookahead + 1):
        probe = base.copy()
        probe['_y_future'] = probe['year'] + h
        merged = probe.merge(onset_table, on=['id', '_y_future'], how='left')
        seen = merged['_onset'].notna().values
        had = (merged['_onset'].fillna(0).values == 1)
        full_followup &= seen
        onset_in_window = np.maximum(onset_in_window, had.astype(int))
    df['_onset_kyr'] = onset_in_window
    df['_full_followup'] = full_followup
    df['_Y'] = 1 - df['_onset_kyr']
    df = df[df['_full_followup']].copy()

    if min_age > 0 and 'birthyear' in df.columns:
        df['_age'] = df['year'] - df['birthyear']
        df = df[df['_age'] >= min_age].copy()
    if min_bmi > 0 and 'bmi' in df.columns:
        df = df[df['bmi'].fillna(-1) >= min_bmi].copy()
    if min_hba1c > 0 and 'lab=hba1c(%)' in df.columns:
        df = df[df['lab=hba1c(%)'].fillna(-1) >= min_hba1c].copy()

    drop_cols = {'id', 'year', '_onset', '_onset_kyr', '_full_followup', '_Y', '_T',
                 ONSET_COL, *ANTIDIABETIC_COLS}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for c in feature_cols:
        if c.startswith(('med=', 'icd9=', 'lab=')):
            df[c] = df[c].fillna(0)
        else:
            mean = df[c].mean()
            if pd.isna(mean):
                mean = 0.0
            df[c] = df[c].fillna(mean)

    arr = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    df[feature_cols] = arr
    return df, feature_cols


def build_arrays(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    T = df['_T'].values.astype(int)
    Y = df['_Y'].values.astype(int)
    return X, T, Y
