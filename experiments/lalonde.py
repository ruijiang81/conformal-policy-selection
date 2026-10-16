import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    run_selection_experiment, aggregate_results, plot_fdr_power, plot_selection_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/lalonde_data.csv')
    parser.add_argument('--c', type=float, default=4000.0)
    parser.add_argument('--n_reps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', nargs='+', default=['BH', 'hete'])
    parser.add_argument('--output_dir', type=str, default='results/lalonde')
    return parser.parse_args()


def load_lalonde(data_path):
    df = pd.read_csv(data_path)
    feature_cols = ['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']
    X = df[feature_cols].values.astype(float)
    T = df['treat'].values.astype(int)
    Y = df['re78'].values.astype(float)
    return X, T, Y, df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_all, T_all, Y_all, df_all = load_lalonde(args.data_path)
    n_total = len(Y_all)
    print(f'Loaded {n_total} observations; treat-rate={T_all.mean():.3f}')

    q_grid = np.arange(0.05, 1.0, 0.1)
    methods = tuple(args.methods)
    results_list = []

    for rep in range(args.n_reps):
        rng = np.random.RandomState(args.seed + rep)
        idx = np.arange(n_total)
        idx_train, idx_rest = train_test_split(idx, test_size=0.7, stratify=T_all, random_state=rng)
        T_rest = T_all[idx_rest]
        idx_calib, idx_test = train_test_split(idx_rest, test_size=3 / 7,
                                               stratify=T_rest, random_state=rng)

        X_train, T_train, Y_train = X_all[idx_train], T_all[idx_train], Y_all[idx_train]
        X_calib, T_calib, Y_calib = X_all[idx_calib], T_all[idx_calib], Y_all[idx_calib]
        X_test, T_test, Y_test = X_all[idx_test], T_all[idx_test], Y_all[idx_test]

        outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=rng.randint(1000000))
        outcome_model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
        outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=rng.randint(1000000))
        outcome_model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

        scaler = StandardScaler().fit(X_train)
        prop_model = LogisticRegression(max_iter=2000, random_state=rng.randint(1000000))
        prop_model.fit(scaler.transform(X_train), T_train)

        def get_actions(X):
            return (outcome_model_1.predict(X) > outcome_model_0.predict(X)).astype(int)

        def predict_outcome(X, T):
            m0 = outcome_model_0.predict(X)
            m1 = outcome_model_1.predict(X)
            return np.where(T == 1, m1, m0)

        a_calib = get_actions(X_calib)
        a_test = get_actions(X_test)

        calib_scores = Y_calib - predict_outcome(X_calib, T_calib)
        mu_test_policy = np.where(a_test == 1, outcome_model_1.predict(X_test),
                                  outcome_model_0.predict(X_test))
        test_scores = args.c - mu_test_policy

        prob_calib = prop_model.predict_proba(scaler.transform(X_calib))[:, 1].clip(0.05, 0.95)
        prob_test = prop_model.predict_proba(scaler.transform(X_test))[:, 1].clip(0.05, 0.95)

        def compute_weights(T_obs, prob, actions):
            p_obs = np.where(T_obs == 1, prob, 1 - prob)
            return (T_obs == actions).astype(float) / p_obs

        calib_weights = compute_weights(T_calib, prob_calib, a_calib)
        test_weights = compute_weights(T_test, prob_test, a_test)

        y_policy = np.where(T_test == a_test, Y_test, mu_test_policy)
        a_opposite = 1 - a_test
        mu_test_opposite = np.where(a_opposite == 1, outcome_model_1.predict(X_test),
                                    outcome_model_0.predict(X_test))
        y_counterfactual = np.where(T_test == a_opposite, Y_test, mu_test_opposite)
        c_test = np.full(len(X_test), args.c)

        res = run_selection_experiment(
            calib_scores, calib_weights, test_scores, test_weights,
            y_policy, c_test, q_grid, methods, y_counterfactual=y_counterfactual)
        results_list.append(res)

        if (rep + 1) % 20 == 0 or rep == 0:
            print(f'Rep {rep + 1}/{args.n_reps} done')

    agg = aggregate_results(results_list, q_grid, methods)
    plot_fdr_power(agg, methods, title='', output_path=os.path.join(args.output_dir, 'lalonde'))
    plot_selection_comparison(agg, methods, title='',
                              output_path=os.path.join(args.output_dir, 'lalonde_selection_comparison'))

    print('\n' + '=' * 60)
    print(f'LALONDE -- threshold c=${args.c:.0f}, reps={args.n_reps}')
    qi_report = np.argmin(np.abs(q_grid - 0.15))
    for method in methods:
        print(f'{method:<12} FDR={agg[f"fdr_{method}_mean"][qi_report]:.4f}  '
              f'Power={agg[f"power_{method}_mean"][qi_report]:.4f}  '
              f'N_sel={agg[f"n_selected_{method}_mean"][qi_report]:.1f}')
    print('=' * 60)

    rows = []
    for qi, q in enumerate(q_grid):
        for method in methods:
            rows.append({
                'q': q, 'method': method,
                'fdr_mean': agg[f'fdr_{method}_mean'][qi],
                'fdr_std': agg[f'fdr_{method}_std'][qi],
                'power_mean': agg[f'power_{method}_mean'][qi],
                'power_std': agg[f'power_{method}_std'][qi],
                'n_selected_mean': agg[f'n_selected_{method}_mean'][qi],
                'outcome_selected_mean': agg[f'outcome_selected_{method}_mean'][qi],
                'outcome_nonselected_mean': agg[f'outcome_nonselected_{method}_mean'][qi],
                'frac_helped_selected_mean': agg[f'frac_helped_selected_{method}_mean'][qi],
                'frac_helped_nonselected_mean': agg[f'frac_helped_nonselected_{method}_mean'][qi],
            })
    csv_path = os.path.join(args.output_dir, 'lalonde_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
