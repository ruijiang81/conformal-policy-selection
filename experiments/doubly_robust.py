import argparse
import os

import numpy as np
import pandas as pd

from experiment_utils import (
    gen_synthetic_data, score_correct, score_wrong, weights_correct, weights_wrong,
    train_ai_policy, get_ai_actions, run_selection_experiment, aggregate_results,
    plot_fdr_power, plot_fdr_panel,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_calib', type=int, default=2000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--p', type=int, default=5)
    parser.add_argument('--sig', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=0.0)
    parser.add_argument('--n_reps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', nargs='+', default=['BH', 'hete'])
    parser.add_argument('--output_dir', type=str, default='results/doubly_robust')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    beta_0 = np.random.normal(0, 1, args.p)
    beta_1 = np.random.normal(0, 1, args.p)
    beta_T = np.random.normal(0, 1, args.p)
    beta_wrong_0 = np.random.normal(0, 1, args.p) * 3
    beta_wrong_1 = np.random.normal(0, 1, args.p) * 3
    beta_wrong_w = np.random.normal(0, 1, args.p)

    q_grid = np.arange(0.05, 1.0, 0.1)
    methods = tuple(args.methods)
    scenario_names = ['Both Correct', 'Correct Propensity Score Only',
                      'Correct Reward Model Only', 'Both Wrong']
    scenario_results = {name: [] for name in scenario_names}

    for rep in range(args.n_reps):
        np.random.seed(args.seed + rep + 1)
        X_train, Y0_train, Y1_train, Y_train, T_train, prob_train = \
            gen_synthetic_data(args.n_train, args.p, beta_0, beta_1, beta_T, args.sig)
        X_calib, Y0_calib, Y1_calib, Y_calib, T_calib, prob_calib = \
            gen_synthetic_data(args.n_calib, args.p, beta_0, beta_1, beta_T, args.sig)
        X_test, Y0_test, Y1_test, Y_test, T_test, prob_test = \
            gen_synthetic_data(args.n_test, args.p, beta_0, beta_1, beta_T, args.sig)

        ai_model = train_ai_policy(X_train, Y_train, T_train)
        a_calib = get_ai_actions(ai_model, X_calib)
        a_test = get_ai_actions(ai_model, X_test)

        # Test point is drawn from the target Q: it receives the policy action
        # (paper setup, T_{n+j} ~ pi(X_{n+j})), so its self-weight is
        # 1/pi_0(pi(X)|X). Calibration stays observational (~ P).
        T_test = a_test

        y_policy = Y0_test * (1 - a_test) + Y1_test * a_test
        c_test = np.full(args.n_test, args.c)

        s_correct_calib = score_correct(X_calib, Y_calib, T_calib, beta_0, beta_1)
        mu_test_correct = X_test @ beta_0 * (1 - a_test) + X_test @ beta_1 * a_test
        s_correct_test = c_test - mu_test_correct

        s_wrong_calib = score_wrong(X_calib, Y_calib, T_calib, beta_wrong_0, beta_wrong_1)
        mu_wrong_test = X_test @ beta_wrong_0 * (1 - a_test) + X_test @ beta_wrong_1 * a_test
        s_wrong_test = c_test - mu_wrong_test

        w_correct_calib = weights_correct(T_calib, prob_calib, a_calib)
        w_correct_test = weights_correct(T_test, prob_test, a_test)
        w_wrong_calib = weights_wrong(X_calib, T_calib, beta_wrong_w, a_calib)
        w_wrong_test = weights_wrong(X_test, T_test, beta_wrong_w, a_test)

        scenario_results[scenario_names[0]].append(run_selection_experiment(
            s_correct_calib, w_correct_calib, s_correct_test, w_correct_test,
            y_policy, c_test, q_grid, methods))
        scenario_results[scenario_names[1]].append(run_selection_experiment(
            s_wrong_calib, w_correct_calib, s_wrong_test, w_correct_test,
            y_policy, c_test, q_grid, methods))
        scenario_results[scenario_names[2]].append(run_selection_experiment(
            s_correct_calib, w_wrong_calib, s_correct_test, w_wrong_test,
            y_policy, c_test, q_grid, methods))
        scenario_results[scenario_names[3]].append(run_selection_experiment(
            s_wrong_calib, w_wrong_calib, s_wrong_test, w_wrong_test,
            y_policy, c_test, q_grid, methods))

        if (rep + 1) % 10 == 0 or rep == 0:
            print(f'Rep {rep + 1}/{args.n_reps} done')

    scenario_aggs = []
    for name in scenario_names:
        agg = aggregate_results(scenario_results[name], q_grid, methods)
        scenario_aggs.append(agg)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        plot_fdr_power(agg, methods, title='', output_path=os.path.join(args.output_dir, safe_name))

    plot_fdr_panel(scenario_aggs, scenario_names, methods,
                   output_path=os.path.join(args.output_dir, 'dr_panel_fdr'))

    print('\n' + '=' * 70)
    print('DOUBLY ROBUST EXPERIMENT -- SUMMARY')
    print('=' * 70)
    qi_report = np.argmin(np.abs(q_grid - 0.15))
    for name, agg in zip(scenario_names, scenario_aggs):
        for method in methods:
            print(f'{name:<30} {method:<12} '
                  f'FDR={agg[f"fdr_{method}_mean"][qi_report]:.4f}  '
                  f'Power={agg[f"power_{method}_mean"][qi_report]:.4f}')
    print('=' * 70)

    rows = []
    for name, agg in zip(scenario_names, scenario_aggs):
        for qi, q in enumerate(q_grid):
            for method in methods:
                rows.append({
                    'scenario': name, 'q': q, 'method': method,
                    'fdr_mean': agg[f'fdr_{method}_mean'][qi],
                    'fdr_std': agg[f'fdr_{method}_std'][qi],
                    'power_mean': agg[f'power_{method}_mean'][qi],
                    'power_std': agg[f'power_{method}_std'][qi],
                })
    csv_path = os.path.join(args.output_dir, 'dr_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
