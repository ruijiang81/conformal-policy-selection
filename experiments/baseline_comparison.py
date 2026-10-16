import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_utils import score_correct, weights_correct, train_ai_policy, get_ai_actions
from ConfSelect.confselect import conformal_select, weighted_BH, weighted_CS, eval_sel


METHODS = ['naive', 'unweighted', 'BH', 'hete']
METHOD_LABELS = {
    'naive': 'Naive thresholding',
    'unweighted': 'Unweighted CS (ignores shift)',
    'BH': 'Weighted BH',
    'hete': 'WCS (hete pruning)',
}
METHOD_COLORS = {'naive': '#999999', 'unweighted': '#E69F00', 'BH': '#0072B2', 'hete': '#D55E00'}
METHOD_MARKERS = {'naive': 'x', 'unweighted': '^', 'BH': 'o', 'hete': 's'}
METHOD_LINESTYLES = {'naive': ':', 'unweighted': '--', 'BH': '-', 'hete': '-.'}


def gen_policy_data(n, p, beta_0, beta_1, beta_T, sig_base, hetero_slope, prop_strength):
    X = np.random.normal(0, 1, n * p).reshape((n, p))
    z = X @ beta_T
    prob = 1.0 / (1.0 + np.exp(-prop_strength * z))
    prob = prob.clip(0.05, 0.95)
    T = np.random.binomial(1, prob, n)
    sd = np.clip(sig_base + hetero_slope * z, 0.1, None)
    Y0 = X @ beta_0 + np.random.normal(0, sd, n)
    Y1 = X @ beta_1 + np.random.normal(0, sd, n)
    Y = Y0 * (1 - T) + Y1 * T
    return X, Y0, Y1, Y, T, prob


def run_one_rep(args, beta_0, beta_1, beta_T, q_grid):
    gen = lambda n: gen_policy_data(n, args.p, beta_0, beta_1, beta_T,
                                    args.sig_base, args.hetero_slope, args.prop_strength)
    X_tr, _, _, Y_tr, T_tr, _ = gen(args.n_train)
    X_ca, _, _, Y_ca, T_ca, prob_ca = gen(args.n_calib)
    X_te, Y0_te, Y1_te, _, T_te, prob_te = gen(args.n_test)

    ai_model = train_ai_policy(X_tr, Y_tr, T_tr)
    a_ca = get_ai_actions(ai_model, X_ca)
    a_te = get_ai_actions(ai_model, X_te)

    y_policy = Y0_te * (1 - a_te) + Y1_te * a_te
    c_te = np.full(args.n_test, args.c)

    calib_scores = score_correct(X_ca, Y_ca, T_ca, beta_0, beta_1)
    mu_te = X_te @ beta_0 * (1 - a_te) + X_te @ beta_1 * a_te
    test_scores = c_te - mu_te
    calib_weights = weights_correct(T_ca, prob_ca, a_ca)
    test_weights = weights_correct(T_te, prob_te, a_te)
    matched = (T_ca == a_ca)

    out = {m: {'fdr': np.zeros(len(q_grid)), 'power': np.zeros(len(q_grid)),
               'n_sel': np.zeros(len(q_grid))} for m in METHODS}

    naive_sel = np.where(mu_te > c_te)[0].astype(int)
    naive_fdp, naive_pow = eval_sel(naive_sel, y_policy, c_te)

    for qi, q in enumerate(q_grid):
        out['naive']['fdr'][qi] = naive_fdp
        out['naive']['power'][qi] = naive_pow
        out['naive']['n_sel'][qi] = len(naive_sel)

        if matched.sum() > 0:
            u_sel, _ = conformal_select(calib_scores[matched], test_scores, q=q)
        else:
            u_sel = np.array([])
        u_sel = np.asarray(u_sel).astype(int)
        f, pw = eval_sel(u_sel, y_policy, c_te)
        out['unweighted']['fdr'][qi] = f
        out['unweighted']['power'][qi] = pw
        out['unweighted']['n_sel'][qi] = len(u_sel)

        b_sel, _ = weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q=q)
        b_sel = np.asarray(b_sel).astype(int)
        f, pw = eval_sel(b_sel, y_policy, c_te)
        out['BH']['fdr'][qi] = f
        out['BH']['power'][qi] = pw
        out['BH']['n_sel'][qi] = len(b_sel)

        _, h_sel, _ = weighted_CS(calib_scores, calib_weights, test_scores, test_weights,
                                  q=q, rand='hete')
        h_sel = np.asarray(h_sel).astype(int)
        f, pw = eval_sel(h_sel, y_policy, c_te)
        out['hete']['fdr'][qi] = f
        out['hete']['power'][qi] = pw
        out['hete']['n_sel'][qi] = len(h_sel)

    return out


def aggregate(reps, q_grid):
    agg = {'q_grid': q_grid}
    for m in METHODS:
        fdr = np.array([r[m]['fdr'] for r in reps])
        pwr = np.array([r[m]['power'] for r in reps])
        nsel = np.array([r[m]['n_sel'] for r in reps])
        agg[f'fdr_{m}_mean'] = fdr.mean(0)
        agg[f'fdr_{m}_std'] = fdr.std(0)
        agg[f'power_{m}_mean'] = pwr.mean(0)
        agg[f'power_{m}_std'] = pwr.std(0)
        agg[f'n_sel_{m}_mean'] = nsel.mean(0)
    return agg


def plot_comparison(agg, output_path):
    q = agg['q_grid']
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(16, 7))
    for m in METHODS:
        style = dict(color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                     linestyle=METHOD_LINESTYLES[m], markersize=9, linewidth=2.5,
                     capsize=4, alpha=0.85, markerfacecolor='none', markeredgewidth=2)
        ax_f.errorbar(q, agg[f'fdr_{m}_mean'], yerr=agg[f'fdr_{m}_std'],
                      label=METHOD_LABELS[m], **style)
        ax_p.errorbar(q, agg[f'power_{m}_mean'], yerr=agg[f'power_{m}_std'],
                      label=METHOD_LABELS[m], **style)
    ax_f.plot(q, q, 'k--', linewidth=2, alpha=0.7, zorder=1, label='Nominal FDR (q)')
    ax_f.set_xlabel('Nominal level q', fontsize=18, fontweight='bold')
    ax_f.set_ylabel('Empirical FDR', fontsize=18, fontweight='bold')
    ax_f.tick_params(labelsize=13)
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=12, frameon=True, loc='upper left')
    ax_p.set_xlabel('Nominal level q', fontsize=18, fontweight='bold')
    ax_p.set_ylabel('Empirical power', fontsize=18, fontweight='bold')
    ax_p.tick_params(labelsize=13)
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=12, frameon=True, loc='upper left')
    fig.tight_layout()
    base = os.path.splitext(output_path)[0]
    for ext in ('pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train', type=int, default=300)
    p.add_argument('--n_calib', type=int, default=1200)
    p.add_argument('--n_test', type=int, default=100)
    p.add_argument('--p', type=int, default=5)
    p.add_argument('--c', type=float, default=0.0)
    p.add_argument('--sig_base', type=float, default=0.4)
    p.add_argument('--hetero_slope', type=float, default=2.5)
    p.add_argument('--prop_strength', type=float, default=5.0)
    p.add_argument('--n_reps', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--q_min', type=float, default=0.05)
    p.add_argument('--q_max', type=float, default=0.95)
    p.add_argument('--q_step', type=float, default=0.1)
    p.add_argument('--output_dir', type=str, default='results/baseline_comparison')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    q_grid = np.arange(args.q_min, args.q_max + 1e-9, args.q_step)

    reps = []
    for rep in range(args.n_reps):
        np.random.seed(args.seed + rep + 1)
        beta_0 = np.random.normal(0, 1, args.p)
        beta_1 = np.random.normal(0, 1, args.p)
        beta_T = np.random.normal(0, 1, args.p)
        reps.append(run_one_rep(args, beta_0, beta_1, beta_T, q_grid))
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f'rep {rep + 1}/{args.n_reps} done')

    agg = aggregate(reps, q_grid)

    rows = []
    for qi, q in enumerate(q_grid):
        for m in METHODS:
            rows.append({
                'q': round(float(q), 4), 'method': m, 'method_label': METHOD_LABELS[m],
                'fdr_mean': agg[f'fdr_{m}_mean'][qi], 'fdr_std': agg[f'fdr_{m}_std'][qi],
                'power_mean': agg[f'power_{m}_mean'][qi], 'power_std': agg[f'power_{m}_std'][qi],
                'n_sel_mean': agg[f'n_sel_{m}_mean'][qi],
            })
    csv_path = os.path.join(args.output_dir, 'baseline_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    plot_comparison(agg, os.path.join(args.output_dir, 'baseline_comparison'))

    print('\n' + '=' * 78)
    print('BASELINE COMPARISON -- empirical FDR (mean over reps); "*" flags FDR > q')
    print('=' * 78)
    header = f'{"q":>6} | ' + ' | '.join(f'{METHOD_LABELS[m][:16]:>16}' for m in METHODS)
    print(header)
    print('-' * len(header))
    for qi, q in enumerate(q_grid):
        cells = []
        for m in METHODS:
            v = agg[f'fdr_{m}_mean'][qi]
            flag = '*' if v > q + 1e-9 else ' '
            cells.append(f'{v:>15.3f}{flag}')
        print(f'{q:>6.2f} | ' + ' | '.join(cells))
    print('=' * 78)


if __name__ == '__main__':
    main()
