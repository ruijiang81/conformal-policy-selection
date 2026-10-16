import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ConfSelect.confselect import weighted_BH, weighted_CS, eval_sel


def gen_synthetic_data(n, p, beta_0, beta_1, beta_T, sig=1.0):
    X = np.random.normal(0, 1, n * p).reshape((n, p))
    logit = X @ beta_T * 3
    prob = 1 / (1 + np.exp(-logit))
    prob = prob.clip(0.1, 0.9)
    T = np.random.binomial(1, prob, n)
    Y0 = X @ beta_0 + np.random.normal(0, sig, n)
    Y1 = X @ beta_1 + np.random.normal(0, sig, n)
    Y = Y0 * (1 - T) + Y1 * T
    return X, Y0, Y1, Y, T, prob


def score_correct(X, Y, T, beta_0, beta_1):
    mu = X @ beta_0 * (1 - T) + X @ beta_1 * T
    return Y - mu


def score_wrong(X, Y, T, beta_wrong_0, beta_wrong_1):
    mu = X @ beta_wrong_0 * (1 - T) + X @ beta_wrong_1 * T
    return Y - mu


def score_model(X_with_T, Y, model):
    return Y - model.predict(X_with_T)


def weights_correct(T_observed, prob, target_actions):
    n = len(T_observed)
    weights = np.zeros(n)
    for i in range(n):
        p_obs = prob[i] if T_observed[i] == 1 else 1 - prob[i]
        weights[i] = (T_observed[i] == target_actions[i]) / p_obs
    return weights


def weights_wrong(X, T_observed, beta_wrong, target_actions):
    logit = -X @ beta_wrong * 8
    prob_wrong = 1 / (1 + np.exp(-logit))
    prob_wrong = prob_wrong.clip(0.02, 0.98)
    n = len(T_observed)
    weights = np.zeros(n)
    for i in range(n):
        p_obs = prob_wrong[i] if T_observed[i] == 1 else 1 - prob_wrong[i]
        weights[i] = (T_observed[i] == target_actions[i]) / p_obs
    return weights


def train_ai_policy(X_train, Y_train, T_train):
    X_with_T = np.column_stack([X_train, T_train])
    return LinearRegression().fit(X_with_T, Y_train)


def get_ai_actions(model, X):
    mu0 = model.predict(np.column_stack([X, np.zeros(len(X))]))
    mu1 = model.predict(np.column_stack([X, np.ones(len(X))]))
    return (mu1 > mu0).astype(int)


def run_selection_experiment(calib_scores, calib_weights, test_scores, test_weights,
                             y_policy, c_test, q_grid, methods=('BH', 'hete'), y_counterfactual=None):
    results = {}
    for method in methods:
        for key in ('fdr', 'power', 'n_selected', 'outcome_selected', 'outcome_nonselected',
                    'outcome_selected_counterfactual', 'outcome_nonselected_counterfactual',
                    'frac_helped_selected', 'frac_helped_nonselected'):
            results[f'{key}_{method}'] = np.zeros(len(q_grid))

    for qi, q in enumerate(q_grid):
        for method in methods:
            if method == 'BH':
                sel_idx, _ = weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q=q)
            else:
                _, sel_idx, _ = weighted_CS(calib_scores, calib_weights, test_scores, test_weights,
                                            q=q, rand=method)

            fdp, power = eval_sel(sel_idx.astype(int), y_policy, c_test)
            results[f'fdr_{method}'][qi] = fdp
            results[f'power_{method}'][qi] = power

            n_test = len(y_policy)
            sel_mask = np.zeros(n_test, dtype=bool)
            if len(sel_idx) > 0:
                sel_mask[sel_idx.astype(int)] = True
            results[f'n_selected_{method}'][qi] = sel_mask.sum()

            if sel_mask.sum() > 0:
                results[f'outcome_selected_{method}'][qi] = y_policy[sel_mask].mean()
                results[f'frac_helped_selected_{method}'][qi] = (y_policy[sel_mask] > c_test[sel_mask]).mean()
                results[f'outcome_selected_counterfactual_{method}'][qi] = \
                    y_counterfactual[sel_mask].mean() if y_counterfactual is not None else np.nan
            else:
                results[f'outcome_selected_{method}'][qi] = np.nan
                results[f'frac_helped_selected_{method}'][qi] = np.nan
                results[f'outcome_selected_counterfactual_{method}'][qi] = np.nan

            if (~sel_mask).sum() > 0:
                results[f'outcome_nonselected_{method}'][qi] = y_policy[~sel_mask].mean()
                results[f'frac_helped_nonselected_{method}'][qi] = (y_policy[~sel_mask] > c_test[~sel_mask]).mean()
                results[f'outcome_nonselected_counterfactual_{method}'][qi] = \
                    y_counterfactual[~sel_mask].mean() if y_counterfactual is not None else np.nan
            else:
                results[f'outcome_nonselected_{method}'][qi] = np.nan
                results[f'frac_helped_nonselected_{method}'][qi] = np.nan
                results[f'outcome_nonselected_counterfactual_{method}'][qi] = np.nan

    return results


def aggregate_results(results_list, q_grid, methods=('BH', 'hete')):
    agg = {}
    for method in methods:
        fdr_stack = np.array([r[f'fdr_{method}'] for r in results_list])
        power_stack = np.array([r[f'power_{method}'] for r in results_list])
        agg[f'fdr_{method}_mean'] = np.mean(fdr_stack, axis=0)
        agg[f'fdr_{method}_std'] = np.std(fdr_stack, axis=0)
        agg[f'power_{method}_mean'] = np.mean(power_stack, axis=0)
        agg[f'power_{method}_std'] = np.std(power_stack, axis=0)
        agg[f'n_selected_{method}_mean'] = np.mean([r[f'n_selected_{method}'] for r in results_list], axis=0)
        for key in ('outcome_selected', 'outcome_nonselected', 'outcome_selected_counterfactual',
                    'outcome_nonselected_counterfactual', 'frac_helped_selected', 'frac_helped_nonselected'):
            agg[f'{key}_{method}_mean'] = np.nanmean([r[f'{key}_{method}'] for r in results_list], axis=0)
    agg['q_grid'] = q_grid
    return agg


METHOD_LABELS = {'BH': 'Weighted BH', 'hete': 'WCS (Hete)', 'homo': 'WCS (Homo)', 'dtm': 'WCS (DTM)'}
METHOD_MARKERS = {'BH': 'o', 'hete': 's', 'homo': '^', 'dtm': 'D'}
METHOD_COLORS = {'BH': '#0072B2', 'hete': '#D55E00', 'homo': '#009E73', 'dtm': '#CC79A7'}
METHOD_LINESTYLES = {'BH': '-', 'hete': '-.', 'homo': ':', 'dtm': (0, (3, 1, 1, 1, 1, 1))}
METHOD_ALPHA = 0.6


def _method_style(method, idx=0, filled=False):
    color = METHOD_COLORS.get(method)
    return dict(color=color, marker=METHOD_MARKERS.get(method, 'o'),
                markersize=max(6, 13 - 3 * idx),
                markerfacecolor=color if filled else 'none', markeredgewidth=2.5,
                alpha=METHOD_ALPHA, linewidth=max(2.0, 4.5 - 1.5 * idx))


def plot_fdr_power(agg, methods, title='', output_path=None):
    q_grid = agg['q_grid']
    if output_path:
        base_path = os.path.splitext(output_path)[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for idx, method in enumerate(methods):
        ax.errorbar(q_grid, agg[f'fdr_{method}_mean'], yerr=agg[f'fdr_{method}_std'],
                    label=METHOD_LABELS.get(method, method), capsize=6,
                    linestyle=METHOD_LINESTYLES.get(method, '-'), **_method_style(method, idx))
    ax.plot(q_grid, q_grid, 'k--', linewidth=2.5, alpha=0.7, zorder=1, label='Nominal FDR')
    ax.set_xlabel('q', fontsize=22, fontweight='bold')
    ax.set_ylabel('FDR', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=2, length=6)
    ax.grid(alpha=0.3, linewidth=1.5)
    ax.legend(fontsize=16, frameon=True, shadow=True, loc='upper center',
              bbox_to_anchor=(0.5, 1.15), ncol=len(methods) + 1)
    fig.tight_layout()
    if output_path:
        fig.savefig(f'{base_path}_FDR.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for idx, method in enumerate(methods):
        ax.errorbar(q_grid, agg[f'power_{method}_mean'], yerr=agg[f'power_{method}_std'],
                    label=METHOD_LABELS.get(method, method), capsize=6,
                    linestyle=METHOD_LINESTYLES.get(method, '-'), **_method_style(method, idx))
    ax.set_xlabel('q', fontsize=22, fontweight='bold')
    ax.set_ylabel('Power', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=2, length=6)
    ax.grid(alpha=0.3, linewidth=1.5)
    ax.legend(fontsize=16, frameon=True, shadow=True, loc='upper center',
              bbox_to_anchor=(0.5, 1.15), ncol=len(methods))
    fig.tight_layout()
    if output_path:
        fig.savefig(f'{base_path}_Power.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_selection_comparison(agg, methods, title='', output_path=None):
    q_grid = agg['q_grid']
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for idx, method in enumerate(methods):
        label = METHOD_LABELS.get(method, method)
        ax.plot(q_grid, agg[f'outcome_selected_{method}_mean'], label=f'{label} (Selected)',
                linestyle='-', **_method_style(method, idx, filled=True))
        ax.plot(q_grid, agg[f'outcome_nonselected_{method}_mean'], label=f'{label} (Non-selected)',
                linestyle='--', **_method_style(method, idx, filled=False))
    ax.set_xlabel('q', fontsize=22, fontweight='bold')
    ax.set_ylabel('Mean Outcome ($)', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=2, length=6)
    ax.grid(alpha=0.3, linewidth=1.5)
    ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.12),
              frameon=True, shadow=True, ncol=2)
    fig.tight_layout()
    if output_path:
        fig.savefig(f'{os.path.splitext(output_path)[0]}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_fdr_panel(scenario_aggs, scenario_labels, methods, output_path=None):
    n = len(scenario_aggs)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 7 * nrows))
    axes = axes.flatten()
    for idx, (agg, label) in enumerate(zip(scenario_aggs, scenario_labels)):
        ax = axes[idx]
        q_grid = agg['q_grid']
        for m_idx, method in enumerate(methods):
            ax.errorbar(q_grid, agg[f'fdr_{method}_mean'], yerr=agg[f'fdr_{method}_std'],
                        label=METHOD_LABELS.get(method, method), capsize=6,
                        linestyle=METHOD_LINESTYLES.get(method, '-'), **_method_style(method, m_idx))
        ax.plot(q_grid, q_grid, 'k--', linewidth=2.5, alpha=0.7, zorder=1, label='Nominal FDR')
        ax.set_xlabel('q', fontsize=20, fontweight='bold')
        ax.set_ylabel('FDR', fontsize=20, fontweight='bold')
        ax.set_title(label, fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16, width=2, length=6)
        ax.legend(fontsize=14, frameon=True, shadow=True, loc='upper left')
        ax.grid(alpha=0.3, linewidth=1.5)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    if output_path:
        fig.savefig(f'{os.path.splitext(output_path)[0]}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
