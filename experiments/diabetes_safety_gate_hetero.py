import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from diabetes_data import load_and_prepare, build_arrays
from ConfSelect import weighted_BH, weighted_CS, eval_sel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=os.environ.get('DIABETES_DATA_PATH'),
                   required='DIABETES_DATA_PATH' not in os.environ,
                   help='path to the EHR csv (or set DIABETES_DATA_PATH)')
    p.add_argument('--lookahead', type=int, default=3)
    p.add_argument('--min_age', type=float, default=60)
    p.add_argument('--min_bmi', type=float, default=30)
    p.add_argument('--pos_frac', type=float, default=0.40)
    p.add_argument('--tau_scale', type=float, default=0.15)
    p.add_argument('--mu0_spread', type=float, default=0.0)
    p.add_argument('--sigma', type=float, default=0.10)
    p.add_argument('--null_quantile', type=float, default=0.50)
    p.add_argument('--c', type=float, default=None)
    p.add_argument('--cate_learner', type=str, default='dr', choices=['dr', 't'])
    p.add_argument('--behavior_frac', type=float, default=0.40)
    p.add_argument('--behavior_scale', type=float, default=1.0)
    p.add_argument('--behavior_seed', type=int, default=7)
    p.add_argument('--n_reps', type=int, default=20)
    p.add_argument('--seed', type=int, default=500)
    p.add_argument('--n_estimators', type=int, default=200)
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--subsample', type=int, default=0)
    p.add_argument('--max_test', type=int, default=250)
    p.add_argument('--train_frac', type=float, default=0.30)
    p.add_argument('--calib_frac', type=float, default=0.60)
    p.add_argument('--output_dir', type=str, default='results/diabetes_safety_gate_hetero')
    return p.parse_args()


def fit_control_surface(Xs, T, Y, n_estimators, n_jobs, seed):
    return RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=20, max_features='sqrt',
        n_jobs=n_jobs, random_state=seed + 1
    ).fit(Xs[T == 0], Y[T == 0]).predict_proba(Xs)[:, 1]


def inject_hetero_tau(Xs, pos_frac, tau_scale, seed):
    rng = np.random.RandomState(seed)
    v = rng.normal(size=Xs.shape[1])
    v /= np.linalg.norm(v)
    raw = Xs @ v
    z = (raw - raw.mean()) / (raw.std() + 1e-12)
    shift = np.quantile(z, 1.0 - pos_frac)
    tau = tau_scale * (z - shift)
    return tau, z


def estimate_policy_cate(Xs_tr, T_tr, Yobs_tr, learner, n_estimators, n_jobs, seed):
    if learner == 't':
        has1 = (T_tr == 1).sum()
        m0 = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=20,
                                   max_features='sqrt', n_jobs=n_jobs, random_state=seed
                                   ).fit(Xs_tr[T_tr == 0], Yobs_tr[T_tr == 0])
        if has1 >= 5:
            m1 = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5,
                                       max_features='sqrt', n_jobs=n_jobs, random_state=seed + 1
                                       ).fit(Xs_tr[T_tr == 1], Yobs_tr[T_tr == 1])
            return lambda Z: m1.predict(Z) - m0.predict(Z)
        learner = 'dr'

    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=20,
                               max_features='sqrt', n_jobs=n_jobs, random_state=seed + 2)
    rf.fit(np.column_stack([Xs_tr, T_tr]), Yobs_tr)
    m0_tr = rf.predict(np.column_stack([Xs_tr, np.zeros(len(Xs_tr))]))
    m1_tr = rf.predict(np.column_stack([Xs_tr, np.ones(len(Xs_tr))]))
    e_tr = LogisticRegression(max_iter=1000).fit(Xs_tr, T_tr).predict_proba(Xs_tr)[:, 1].clip(0.05, 0.95)
    psi = (m1_tr - m0_tr
           + T_tr / e_tr * (Yobs_tr - m1_tr)
           - (1 - T_tr) / (1 - e_tr) * (Yobs_tr - m0_tr))
    dr = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=50,
                               max_features='sqrt', n_jobs=n_jobs, random_state=seed + 3)
    dr.fit(Xs_tr, psi)
    return lambda Z: dr.predict(Z)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, feats = load_and_prepare(args.data_path, lookahead=args.lookahead,
                                 min_age=args.min_age, min_bmi=args.min_bmi)
    if args.subsample and args.subsample < len(df):
        df = df.sample(n=args.subsample, random_state=args.seed).reset_index(drop=True)
    X, T, Y = build_arrays(df, feats)
    n = len(Y)
    Xs = StandardScaler().fit_transform(X)
    print(f'subgroup n={n:,}  treat-rate={T.mean()*100:.1f}%  '
          f'raw avoid T1/T0={Y[T == 1].mean()*100:.1f}/{Y[T == 0].mean()*100:.1f}%', flush=True)

    print('fitting oracle control surface mu0(x) ...', flush=True)
    mu0 = fit_control_surface(Xs, T, Y, args.n_estimators, args.n_jobs, args.seed)
    if args.mu0_spread > 0:
        z0 = (mu0 - mu0.mean()) / (mu0.std() + 1e-12)
        mu0 = np.clip(0.5 + args.mu0_spread * z0, 0.02, 0.98)
    tau, _ = inject_hetero_tau(Xs, args.pos_frac, args.tau_scale, args.seed)
    mu1 = np.clip(mu0 + tau, 0.001, 0.999)
    print(f'oracle: mu0 mean={mu0.mean():.3f}  mu1 mean={mu1.mean():.3f}  '
          f'INJECTED tau mean={tau.mean():+.3f}  frac(tau>0)={(tau > 0).mean()*100:.1f}%', flush=True)

    c = float(args.c) if args.c is not None else float(np.quantile(mu0, args.null_quantile))
    print(f'prespecified threshold c={c:.3f} (= mu0 quantile {args.null_quantile:.2f})', flush=True)

    rng_b = np.random.RandomState(args.behavior_seed)
    vb = rng_b.normal(size=Xs.shape[1]); vb /= np.linalg.norm(vb)
    zb_raw = Xs @ vb
    zb = (zb_raw - zb_raw.mean()) / (zb_raw.std() + 1e-12)
    shift_b = np.quantile(zb, 1.0 - args.behavior_frac)
    e0 = np.clip(1.0 / (1.0 + np.exp(-args.behavior_scale * (zb - shift_b))), 0.15, 0.85)
    print(f'simulated behavior policy: mean e0={e0.mean():.3f}, '
          f'overlap [{e0.min():.2f},{e0.max():.2f}]', flush=True)

    q_grid = np.arange(0.05, 1.0, 0.1)
    methods = ('BH', 'hete')
    rest_test_frac = 1 - args.calib_frac / (1 - args.train_frac)

    keys = ['fdr', 'power', 'nsel', 'n_treat_sel', 'mean_out_sel',
            'mean_out_treat_sel', 'benef_treat_sel', 'harmful_dep']
    agg = {m: {q: {k: [] for k in keys} for q in q_grid} for m in methods}
    null_fracs, treatrates, treat_benef_rates = [], [], []

    for rep in range(args.n_reps):
        rng = np.random.RandomState(args.seed + rep)
        T = (rng.rand(n) < e0).astype(int)
        Y0 = mu0 + rng.randn(n) * args.sigma
        Y1 = mu1 + rng.randn(n) * args.sigma
        Yobs = np.where(T == 1, Y1, Y0)

        idx = np.arange(n)
        i_tr, i_rest = train_test_split(idx, test_size=1 - args.train_frac,
                                        stratify=T, random_state=rng)
        i_ca, i_te = train_test_split(i_rest, test_size=rest_test_frac,
                                      stratify=T[i_rest], random_state=rng)
        if args.max_test and len(i_te) > args.max_test:
            i_te = rng.choice(i_te, size=args.max_test, replace=False)

        rf = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=20,
                                   max_features='sqrt', n_jobs=args.n_jobs, random_state=rep)
        rf.fit(np.column_stack([Xs[i_tr], T[i_tr]]), Yobs[i_tr])

        def mu_hat(Z):
            m0 = rf.predict(np.column_stack([Z, np.zeros(len(Z))]))
            m1 = rf.predict(np.column_stack([Z, np.ones(len(Z))]))
            return m0, m1

        m0c, m1c = mu_hat(Xs[i_ca])
        m0t, m1t = mu_hat(Xs[i_te])

        tau_hat = estimate_policy_cate(Xs[i_tr], T[i_tr], Yobs[i_tr],
                                       args.cate_learner, args.n_estimators, args.n_jobs, rep + 11)
        a_ca = (tau_hat(Xs[i_ca]) > 0).astype(int)
        a_te = (tau_hat(Xs[i_te]) > 0).astype(int)
        treatrates.append(a_te.mean())
        treated_mask = a_te == 1
        treat_benef_rates.append(float((tau[i_te][treated_mask] > 0).mean())
                                 if treated_mask.any() else np.nan)

        prop = LogisticRegression(max_iter=1000).fit(Xs[i_tr], T[i_tr])
        pc = prop.predict_proba(Xs[i_ca])[:, 1].clip(0.05, 0.95)
        pt = prop.predict_proba(Xs[i_te])[:, 1].clip(0.05, 0.95)

        Ypi = np.where(a_te == 1, Y1[i_te], Y0[i_te])
        null_fracs.append(float(np.mean(Ypi <= c)))

        mu_ca_fac = np.where(T[i_ca] == 1, m1c, m0c)
        mu_te_pol = np.where(a_te == 1, m1t, m0t)
        calib_scores = Yobs[i_ca] - mu_ca_fac
        test_scores = c - mu_te_pol
        calib_w = (T[i_ca] == a_ca) / np.where(T[i_ca] == 1, pc, 1 - pc)
        test_w = (T[i_te] == a_te) / np.where(T[i_te] == 1, pt, 1 - pt)

        for q in q_grid:
            for m in methods:
                if m == 'BH':
                    sel, _ = weighted_BH(calib_scores, calib_w, test_scores, test_w, q)
                else:
                    _, sel, _ = weighted_CS(calib_scores, calib_w, test_scores, test_w, q=q, rand='hete')
                sel = np.array(sel, dtype=int)
                fdp, power = eval_sel(sel, Ypi, np.full(len(i_te), c))
                d = agg[m][q]
                d['fdr'].append(fdp)
                d['power'].append(power)
                d['nsel'].append(len(sel))
                if len(sel):
                    treat_sel = a_te[sel] == 1
                    d['n_treat_sel'].append(int(treat_sel.sum()))
                    d['mean_out_sel'].append(float(Ypi[sel].mean()))
                    d['mean_out_treat_sel'].append(float(Ypi[sel][treat_sel].mean())
                                                  if treat_sel.any() else np.nan)
                    d['benef_treat_sel'].append(int(np.sum(treat_sel & (Ypi[sel] > c))))
                    d['harmful_dep'].append(int(np.sum(treat_sel & (Ypi[sel] <= c))))
                else:
                    d['n_treat_sel'].append(0)
                    d['mean_out_sel'].append(np.nan)
                    d['mean_out_treat_sel'].append(np.nan)
                    d['benef_treat_sel'].append(0)
                    d['harmful_dep'].append(0)

        if (rep + 1) % 5 == 0 or rep == 0:
            print(f'  rep {rep + 1}/{args.n_reps} done  (policy treat-rate={a_te.mean()*100:.1f}%)', flush=True)

    null_frac = float(np.nanmean(null_fracs))
    treat_rate = float(np.mean(treatrates))
    treat_benef = float(np.nanmean(treat_benef_rates))

    rows = []
    for m in methods:
        for q in q_grid:
            d = agg[m][q]
            rows.append({
                'method': m, 'q': q,
                'fdr_mean': np.nanmean(d['fdr']), 'fdr_std': np.nanstd(d['fdr']),
                'power_mean': np.nanmean(d['power']),
                'n_selected_mean': np.nanmean(d['nsel']),
                'n_treated_selected_mean': np.nanmean(d['n_treat_sel']),
                'frac_selected_treated': np.nanmean(d['n_treat_sel']) / max(np.nanmean(d['nsel']), 1e-9),
                'mean_outcome_selected': np.nanmean(d['mean_out_sel']),
                'mean_outcome_treated_selected': np.nanmean(d['mean_out_treat_sel']),
                'beneficial_treated_selected_mean': np.nanmean(d['benef_treat_sel']),
                'harmful_deployed_mean': np.nanmean(d['harmful_dep']),
            })
    res = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'safety_gate_hetero_results.csv')
    res.to_csv(csv_path, index=False)

    _plot(res, q_grid, methods, c, null_frac, args.output_dir)

    print('\n' + '=' * 78)
    print('HETEROGENEOUS SAFETY-GATE SUMMARY')
    print(f'  injected frac(tau>0)={(tau > 0).mean()*100:.1f}%  policy treat-rate={treat_rate*100:.1f}%  '
          f'of-treated-truly-beneficial={treat_benef*100:.1f}%')
    print(f'  prespecified c={c:.3f}  realized null fraction={null_frac:.3f}  ({args.n_reps} reps)')
    print('=' * 78)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nSaved {csv_path}')


def _plot(res, q_grid, methods, c, null_frac, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COL = {'BH': '#0072B2', 'hete': '#D55E00'}
    LBL = {'BH': 'Weighted BH', 'hete': 'Weighted CS (hete)'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axA, axB, axC = axes

    axA.plot(q_grid, q_grid, 'k--', lw=2, alpha=0.6, zorder=1, label='Nominal FDR $q$')
    for m in methods:
        sub = res[res['method'] == m]
        axA.plot(sub['q'], sub['fdr_mean'], '-o', color=COL[m], lw=2.3, ms=6,
                 markerfacecolor='none', markeredgewidth=2, label=f'FDR ({LBL[m]})')
        axA.plot(sub['q'], sub['power_mean'], '--s', color=COL[m], lw=2.0, ms=6,
                 alpha=0.6, markerfacecolor='none', markeredgewidth=2, label=f'Power ({LBL[m]})')
    axA.set_xlabel('Nominal FDR level $q$'); axA.set_ylabel('FDR / Power')
    axA.set_ylim(-0.03, 1.03); axA.grid(alpha=0.25)
    axA.legend(fontsize=7.5, loc='upper left', ncol=1)
    axA.set_title('Graded operating curve: FDR control + power', fontsize=10)

    for m in methods:
        sub = res[res['method'] == m]
        axB.plot(sub['q'], sub['n_treated_selected_mean'], '-o', color=COL[m], lw=2.3, ms=6,
                 markerfacecolor='none', markeredgewidth=2, label=f'# treated & selected ({LBL[m]})')
    axB.axhline(0, color='gray', lw=0.8)
    axB.set_xlabel('Nominal FDR level $q$'); axB.set_ylabel('Mean # policy-treated units selected')
    axB.grid(alpha=0.25)
    axB2 = axB.twinx()
    for m in methods:
        sub = res[res['method'] == m]
        axB2.plot(sub['q'], sub['mean_outcome_treated_selected'], ':^', color=COL[m], lw=1.8, ms=6,
                  alpha=0.7, label=f'mean $Y(\\pi)$ of treated-selected ({LBL[m]})')
    axB2.axhline(c, color='green', ls=':', lw=1.6, alpha=0.8, label=f'threshold $c$={c:.2f}')
    axB2.set_ylabel('Mean true $Y(\\pi)$ of treated-selected')
    l1, lab1 = axB.get_legend_handles_labels()
    l2, lab2 = axB2.get_legend_handles_labels()
    axB.legend(l1 + l2, lab1 + lab2, fontsize=7, loc='upper left')
    axB.set_title('Gate certifies beneficial treatments', fontsize=10)

    for m in methods:
        sub = res[res['method'] == m]
        axC.plot(sub['q'], sub['harmful_deployed_mean'], '-D', color=COL[m], lw=2.3, ms=6,
                 markerfacecolor=COL[m], markeredgewidth=2, label=f'{LBL[m]}')
    axC.set_xlabel('Nominal FDR level $q$'); axC.set_ylabel('Mean # harmful treatments deployed')
    axC.grid(alpha=0.25); axC.legend(fontsize=8, loc='upper left')
    axC.set_title('Harmful deployments (treated, selected, $Y(\\pi)\\leq c$)', fontsize=10)

    fig.suptitle(f'Constructed heterogeneous benchmark (null frac={null_frac:.2f})', fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'diabetes_safety_gate_hetero.{ext}'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
