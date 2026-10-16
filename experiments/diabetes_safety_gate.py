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
from ConfSelect import weighted_BH, eval_sel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=os.environ.get('DIABETES_DATA_PATH'),
                   required='DIABETES_DATA_PATH' not in os.environ,
                   help='path to the EHR csv (or set DIABETES_DATA_PATH)')
    p.add_argument('--lookahead', type=int, default=3)
    p.add_argument('--min_age', type=float, default=60)
    p.add_argument('--min_bmi', type=float, default=30)
    p.add_argument('--treat_frac', type=float, default=0.10)
    p.add_argument('--sigma', type=float, default=0.10)
    p.add_argument('--n_reps', type=int, default=20)
    p.add_argument('--seed', type=int, default=500)
    p.add_argument('--n_estimators', type=int, default=200)
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--train_frac', type=float, default=0.30)
    p.add_argument('--calib_frac', type=float, default=0.60)
    p.add_argument('--output_dir', type=str, default='results/diabetes_safety_gate')
    return p.parse_args()


def fit_oracle(Xs, T, Y, n_estimators, n_jobs, seed):
    from econml.dml import CausalForestDML
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=20,
                                      n_jobs=n_jobs, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=20,
                                       n_jobs=n_jobs, random_state=seed),
        discrete_treatment=True, n_estimators=300, min_samples_leaf=20,
        random_state=seed, cv=3)
    cf.fit(Y.astype(float), T, X=Xs)
    tau = cf.effect(Xs)
    mu0 = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt',
                                 n_jobs=n_jobs, random_state=seed + 1
                                 ).fit(Xs[T == 0], Y[T == 0]).predict_proba(Xs)[:, 1]
    mu1 = np.clip(mu0 + tau, 0.001, 0.999)
    return mu0, mu1, tau


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, feats = load_and_prepare(args.data_path, lookahead=args.lookahead,
                                 min_age=args.min_age, min_bmi=args.min_bmi)
    X, T, Y = build_arrays(df, feats)
    n = len(Y)
    Xs = StandardScaler().fit_transform(X)
    onset = 1 - Y
    print(f'subgroup n={n:,}  treat-rate={T.mean()*100:.1f}%  '
          f'raw avoid T1/T0={Y[T == 1].mean()*100:.1f}/{Y[T == 0].mean()*100:.1f}%', flush=True)

    print('fitting causal-forest oracle ...', flush=True)
    mu0, mu1, tau = fit_oracle(Xs, T, Y, args.n_estimators, args.n_jobs, args.seed)
    print(f'oracle: mu0 mean={mu0.mean():.3f}  mu1 mean={mu1.mean():.3f}  '
          f'tau mean={tau.mean():+.3f}  frac(tau>0)={(tau > 0).mean()*100:.1f}%', flush=True)

    q_grid = np.arange(0.05, 1.0, 0.1)
    rest_test_frac = 1 - args.calib_frac / (1 - args.train_frac)
    agg = {q: {'fdr': [], 'power': [], 'nsel': [], 'htx_dep': [], 'htx_avail': []} for q in q_grid}
    floors, treatrates = [], []

    for rep in range(args.n_reps):
        rng = np.random.RandomState(args.seed + rep)
        Y0 = mu0 + rng.randn(n) * args.sigma
        Y1 = mu1 + rng.randn(n) * args.sigma
        Yobs = np.where(T == 1, Y1, Y0)

        idx = np.arange(n)
        i_tr, i_rest = train_test_split(idx, test_size=1 - args.train_frac,
                                        stratify=T, random_state=rng)
        i_ca, i_te = train_test_split(i_rest, test_size=rest_test_frac,
                                      stratify=T[i_rest], random_state=rng)

        rf = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=20,
                                   max_features='sqrt', n_jobs=args.n_jobs, random_state=rep)
        rf.fit(np.column_stack([Xs[i_tr], T[i_tr]]), Yobs[i_tr])

        def mu_hat(Z):
            m0 = rf.predict(np.column_stack([Z, np.zeros(len(Z))]))
            m1 = rf.predict(np.column_stack([Z, np.ones(len(Z))]))
            return m0, m1

        m0c, m1c = mu_hat(Xs[i_ca])
        m0t, m1t = mu_hat(Xs[i_te])

        risk = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_leaf=20,
                                      max_features='sqrt', n_jobs=args.n_jobs,
                                      random_state=rep + 7).fit(Xs[i_tr], onset[i_tr])
        r_tr = risk.predict_proba(Xs[i_tr])[:, 1]
        thr = np.quantile(r_tr, 1 - args.treat_frac)
        a_ca = (risk.predict_proba(Xs[i_ca])[:, 1] >= thr).astype(int)
        a_te = (risk.predict_proba(Xs[i_te])[:, 1] >= thr).astype(int)
        treatrates.append(a_te.mean())

        prop = LogisticRegression(max_iter=1000).fit(Xs[i_tr], T[i_tr])
        pc = prop.predict_proba(Xs[i_ca])[:, 1].clip(0.05, 0.95)
        pt = prop.predict_proba(Xs[i_te])[:, 1].clip(0.05, 0.95)

        Ypi = np.where(a_te == 1, Y1[i_te], Y0[i_te])
        C = np.median(Ypi)
        floors.append(float(np.mean(Ypi <= C)))

        mu_ca_fac = np.where(T[i_ca] == 1, m1c, m0c)
        mu_te_pol = np.where(a_te == 1, m1t, m0t)
        calib_scores = Yobs[i_ca] - mu_ca_fac
        test_scores = C - mu_te_pol
        calib_w = (T[i_ca] == a_ca) / np.where(T[i_ca] == 1, pc, 1 - pc)
        test_w = (T[i_te] == a_te) / np.where(T[i_te] == 1, pt, 1 - pt)
        c_vec = np.full(len(i_te), C)

        for q in q_grid:
            sel, _ = weighted_BH(calib_scores, calib_w, test_scores, test_w, q)
            sel = np.array(sel, dtype=int)
            fdp, power = eval_sel(sel, Ypi, c_vec)
            agg[q]['fdr'].append(fdp)
            agg[q]['power'].append(power)
            agg[q]['nsel'].append(len(sel))
            agg[q]['htx_dep'].append(int(a_te[sel].sum()) if len(sel) else 0)
            agg[q]['htx_avail'].append(int(a_te.sum()))

        if (rep + 1) % 5 == 0 or rep == 0:
            print(f'  rep {rep + 1}/{args.n_reps} done', flush=True)

    rows = []
    for q in q_grid:
        rows.append({
            'q': q,
            'fdr_mean': np.mean(agg[q]['fdr']), 'fdr_std': np.std(agg[q]['fdr']),
            'power_mean': np.mean(agg[q]['power']),
            'n_selected_mean': np.mean(agg[q]['nsel']),
            'harmful_tx_deployed_mean': np.mean(agg[q]['htx_dep']),
            'harmful_tx_available_mean': np.mean(agg[q]['htx_avail']),
        })
    res = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'safety_gate_results.csv')
    res.to_csv(csv_path, index=False)

    floor = float(np.mean(floors))
    _plot(res, q_grid, floor, args.output_dir)

    print('\n' + '=' * 72)
    print(f'SAFETY-GATE SUMMARY  (policy treat-rate={np.mean(treatrates)*100:.1f}%, '
          f'floor={floor:.3f}, {args.n_reps} reps)')
    print('=' * 72)
    print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nSaved {csv_path}')


def _plot(res, q_grid, floor, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BLUE, VERM = '#0072B2', '#D55E00'
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.plot(q_grid, q_grid, 'k--', lw=2, alpha=0.7, zorder=1, label='Nominal FDR $q$')
    axL.axhline(floor, color='gray', ls=':', lw=2, alpha=0.8, label=f'Select-all floor ({floor:.2f})')
    axL.plot(q_grid, res['fdr_mean'], '-o', color=BLUE, lw=2.5, ms=7,
             markerfacecolor='none', markeredgewidth=2, label='Empirical FDR')
    axL.plot(q_grid, res['power_mean'], '-s', color=VERM, lw=2.5, ms=7,
             markerfacecolor='none', markeredgewidth=2, alpha=0.7, label='Power')
    axL.set_xlabel('Nominal FDR level $q$'); axL.set_ylabel('FDR / Power')
    axL.set_ylim(-0.03, 1.03); axL.legend(fontsize=9, loc='upper left'); axL.grid(alpha=0.25)

    avail = res['harmful_tx_available_mean'].iloc[-1]
    frac = 100 * res['harmful_tx_deployed_mean'] / np.maximum(avail, 1)
    axR.plot(q_grid, frac, '-D', color=VERM, lw=2.5, ms=7,
             markerfacecolor=VERM, markeredgewidth=2)
    axR.set_xlabel('Nominal FDR level $q$')
    axR.set_ylabel('Harmful AI treatments deployed (%)')
    axR.set_ylim(-2, 102); axR.grid(alpha=0.25)
    axR.set_title('Safety gate blocks harmful deployments', fontsize=10)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'diabetes_safety_gate.{ext}'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
