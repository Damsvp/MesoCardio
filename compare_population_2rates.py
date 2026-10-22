"""Compare finite-N soft-device simulations with a large-population Monte Carlo reference,
for the two-rate model of hmodel_soft_2rates.py (Gaussian attachment, uniform detachment).

Place beside hmodel_soft_2rates.py and hmodel_soft.py. The rates (const / x / custom) are chosen
at the top of hmodel_soft_2rates.py; --rates overrides that choice from the command line.
    python compare_population_2rates.py
    python compare_population_2rates.py --Ns 25 100 400 --Mref 2000 --replicas 3 --rates const

Finite-N model: nu_N = N * nu_per_head and F_ext(t) = -N*pbar(t), so the per-head force balance
has a finite N -> infinity limit (independent heads driven by a deterministic x_c(t)).
The reference is the same model with M >> N heads, averaged over --ref-replicas runs.
Diagnostics vs N: bias^2 of the replica-mean against the reference (sampling-variance terms
subtracted) and std*sqrt(N) of single runs (constant if fluctuations scale like N^-1/2).
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import hmodel_soft_2rates as H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ns', nargs='+', type=int, default=[50, 100, 200])
    parser.add_argument('--Mref', type=int, default=5000,
                        help='number of heads in the large-population Monte Carlo reference')
    parser.add_argument('--ref-replicas', type=int, default=3,
                        help='independent large-M reference realizations (averaged)')
    parser.add_argument('--replicas', type=int, default=10,
                        help='independent finite-N realizations per N')
    parser.add_argument('--nu-per-head', type=float, default=0.05)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--hold', type=float, default=10.0)
    parser.add_argument('--ramp', type=float, default=30.0)
    parser.add_argument('--tau0', type=float, default=0.88)
    parser.add_argument('--record-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--rates', choices=['const', 'x', 'custom'], default=None,
                        help='override RATES set at the top of hmodel_soft_2rates.py')
    parser.add_argument('--out', default='population_comparison_2rates')
    args = parser.parse_args()

    if (any(n <= 0 for n in args.Ns) or args.Mref <= 0 or args.replicas <= 0
            or args.ref_replicas <= 0):
        parser.error('Ns, Mref, replicas and ref-replicas must be positive')

    # Same load protocol for every population size; pbar is the load per head.
    p_lo, p_hi = 0.25 * args.tau0, 1.30 * args.tau0

    def pbar(t):
        if t < args.hold:
            return p_lo
        s = t - args.hold
        if s < args.ramp:
            return p_lo + (p_hi - p_lo) * s / args.ramp
        s -= args.ramp
        if s < args.ramp:
            return p_hi - (p_hi - p_lo) * s / args.ramp
        return p_lo

    T = args.hold + 2 * args.ramp
    model = H.HModel(rates=args.rates)
    print(f'rates = {model.rates}')

    # rates figure (check what is being simulated)
    xs = np.linspace(-40, 40, 801)
    plt.figure()
    plt.plot(xs, model.lam_plus(xs), label=r'$\lambda_+(x)$')
    plt.plot(xs, model.lam_minus(xs), label=r'$\lambda_-(x)$')
    plt.xlabel('x (nm)'); plt.ylabel(r'total rate (ms$^{-1}$)'); plt.legend()
    plt.tight_layout(); plt.savefig(args.out + '_rates.png', dpi=180); plt.close()

    n_runs = args.ref_replicas + args.replicas * len(args.Ns)
    child_seeds = iter(np.random.SeedSequence(args.seed).spawn(n_runs))

    def simulate(n, seed, label):
        start = time.time()
        out = H.run(
            model, N=n, T=T, dt=args.dt, mode='soft',
            nu=n * args.nu_per_head,
            Fext_fn=lambda t, xc: -n * pbar(t),
            rng=np.random.default_rng(seed),
            record_every=args.record_every,
        )
        print(f'{label}: N={n}, runtime={time.time()-start:.1f}s, '
              f'mean tau={out["tau_mean"]:.4f}, mean attached={out["na_mean"]:.4f}',
              flush=True)
        return out

    def stack(runs):
        return {k: np.stack([o[k] for o in runs]) for k in ('tau', 'vel', 'na', 'xc')}

    # Large-M Monte Carlo approximation to the independent-head limit (mean of several runs).
    ref_runs = [simulate(args.Mref, next(child_seeds), f'reference {r+1}/{args.ref_replicas}')
                for r in range(args.ref_replicas)]
    t_ref = ref_runs[0]['t']
    ref = stack(ref_runs)
    R0 = args.ref_replicas
    ref_m = {k: ref[k].mean(axis=0) for k in ('tau', 'vel', 'na', 'xc')}
    ref_var = {k: (ref[k].var(axis=0, ddof=1) / R0 if R0 > 1 else np.zeros_like(ref_m[k]))
               for k in ('tau', 'vel')}

    results = {}
    for n in args.Ns:
        runs = [simulate(n, next(child_seeds), f'N={n}, replica {r+1}/{args.replicas}')
                for r in range(args.replicas)]
        e = stack(runs)
        z = np.zeros_like(e['tau'][0])
        results[n] = {
            'tau_mean': e['tau'].mean(axis=0),
            'tau_std': e['tau'].std(axis=0, ddof=1) if args.replicas > 1 else z,
            'vel_mean': e['vel'].mean(axis=0),
            'vel_std': e['vel'].std(axis=0, ddof=1) if args.replicas > 1 else z,
            'na_mean': e['na'].mean(axis=0), 'xc_mean': e['xc'].mean(axis=0),
        }

    # Time-averaged discrepancies of the replica-mean against the reference mean.
    # bias2 subtracts the sampling-variance terms (replica mean and reference mean).
    print('\n    N   MSE(tau)   bias2(tau)   MSE(vel)   bias2(vel)   std(tau)*sqrt(N)   std(vel)*sqrt(N)')
    mse = {'tau': [], 'vel': []}
    bias = {'tau': [], 'vel': []}
    sdn = {'tau': [], 'vel': []}
    for n in args.Ns:
        r = results[n]
        row = []
        for key in ('tau', 'vel'):
            d2 = (r[key + '_mean'] - ref_m[key]) ** 2
            var_m = (r[key + '_std'] ** 2) / args.replicas
            m = float(np.mean(d2))
            b = float(np.mean(d2 - var_m - ref_var[key]))
            s = float(np.sqrt(np.mean(r[key + '_std'] ** 2)) * np.sqrt(n)) if args.replicas > 1 else np.nan
            mse[key].append(m); bias[key].append(b); sdn[key].append(s)
            row.append((m, b, s))
        print(f'{n:5d}  {row[0][0]:.3e}  {row[0][1]:.3e}  {row[1][0]:.3e}  {row[1][1]:.3e}'
              f'  {row[0][2]:16.4f}  {row[1][2]:16.4f}')

    # Force per head.
    plt.figure()
    plt.plot(t_ref, ref_m['tau'], label=f'large-M reference (M={args.Mref}, mean of {R0})', linewidth=2)
    for n, r in results.items():
        plt.plot(t_ref, r['tau_mean'], label=f'N={n}, mean over replicas')
        if args.replicas > 1:
            plt.fill_between(t_ref, r['tau_mean'] - r['tau_std'], r['tau_mean'] + r['tau_std'], alpha=0.15)
    plt.xlabel('time (ms)')
    plt.ylabel(r'force per head $\tau_c$ (pN)')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out + '_force.png', dpi=180)

    # Contractile-element velocity.
    plt.figure()
    plt.plot(t_ref, ref_m['vel'], label=f'large-M reference (M={args.Mref}, mean of {R0})', linewidth=2)
    for n, r in results.items():
        plt.plot(t_ref, r['vel_mean'], label=f'N={n}, mean over replicas')
        if args.replicas > 1:
            plt.fill_between(t_ref, r['vel_mean'] - r['vel_std'], r['vel_mean'] + r['vel_std'], alpha=0.15)
    plt.xlabel('time (ms)')
    plt.ylabel(r'$\dot{x}_c$ (nm/ms)')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out + '_velocity.png', dpi=180)

    # Convergence diagnostic.
    ns = np.array(args.Ns, dtype=float)
    plt.figure()
    plt.loglog(ns, mse['tau'], 'o-', label='MSE force per head')
    plt.loglog(ns, mse['vel'], 's-', label='MSE velocity')
    plt.loglog(ns, np.abs(bias['tau']), 'o--', label='|bias$^2$| force')
    plt.loglog(ns, np.abs(bias['vel']), 's--', label='|bias$^2$| velocity')
    plt.xlabel('finite population size N')
    plt.ylabel('time-averaged squared discrepancy')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out + '_convergence.png', dpi=180)

    payload = {'t': t_ref, 'ref_tau': ref_m['tau'], 'ref_vel': ref_m['vel'],
               'ref_na': ref_m['na'], 'ref_xc': ref_m['xc'], 'ref_tau_runs': ref['tau'],
               'ref_vel_runs': ref['vel'], 'Mref': args.Mref, 'Ns': np.array(args.Ns),
               'nu_per_head': args.nu_per_head, 'dt': args.dt, 'T': T, 'rates': model.rates,
               'bias2_tau': np.array(bias['tau']), 'bias2_vel': np.array(bias['vel'])}
    for n, r in results.items():
        payload[f'N{n}_tau_mean'] = r['tau_mean']
        payload[f'N{n}_tau_std'] = r['tau_std']
        payload[f'N{n}_vel_mean'] = r['vel_mean']
        payload[f'N{n}_vel_std'] = r['vel_std']
        payload[f'N{n}_na_mean'] = r['na_mean']
        payload[f'N{n}_xc_mean'] = r['xc_mean']
    np.savez(args.out + '.npz', **payload)
    print(f'Wrote {args.out}.npz and {args.out}_rates.png, _force.png, _velocity.png, _convergence.png')


if __name__ == '__main__':
    main()
