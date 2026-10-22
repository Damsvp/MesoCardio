"""Compare finite-N soft-device simulations with a large-population Monte Carlo reference.

Place this file beside hmodel_soft.py, soft_ramp.py and run:
    python compare_population.py
Optional:
    python compare_population.py --Ns 25 100 400 --Mref 2000 --replicas 3

The finite-N model uses nu_N = N * nu_per_head and F_ext(t) = -N*pbar(t),
so the per-head force balance has a finite N -> infinity limit.
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import hmodel_soft as H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ns', nargs='+', type=int, default= [50, 100, 200])
    parser.add_argument('--Mref', type=int, default=5000,
                        help='number of heads in the large-population Monte Carlo reference')
    parser.add_argument('--replicas', type=int, default=10,
                        help='independent finite-N realizations per N')
    parser.add_argument('--nu-per-head', type=float, default=0.05)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--hold', type=float, default=10.0)
    parser.add_argument('--ramp', type=float, default=30.0)
    parser.add_argument('--record-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default='population_comparison')
    args = parser.parse_args()

    if any(n <= 0 for n in args.Ns) or args.Mref <= 0 or args.replicas <= 0:
        parser.error('Ns, Mref and replicas must be positive')

    # Same load protocol for every population size.
    # pbar is the external load opposing contraction, expressed per head.
    tau0 = 0.88  # same reference value as soft_ramp.py; adjust if needed
    p_lo, p_hi = 0.25 * tau0, 1.30 * tau0

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
    model = H.HModel()
    seed_seq = np.random.SeedSequence(args.seed)
    child_seeds = iter(seed_seq.spawn(args.replicas * len(args.Ns) + 1))

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

    # Large-M Monte Carlo approximation to the independent-head limit.
    ref_seed = next(child_seeds)
    ref = simulate(args.Mref, ref_seed, 'reference')
    t_ref = ref['t']

    results = {}
    for n in args.Ns:
        runs = []
        for r in range(args.replicas):
            seed = next(child_seeds)
            runs.append(simulate(n, seed, f'N={n}, replica {r+1}/{args.replicas}'))
        # The saved time grid is identical for every run.
        tau = np.stack([o['tau'] for o in runs])
        vel = np.stack([o['vel'] for o in runs])
        na = np.stack([o['na'] for o in runs])
        xc = np.stack([o['xc'] for o in runs])
        results[n] = {
            'tau_mean': tau.mean(axis=0), 'tau_std': tau.std(axis=0, ddof=1) if args.replicas > 1 else np.zeros_like(tau[0]),
            'vel_mean': vel.mean(axis=0), 'vel_std': vel.std(axis=0, ddof=1) if args.replicas > 1 else np.zeros_like(vel[0]),
            'na_mean': na.mean(axis=0), 'xc_mean': xc.mean(axis=0),
        }

    # Reference is one large-M realization. The finite-N ensemble means are compared to it.
    # Use the common recorded time grid; report time-averaged trajectory discrepancies.
    print('\nTime-averaged squared discrepancies against the large-M reference:')
    for n, r in results.items():
        e_tau = np.mean((r['tau_mean'] - ref['tau'])**2)
        e_vel = np.mean((r['vel_mean'] - ref['vel'])**2)
        print(f'N={n:5d}: MSE(tau)={e_tau:.6g}, MSE(velocity)={e_vel:.6g}')

    # Force per head.
    plt.figure()
    plt.plot(t_ref, ref['tau'], label=f'large-M reference (M={args.Mref})', linewidth=2)
    for n, r in results.items():
        plt.plot(t_ref, r['tau_mean'], label=f'N={n}, mean over replicas')
        if args.replicas > 1:
            plt.fill_between(t_ref, r['tau_mean']-r['tau_std'], r['tau_mean']+r['tau_std'], alpha=0.15)
    plt.xlabel('time (ms)')
    plt.ylabel(r'force per head $\tau_c$ (pN)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out + '_force.png', dpi=180)

    # Contractile-element velocity.
    plt.figure()
    plt.plot(t_ref, ref['vel'], label=f'large-M reference (M={args.Mref})', linewidth=2)
    for n, r in results.items():
        plt.plot(t_ref, r['vel_mean'], label=f'N={n}, mean over replicas')
        if args.replicas > 1:
            plt.fill_between(t_ref, r['vel_mean']-r['vel_std'], r['vel_mean']+r['vel_std'], alpha=0.15)
    plt.xlabel('time (ms)')
    plt.ylabel(r'$\dot{x}_c$ (nm/ms)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out + '_velocity.png', dpi=180)

    # Convergence diagnostic: discrepancies against the reference as N increases.
    ns = np.array(args.Ns, dtype=float)
    mse_tau = np.array([np.mean((results[n]['tau_mean'] - ref['tau'])**2) for n in args.Ns])
    mse_vel = np.array([np.mean((results[n]['vel_mean'] - ref['vel'])**2) for n in args.Ns])
    plt.figure()
    plt.loglog(ns, mse_tau, 'o-', label='force per head')
    plt.loglog(ns, mse_vel, 's-', label='velocity')
    plt.xlabel('finite population size N')
    plt.ylabel('time-averaged squared discrepancy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out + '_convergence.png', dpi=180)

    payload = {'t': t_ref, 'ref_tau': ref['tau'], 'ref_vel': ref['vel'],
               'ref_na': ref['na'], 'ref_xc': ref['xc'], 'Mref': args.Mref,
               'Ns': np.array(args.Ns), 'nu_per_head': args.nu_per_head,
               'dt': args.dt, 'T': T}
    for n, r in results.items():
        payload[f'N{n}_tau_mean'] = r['tau_mean']
        payload[f'N{n}_tau_std'] = r['tau_std']
        payload[f'N{n}_vel_mean'] = r['vel_mean']
        payload[f'N{n}_vel_std'] = r['vel_std']
        payload[f'N{n}_na_mean'] = r['na_mean']
        payload[f'N{n}_xc_mean'] = r['xc_mean']
    np.savez(args.out + '.npz', **payload)
    print(f'Wrote {args.out}.npz and {args.out}_force.png, _velocity.png, _convergence.png')


if __name__ == '__main__':
    main()
