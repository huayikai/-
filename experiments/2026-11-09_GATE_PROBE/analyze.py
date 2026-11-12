"""Offline same-sample interpolation study. Requires only NumPy, no SC2."""
import argparse
import json
from pathlib import Path
import numpy as np


def elu(x):
    return np.maximum(x, 0) + np.expm1(np.minimum(x, 0))


def evaluate_file(path, repeats=20):
    with np.load(str(path), allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    pb, pd, w2, b2, target = [a[k].astype(np.float64) for k in ('pre_bm','pre_dvd','w2','b2','target')]
    gate = a['gate'].astype(np.float64)
    eligible = a['eligible'].astype(bool)
    n = len(gate)
    if not n or not all(np.isfinite(a[k]).all() for k in ('pre_bm','pre_dvd','w2','b2','target','gate','route_target')):
        raise ValueError('empty or nonfinite probe: '+str(path))
    cap = float(a['gate_max'])
    if (gate < 0).any() or (gate > cap + 1e-6).any():
        raise ValueError('invalid recorded gate')

    def predict(g):
        g = np.broadcast_to(g, gate.shape)
        pre = (1-g[:,None])*pb + g[:,None]*pd
        return (elu(pre)*w2).sum(axis=1) + b2

    reconstruction = {}
    for label, g in [('bm_q',0.),('dvd_q',1.),('mixed_q',a['applied_gate'])]:
        values = predict(g)
        observed = a[label].astype(np.float64)
        reconstruction[label] = float(np.max(np.abs(values-observed)))
        if not np.allclose(values, observed, atol=1e-4, rtol=1e-4):
            raise ValueError('reconstruction mismatch '+label+' in '+str(path))

    base_error = (predict(0.)-target)**2
    def error(g):
        return (predict(g)-target)**2
    def stats(err):
        return {'mse':float(err.mean()), 'mae':float(np.sqrt(err).mean()),
                'mse_gain_vs_bm':float((base_error-err).mean()),
                'relative_mse_gain_vs_bm':float((base_error.mean()-err.mean())/(base_error.mean()+1e-12)),
                'benefit_rate':float((err<base_error-1e-12).mean()),
                'harm_rate':float((err>base_error+1e-12).mean())}
    mean_gate = float(gate[eligible].mean()) if eligible.any() else 0.
    options = {'bm':np.zeros(n), 'dynamic':gate,
               'fixed013':np.where(eligible,min(.13,cap),0.),
               'batch_mean':np.where(eligible,mean_gate,0.),
               'route_label':a['route_target'].astype(np.float64)}
    errors = {k:error(g) for k,g in options.items()}
    # Include dynamic and both constants exactly, even off the 0.01 grid.
    best = errors['dynamic'].copy()
    best_g = gate.copy()
    for k in ('bm','fixed013','batch_mean','route_label'):
        better = errors[k] < best
        best[better], best_g[better] = errors[k][better], options[k][better]
    for g in np.linspace(0.,cap,101):
        candidate = np.where(eligible,g,0.)
        e = error(candidate)
        better = e < best
        best[better], best_g[better] = e[better], candidate[better]
    errors['oracle_grid_augmented'] = best
    rng = np.random.RandomState((int(a['seed']) ^ int(a['t_env']) ^ 0x3921) & 0xffffffff)
    shuffle_stats = []
    for _ in range(repeats):
        permuted = gate.copy()
        permuted[eligible] = rng.permutation(gate[eligible])
        e = error(permuted)
        shuffle_stats.append(stats(e))
        # A shuffled gate can lie between grid points. Include it exactly so
        # discretization cannot make a tested shuffle beat the oracle.
        better = e < best
        best[better], best_g[better] = e[better], permuted[better]
    result = {k:stats(e) for k,e in errors.items()}
    result['shuffle'] = {k:float(np.mean([s[k] for s in shuffle_stats])) for k in shuffle_stats[0]}
    result['shuffle']['mse_std_across_permutations'] = float(np.std([s['mse'] for s in shuffle_stats]))
    for label, s in result.items():
        s['mse_gap_to_oracle'] = float(s['mse'] - best.mean())
    positive = a['route_target'] > 0
    return {'path':str(path),'name':str(a['name']), 'planned_step':int(a['planned_step']),
            't_env':int(a['t_env']), 'n':n,'eligible_count':int(eligible.sum()),
            'gate_mean':float(gate.mean()), 'gate_std':float(gate.std()),
            'reconstruction_max_abs_error':reconstruction, 'methods':result,
            'route_positive_count':int(positive.sum()),
            'route_positive_label_blend_harm_rate':float((errors['route_label'][positive]>base_error[positive]+1e-12).mean()) if positive.any() else None,
            'oracle_gate_mean':float(best_g.mean()), 'grid_step':cap/100,
            'shuffle_repeats':repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='gate_probes directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shuffle-repeats', type=int, default=20)
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        parser.error('shuffle repeats must be positive')
    files = sorted(Path(args.input).rglob('*.npz'))
    if not files:
        parser.error('no probe files found')
    rows = [evaluate_file(p,args.shuffle_repeats) for p in files]
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    payload = {'note':'Pre-update replay TD fitting only. Oracle is per-sample grid plus exact recorded candidates; no true-return or policy-optimality claim. Batch means and 20 permutations are diagnostic repeats, not independent training seeds.', 'rows':rows}
    (output/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# Gate fitting diagnostics','', 'Each row is one frozen pre-update batch; MSE against its saved BM TD target.', '',
             '| Run | Stage | Actual step | n | Dynamic MSE | Fixed .13 | Batch mean | Shuffle mean | Oracle |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        m=r['methods']
        lines.append('| %s | %d | %d | %d | %.6g | %.6g | %.6g | %.6g | %.6g |' %
                     (r['name'],r['planned_step'],r['t_env'],r['n'],m['dynamic']['mse'],m['fixed013']['mse'],m['batch_mean']['mse'],m['shuffle']['mse'],m['oracle_grid_augmented']['mse']))
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Analyzed %d frozen batches. Results: %s' % (len(rows),output))


if __name__ == '__main__':
    main()
