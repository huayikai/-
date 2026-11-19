"""Run native-engine checks on the server before allocating training GPUs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='3_vs_1')
    parser.add_argument('--seed', type=int, default=79)
    parser.add_argument('--episodes', type=int, default=2)
    args = parser.parse_args()
    # Report installation failures directly, before PyMARL's optional registry.
    import gfootball.env
    import main_football
    from envs.gfootball import GoogleFootballEnv

    config, _, _ = main_football.load_configuration(['--scenario=' + args.scenario])
    env = GoogleFootballEnv(seed=args.seed, **config['env_args'])
    rng = np.random.RandomState(args.seed)
    try:
        for episode in range(args.episodes):
            obs, state = env.reset()
            assert obs.shape == (env.n_agents, 115)
            assert state.shape == (env.n_agents * 115,)
            total = 0.
            for step in range(env.episode_limit):
                reward, done, info = env.step(rng.randint(env.n_actions, size=env.n_agents))
                assert np.isfinite(reward) and np.isfinite(env.get_obs()).all()
                assert np.asarray(env.get_avail_actions()).shape == (env.n_agents, env.n_actions)
                total += reward
                if done:
                    print(json.dumps({'episode': episode, 'steps': step+1,
                        'return': total, 'success': info['battle_won'],
                        'truncated': info['episode_limit']}, sort_keys=True))
                    break
            else:
                raise AssertionError('episode did not terminate within time_limit')
        versions = {}
        for package in ('gfootball', 'gym', 'numpy', 'torch', 'sacred'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = 'not installed'
        print(json.dumps({'env_info': env.get_env_info(), 'versions': versions}, indent=2))
        print('Native GRF smoke check passed. Random-policy success is not required.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
