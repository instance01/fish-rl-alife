import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import scipy.stats as st


def load(fname):
    from pipeline import Experiment

    tot_rewards = []
    scenarios = [1, 2, 5, 10]
    for scenario in scenarios:
        print('Scenario', scenario)
        for _ in range(100):
            exp = Experiment('8_obs', show_gui=False, dump_cfg=False)
            exp.env.select_fish_types(0, scenario, 0)
            exp.load_eval(fname, steps=500, initial_survival_time=3000)
            tot_rewards.append(10. * exp.env.dead_fishes)

        ci = (0, 0)
        if len(tot_rewards) > 1:
            ci = st.t.interval(0.95, len(tot_rewards) - 1, loc=np.mean(tot_rewards), scale=st.sem(tot_rewards))
        m = np.mean(tot_rewards)
        print('avg_tot_rewards: %d +- %d' % (m, m - ci[0]))


def main():
    models = [
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-30316607-model-4'
    ]
    load(models[0])


if __name__ == '__main__':
    main()
