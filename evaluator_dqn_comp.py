import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import scipy.stats as st


def load(fname):
    from pipeline import Experiment

    scenarios = [1, 2, 5, 10]
    for scenario in scenarios:
        tot_rewards = []
        print('Scenario', scenario)
        for _ in range(100):
            exp = Experiment('8_obs', show_gui=False, dump_cfg=False)
            exp.env.select_fish_types(0, scenario, 0)
            _, rewards = exp.load_eval(fname, steps=500, initial_survival_time=3000)
            tot_rewards.append(sum(rewards))

        ci = (0, 0)
        if len(tot_rewards) > 1:
            ci = st.t.interval(0.95, len(tot_rewards) - 1, loc=np.mean(tot_rewards), scale=st.sem(tot_rewards))
        m = np.mean(tot_rewards)
        print('avg_tot_rewards: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))


def main():
    models = [
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-30316607-model-4'
    ]
    load(models[0])


if __name__ == '__main__':
    main()
