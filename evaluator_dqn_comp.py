import os
import numpy as np
import scipy.stats as st
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load(fnames):
    from pipeline import Experiment

    scenarios = [1, 2, 5, 10]
    for scenario in scenarios:
        print('Scenario', scenario)
        tot_rewards = []
        for fname in fnames:
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
        # 'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-30316607-model-4'
        'models/8_obs-smaragd.cip.ifi.lmu.de-20.12.26-15:47:34-21372910-model-6',
        # 'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:08-42810204-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-64795473-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-30316607-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-26456983-model-6',
        'models/8_obs-jachen.cip.ifi.lmu.de-20.12.26-15:44:37-70807588-model-F',
        'models/8_obs-jachen.cip.ifi.lmu.de-20.12.26-15:44:33-59934980-model-F',
        'models/8_obs-itz.cip.ifi.lmu.de-20.12.26-15:44:33-57979089-model-F',
        'models/8_obs-itz.cip.ifi.lmu.de-20.12.26-15:44:29-47874541-model-F',
        # 'models/8_obs-diamant.cip.ifi.lmu.de-20.12.26-15:44:55-1934685-model-6',
        'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-14082254-model-F',
        'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-32890563-model-F',
        'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-80420989-model-F'
    ]
    load(models)


if __name__ == '__main__':
    main()
