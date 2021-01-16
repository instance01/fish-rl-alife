import os
import multiprocessing
from multiprocessing import Process
import numpy as np
import scipy.stats as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pipeline import Experiment


MAX_STEPS = 1000
# 8_obs_rand is for 100sp
# 8_obs_rand_sp200 is for 200sp
# 8_obs_rand_sp300 is for 300sp
# 6_obs_rand is for 500sp
# BASE_CFG = '8_obs_rand_sp300'
BASE_CFG = '8_obs_rand'


def load(fname, scenario, ret_dead_fishes, ret_last_pops):
    xx = []
    for _ in range(100):
        exp = Experiment(BASE_CFG, show_gui=False, dump_cfg=False)
        exp.env.select_fish_types(0, scenario, 0)
        _, _ = exp.load_eval(fname, steps=MAX_STEPS, initial_survival_time=3000)
        ret_dead_fishes.append(exp.env.dead_fishes)
        ret_last_pops.append(len(exp.env.fishes))

        xx.append(exp.env.dead_fishes)
    print(fname, xx)


def run(fnames):
    print('EVALUATING WITH MAX_STEPS ', MAX_STEPS)

    scenarios = [1, 2]
    for scenario in scenarios:
        print('Scenario', scenario)

        manager = multiprocessing.Manager()
        dead_fishes = manager.list()
        last_pops = manager.list()
        procs = []
        for fname in fnames:
            p = Process(target=load, args=(fname, scenario, dead_fishes, last_pops))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        print(dead_fishes)
        ci = st.t.interval(0.95, len(dead_fishes) - 1, loc=np.mean(dead_fishes), scale=st.sem(dead_fishes))
        m = np.mean(dead_fishes)
        print('avg_tot_rewards_dead: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
        ci = st.t.interval(0.95, len(last_pops) - 1, loc=np.mean(last_pops), scale=st.sem(last_pops))
        m = np.mean(last_pops)
        print('avg_tot_rewards_pops: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))


def main():
    models = [
        'models/8_obs-smaragd.cip.ifi.lmu.de-20.12.26-15:47:34-21372910-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:08-42810204-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-64795473-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-30316607-model-6',
        'models/8_obs-opal.cip.ifi.lmu.de-20.12.26-15:48:05-26456983-model-6',
        'models/8_obs-jachen.cip.ifi.lmu.de-20.12.26-15:44:37-70807588-model-F',
        'models/8_obs-jachen.cip.ifi.lmu.de-20.12.26-15:44:33-59934980-model-F',
        'models/8_obs-itz.cip.ifi.lmu.de-20.12.26-15:44:33-57979089-model-F',
        'models/8_obs-itz.cip.ifi.lmu.de-20.12.26-15:44:29-47874541-model-F',
        'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-32890563-model-F'

        # 'models/8_obs-diamant.cip.ifi.lmu.de-20.12.26-15:44:55-1934685-model-6',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-80420989-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-14082254-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.27-13:09:56-95687101-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.28-10:22:10-55570274-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.28-10:27:28-28087104-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.28-10:34:38-56227544-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.28-12:00:38-8934372-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.29-02:44:34-99012202-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.29-04:35:15-69783370-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.29-05:39:42-26822818-model-F',
        # 'models/8_obs-anlauter.cip.ifi.lmu.de-20.12.29-06:36:43-49685912-model-F'
    ]
    run(models)


if __name__ == '__main__':
    main()
