import sys
import os
import multiprocessing
from multiprocessing import Process
import numpy as np
import scipy.stats as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pipeline import Experiment


MAX_STEPS = 3000
# 8_obs_rand is for 100sp
# 8_obs_rand_sp200
# 8_obs_rand_sp300
# 8_obs_rand_sp400
# 8_obs_rand_sp500
# 8_obs_rand_sp600
#BASE_CFG = '8_obs_rand_sp600'
BASE_CFG = sys.argv[1]


def load(fname, scenario, ret_steps_til_first_kill, ret_fish_pop):
    for _ in range(10):
        exp = Experiment(BASE_CFG, show_gui=False, dump_cfg=False)
        exp.env.select_fish_types(0, scenario, 0)
        _, _ = exp.load_eval(fname, steps=MAX_STEPS, initial_survival_time=3000)
        ret_steps_til_first_kill.append(exp.env.steps_til_first_kill)
        ret_fish_pop.append(len(exp.env.fishes))


def run(fnames):
    print('EVALUATING WITH MAX_STEPS ', MAX_STEPS)
    print('EVALUATING WITH BASECFG', BASE_CFG)

    scenarios = [1]
    for scenario in scenarios:
        print('Scenario', scenario)

        manager = multiprocessing.Manager()
        steps_til_first_kill = manager.list()
        fish_pop = manager.list()
        procs = []
        for fname in fnames:
            p = Process(target=load, args=(fname, scenario, steps_til_first_kill, fish_pop))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        ci = st.t.interval(0.95, len(fish_pop) - 1, loc=np.mean(fish_pop), scale=st.sem(fish_pop))
        m = np.mean(fish_pop)
        print('avg_tot_rewards_pops: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
        ci = st.t.interval(0.95, len(steps_til_first_kill) - 1, loc=np.mean(steps_til_first_kill), scale=st.sem(steps_til_first_kill))
        m = np.mean(steps_til_first_kill)
        print('avg_tot_rewards_kill: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))


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
    ]
    run(models)


if __name__ == '__main__':
    main()
