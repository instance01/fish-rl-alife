import sys
import os
import multiprocessing
from multiprocessing import Process
import numpy as np
import scipy.stats as st
sys.path.append('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pipeline import Experiment


MAX_STEPS = 1000
# 8_obs_rand is for 100sp
# 8_obs_rand_sp200 is for 200sp
# 8_obs_rand_sp300 is for 300sp
# 6_obs_rand is for 500sp
BASE_CFG = '8_obs_rand_sp200'


def load(scenario, return_dict, wait):
    dead_fishes = []
    last_pops = []
    tot_rewards = []

    for _ in range(100):
        exp = Experiment(BASE_CFG, show_gui=False, dump_cfg=False)
        exp.env.select_fish_types(0, scenario, 0)
        if wait:
            rewards = exp.evaluate_static_wait(steps=MAX_STEPS, initial_survival_time=3000)
        else:
            rewards = exp.evaluate_static(steps=MAX_STEPS, initial_survival_time=3000)
        tot_rewards.append(sum(rewards))
        dead_fishes.append(exp.env.dead_fishes)
        last_pops.append(len(exp.env.fishes))

    return_dict[scenario] = (dead_fishes, last_pops)


def run():
    print('EVALUATING WITH MAX_STEPS ', MAX_STEPS)

    scenarios = [1, 2, 5, 10, 15, 20]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict_wait = manager.dict()
    procs = []
    for scenario in scenarios:
        p = Process(target=load, args=(scenario, return_dict, False))
        p2 = Process(target=load, args=(scenario, return_dict_wait, True))
        p.start()
        p2.start()
        procs.append(p)
        procs.append(p2)
    for p in procs:
        p.join()

    print('')
    print('STATIC')
    for scenario in scenarios:
        print('Scenario', scenario)
        dead_fishes = return_dict[scenario][0]
        last_pops = return_dict[scenario][1]

        ci = st.t.interval(0.95, len(dead_fishes) - 1, loc=np.mean(dead_fishes), scale=st.sem(dead_fishes))
        m = np.mean(dead_fishes)
        print('avg_tot_rewards_dead: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
        ci = st.t.interval(0.95, len(last_pops) - 1, loc=np.mean(last_pops), scale=st.sem(last_pops))
        m = np.mean(last_pops)
        print('avg_tot_rewards_pops: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
    print('')

    print('STATIC _ WAIT')
    for scenario in scenarios:
        print('Scenario', scenario)
        dead_fishes = return_dict_wait[scenario][0]
        last_pops = return_dict_wait[scenario][1]

        ci = st.t.interval(0.95, len(dead_fishes) - 1, loc=np.mean(dead_fishes), scale=st.sem(dead_fishes))
        m = np.mean(dead_fishes)
        print('avg_tot_rewards_dead: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
        ci = st.t.interval(0.95, len(last_pops) - 1, loc=np.mean(last_pops), scale=st.sem(last_pops))
        m = np.mean(last_pops)
        print('avg_tot_rewards_pops: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))


def main():
    run()


if __name__ == '__main__':
    main()
