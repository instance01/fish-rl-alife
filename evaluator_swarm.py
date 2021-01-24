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
# BASE_CFG = '8_obs_10obs'
BASE_CFG = '8_obs_rand_swarm_i2'
# BASE_CFG = '8_obs_rand_turnaway'  # Ok, seems to get similar performance as dqn comp.. Hmm..


def load(fname, scenario, ret_dead_fishes, ret_last_pops):
    static = fname == 'static'
    static_wait = fname == 'static_wait'

    temp = []
    for _ in range(100):
        exp = Experiment(BASE_CFG, show_gui=False, dump_cfg=False)
        # exp.env.select_fish_types(0, scenario, 0)
        if static_wait:
            _ = exp.evaluate_static_wait(steps=MAX_STEPS, initial_survival_time=3000)
        elif static:
            _ = exp.evaluate_static(steps=MAX_STEPS, initial_survival_time=3000)
        else:
            _, _ = exp.load_eval(fname, steps=MAX_STEPS, initial_survival_time=3000)
        temp.append(exp.env.dead_fishes)
        ret_dead_fishes.append(exp.env.dead_fishes)
        ret_last_pops.append(len(exp.env.fishes))
    print(fname, temp)


def run(fnames):
    print('EVALUATING WITH MAX_STEPS ', MAX_STEPS)

    # scenarios = [1, 2]
    scenarios = [0]  # TODO: No such thing here..
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

        # print(dead_fishes)
        ci = st.t.interval(0.95, len(dead_fishes) - 1, loc=np.mean(dead_fishes), scale=st.sem(dead_fishes))
        m = np.mean(dead_fishes)
        print('avg_tot_rewards_dead: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))
        ci = st.t.interval(0.95, len(last_pops) - 1, loc=np.mean(last_pops), scale=st.sem(last_pops))
        m = np.mean(last_pops)
        print('avg_tot_rewards_pops: %f +- %f' % (round(m, 2), round(m - ci[0], 2)))


def main():
    models = [
        # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:37-1354441-model-F',
        # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:37-19127995-model-F',
        # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:37-47505142-model-F',
        # # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:37-91605023-model-F',
        # # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:38-11473404-model-F',
        # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:38-13927440-model-F',
        # # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:38-50292287-model-F',
        # 'models/8_obs_10obs-danburit.cip.ifi.lmu.de-21.01.13-11:24:38-76730005-model-F',
        # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:36-39319895-model-F',
        # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:36-61244085-model-F',
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:36-93014761-model-F',  # a
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:36-99900334-model-F',
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:38-37102470-model-F',  # b
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:39-70995972-model-F',
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:40-57687272-model-F',
        # # 'models/8_obs_10obs-diamant.cip.ifi.lmu.de-21.01.13-11:24:40-93032615-model-F',
        # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:29-40997240-model-F',
        # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:29-4385117-model-F',
        # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:29-50305939-model-F',
        # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:29-55462541-model-F',
        # # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:29-78948439-model-F',
        # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:30-26907477-model-F'
        # # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:30-29688266-model-F',
        # # 'models/8_obs_10obs-dioptas.cip.ifi.lmu.de-21.01.13-11:24:30-43295747-model-F'

        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-24684580-model-5',
        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-25712339-model-5',
        # # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-39023314-model-5',
        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-58201244-model-5',
        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-59344279-model-5',
        # # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-75461112-model-5',
        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:02-97307731-model-5',
        # 'models/8_obs_10obs_2-dioptas.cip.ifi.lmu.de-21.01.14-18:08:03-30881074-model-5'

        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-12718812-model-5',
        # # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-16749783-model-5',
        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-17761534-model-5',
        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-19538209-model-5',
        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-34548996-model-5',
        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-61586416-model-5',
        # # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-64396373-model-5',
        # 'models/8_obs_10obs_3-diamant.cip.ifi.lmu.de-21.01.14-18:08:06-75594785-model-5'

        # # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:09-10706060-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:09-13632977-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:09-73351089-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:09-96925094-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:10-34557659-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:10-39121644-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:10-63863466-model-5',
        # 'models/8_obs_10obs_4-danburit.cip.ifi.lmu.de-21.01.14-18:08:11-50013430-model-5'


        # # TODO BEST SO FAR
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:21-2907918-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:21-33770853-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:21-74952212-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:21-93878827-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:22-25816483-model-4',
        # # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:22-59903347-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:22-75121666-model-4',
        # 'models/8_obs_swarm_6-heliodor.cip.ifi.lmu.de-21.01.15-19:26:22-94419903-model-4'

        # TODO EVEN BETTER ! 24+5 ~= 29! log_7587.txt
        # 'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-127351-model-3',
        # 'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-33111507-model-3',
        # 'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-71672721-model-3',
        # 'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-76676435-model-3',
        # 'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-96924183-model-3'

        # TODO BEST ! 33+7 ~= 30! log_8567.log
        # Same as above but with 4k model and not 3k model.
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-127351-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-22008984-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-33111507-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-71672721-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-76676435-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-76689731-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-87702858-model-4',
        'models/8_obs_swarm_7-leucit.cip.ifi.lmu.de-21.01.16-08:03:01-96924183-model-4'

    ]
    print('PPO')
    run(models)
    print('STATIC')
    run(['static'])
    print('STATIC WAIT')
    run(['static_wait'])


if __name__ == '__main__':
    main()
