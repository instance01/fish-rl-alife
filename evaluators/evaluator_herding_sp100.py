import sys
import glob
import pickle
import multiprocessing
from multiprocessing import Process
from collections import OrderedDict
sys.path.append('..')
from pipeline import Experiment


def load(id_, cfg_id, return_dict):
    # base_cfg_id = 'ma3_obs'  # for i10, i5 (sp600)
    # base_cfg_id = 'ma3_obs_sp100'  # for i2
    # base_cfg_id = 'ma3_obs'
    base_cfg_id = 'ma3_obs_eval'  # sp100
    base_paths = ['models']
    res = []
    two_net = 'two_net' in cfg_id
    for base_path in base_paths:
        id__ = '../' + base_path + "/%s-*-F"
        if two_net:
            id__ = '../' + base_path + "/%s-*-F-m1"
        res.extend(list(glob.glob(id__ % cfg_id)))
    counter_coop = 0.
    counter_fail = 0.
    counter_greedy = 0.
    for fname in res:
        if two_net:
            fname = fname[:-3]
        print('#############################')
        print(fname)
        for _ in range(3):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            fish_pop_hist, _ = exp.load_eval(fname, steps=5000)
            if fish_pop_hist[-1] > 0 and fish_pop_hist[-1] < 10:
                counter_coop += 1
            if fish_pop_hist[-1] == 0:
                counter_greedy += 1
            if fish_pop_hist[-1] >= 10:
                counter_fail += 1

            # TODO: For i2 !
            # if fish_pop_hist[-1] > 0 and exp.env.dead_fishes > 0:
            #     counter_coop += 1
            # print(exp.env.dead_fishes)

        total = counter_coop + counter_greedy + counter_fail
        print('c:%d' % counter_coop, 'f:%d' % counter_fail, 'g:%d' % counter_greedy, 't:%d' % total)

    print('coop', counter_coop / total)
    print('greedy', counter_greedy / total)
    print('fail', counter_fail / total)
    return_dict[id_] = (counter_coop / total, total)
    # return id_, (counter_coop / total, total)


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_i10_p150 = {
        't200': 'ma3_obs_starve_maxsteps_t200_i10_p150',
        't300': 'ma3_obs_starve_maxsteps_t300_i10_p150',
        't400': 'ma3_obs_starve_maxsteps_t400_i10_p150',
        't500': 'ma3_obs_starve_maxsteps_t500_i10_p150',
        't600': 'ma3_obs_starve_maxsteps_t600_i10_p150',
        't700': 'ma3_obs_starve_maxsteps_t700_i10_p150',
        't800': 'ma3_obs_starve_maxsteps_t800_i10_p150',
        't1000': 'ma3_obs_starve_maxsteps_t1000_i10_p150',
        't1200': 'ma3_obs_starve_maxsteps_t1200_i10_p150',
        't1500': 'ma3_obs_starve_maxsteps_t1500_i10_p150',
        't2000': 'ma3_obs_starve_maxsteps_t2000_i10_p150'
    }

    cfg_ids_i5_p150 = {
        't200': 'ma3_obs_starve_maxsteps_t200_i5_p150',
        't300': 'ma3_obs_starve_maxsteps_t300_i5_p150',
        't400': 'ma3_obs_starve_maxsteps_t400_i5_p150',
        't500': 'ma3_obs_starve_maxsteps_t500_i5_p150',
        't600': 'ma3_obs_starve_maxsteps_t600_i5_p150',
        't700': 'ma3_obs_starve_maxsteps_t700_i5_p150',
        't800': 'ma3_obs_starve_maxsteps_t800_i5_p150',
        't1000': 'ma3_obs_starve_maxsteps_t1000_i5_p150',
        't1200': 'ma3_obs_starve_maxsteps_t1200_i5_p150',
        't1500': 'ma3_obs_starve_maxsteps_t1500_i5_p150',
        't2000': 'ma3_obs_starve_maxsteps_t2000_i5_p150'
    }

    cfg_ids_i2_p150 = {
        't200': 'ma3_obs_starve_maxsteps_t200_i2_p150',
        't300': 'ma3_obs_starve_maxsteps_t300_i2_p150',
        't400': 'ma3_obs_starve_maxsteps_t400_i2_p150',
        't500': 'ma3_obs_starve_maxsteps_t500_i2_p150',
        't600': 'ma3_obs_starve_maxsteps_t600_i2_p150',
        't700': 'ma3_obs_starve_maxsteps_t700_i2_p150',
        't800': 'ma3_obs_starve_maxsteps_t800_i2_p150',
        't1000': 'ma3_obs_starve_maxsteps_t1000_i2_p150',
        't1200': 'ma3_obs_starve_maxsteps_t1200_i2_p150',
        't1500': 'ma3_obs_starve_maxsteps_t1500_i2_p150',
        't2000': 'ma3_obs_starve_maxsteps_t2000_i2_p150'
    }

    two_net = True

    kv = None
    if id_ == 'i10':
        kv = cfg_ids_i10_p150
    elif id_ == 'i5':
        kv = cfg_ids_i5_p150
    elif id_ == 'i2':
        kv = cfg_ids_i2_p150

    # What the fuck. Pool doesn't work.
    # Didn't have the time to investigate so I went for the hacky solution.
    # TODO: At some point check this out, might learn something.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for k, v in kv.items():
        p = Process(target=load, args=(k, v, return_dict))
        print('starting', k, v)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print(return_dict)
    result_kv_tuples = return_dict

    # pool = multiprocessing.Pool(processes=4)
    # multiple_results = [
    #     pool.apply_async(load, (k, v))
    #     for k, v in kv.items()
    # ]
    # result_kv_tuples = ([res.get() for res in multiple_results])
    # pool.close()
    # pool.join()

    names = []
    values = []
    # data = OrderedDict()  # tXXX -> coop percent
    # for k, percent_coop_and_total in result_kv_tuples.items():
    for k in kv:
        names.append(k)
        values.append(result_kv_tuples[k])

    print(names)
    print(values)
    if two_net:
        id_ += '_two_net'
    fname = '../pickles/' + id_ + '_herding_sp100.pickle'
    print(fname)
    with open(fname, 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
