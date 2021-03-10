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
    base_cfg_id = 'ma3_obs'
    base_paths = ['models', 'modelsDec10-14', 'modelsDec15-20']
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
        for _ in range(20):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            fish_pop_hist, _ = exp.load_eval(fname, steps=5000)
            if fish_pop_hist[-1] > 0 and fish_pop_hist[-1] < 10:
                counter_coop += 1
            if fish_pop_hist[-1] == 0:
                counter_greedy += 1
            if fish_pop_hist[-1] >= 10:
                counter_fail += 1

        total = counter_coop + counter_greedy + counter_fail
        print('c:%d' % counter_coop, 'f:%d' % counter_fail, 'g:%d' % counter_greedy, 't:%d' % total)

    print('coop', counter_coop / total)
    print('greedy', counter_greedy / total)
    print('fail', counter_fail / total)
    return_dict[id_] = (counter_coop / total, total)


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_i10_p75 = {
        't200': 'ma3_obs_starve_maxsteps_t200',
        't300': 'ma3_obs_starve_maxsteps_t300',
        't400': 'ma3_obs_starve_maxsteps_t400',
        't500': 'ma3_obs_starve_maxsteps_t500',
        't600': 'ma3_obs_starve_maxsteps_t600',
        't700': 'ma3_obs_starve_maxsteps_t700',
        't800': 'ma3_obs_starve_maxsteps_t800',
        't1000': 'ma3_obs_starve_maxsteps',
        't1200': 'ma3_obs_starve_maxsteps_t1200',
        't1500': 'ma3_obs_starve_maxsteps_t1500',
        't2000': 'ma3_obs_starve_maxsteps_t2000'
    }

    cfg_ids_i10_p150 = {
        't200': 'ma3_obs_starve_maxsteps_t200_p150',
        't300': 'ma3_obs_starve_maxsteps_t300_p150',
        't400': 'ma3_obs_starve_maxsteps_t400_p150',
        't500': 'ma3_obs_starve_maxsteps_t500_p150',
        't600': 'ma3_obs_starve_maxsteps_t600_p150',
        't700': 'ma3_obs_starve_maxsteps_t700_p150',
        't800': 'ma3_obs_starve_maxsteps_t800_p150',
        't1000': 'ma3_obs_starve_maxsteps_t1000_p150',
        't1200': 'ma3_obs_starve_maxsteps_t1200_p150',
        't1500': 'ma3_obs_starve_maxsteps_t1500_p150',
        't2000': 'ma3_obs_starve_maxsteps_t2000_p150'
    }

    cfg_ids_i5_p75 = {
        't200': 'ma3_obs_starve_maxsteps_t200_5fish',
        't300': 'ma3_obs_starve_maxsteps_t300_5fish',
        't400': 'ma3_obs_starve_maxsteps_t400_5fish',
        't500': 'ma3_obs_starve_maxsteps_t500_5fish',
        't600': 'ma3_obs_starve_maxsteps_t600_5fish',
        't700': 'ma3_obs_starve_maxsteps_t700_5fish',
        't800': 'ma3_obs_starve_maxsteps_t800_5fish',
        't1000': 'ma3_obs_starve_maxsteps_5fish',
        't1200': 'ma3_obs_starve_maxsteps_t1200_5fish',
        't1500': 'ma3_obs_starve_maxsteps_t1500_5fish',
        't2000': 'ma3_obs_starve_maxsteps_t2000_5fish'
    }

    cfg_ids_i5_p150 = {
        't200': 'ma3_obs_starve_maxsteps_t200_p150_5fish',
        't300': 'ma3_obs_starve_maxsteps_t300_p150_5fish',
        't400': 'ma3_obs_starve_maxsteps_t400_p150_5fish',
        't500': 'ma3_obs_starve_maxsteps_t500_p150_5fish',
        't600': 'ma3_obs_starve_maxsteps_t600_p150_5fish',
        't700': 'ma3_obs_starve_maxsteps_t700_p150_5fish',
        't800': 'ma3_obs_starve_maxsteps_t800_p150_5fish',
        't1000': 'ma3_obs_starve_maxsteps_t1000_p150_5fish',
        't1200': 'ma3_obs_starve_maxsteps_t1200_p150_5fish',
        't1500': 'ma3_obs_starve_maxsteps_t1500_p150_5fish',
        't2000': 'ma3_obs_starve_maxsteps_t2000_p150_5fish'
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

    cfg_ids_i2_p150_sp500 = {
        't200': 'ma3_obs_starve_maxsteps_t200_i2_p150_sp500',
        't300': 'ma3_obs_starve_maxsteps_t300_i2_p150_sp500',
        't400': 'ma3_obs_starve_maxsteps_t400_i2_p150_sp500',
        't500': 'ma3_obs_starve_maxsteps_t500_i2_p150_sp500',
        't600': 'ma3_obs_starve_maxsteps_t600_i2_p150_sp500',
        't700': 'ma3_obs_starve_maxsteps_t700_i2_p150_sp500',
        't800': 'ma3_obs_starve_maxsteps_t800_i2_p150_sp500',
        't1000': 'ma3_obs_starve_maxsteps_t1000_i2_p150_sp500',
        't1200': 'ma3_obs_starve_maxsteps_t1200_i2_p150_sp500',
        't1500': 'ma3_obs_starve_maxsteps_t1500_i2_p150_sp500',
        't2000': 'ma3_obs_starve_maxsteps_t2000_i2_p150_sp500'
    }

    two_net = True

    kv = None
    if id_ == 'i10_p75':
        kv = cfg_ids_i10_p75
    elif id_ == 'i10_p150':
        kv = cfg_ids_i10_p150
    elif id_ == 'i5_p75':
        kv = cfg_ids_i5_p75
    elif id_ == 'i5_p150':
        kv = cfg_ids_i5_p150
    elif id_ == 'i2_p150':
        kv = cfg_ids_i2_p150
    elif id_ == 'i2_p150_sp500':
        kv = cfg_ids_i2_p150_sp500

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
    fname = '../pickles/' + id_ + '_herding.pickle'
    print(fname)
    with open(fname, 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
