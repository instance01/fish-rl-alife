import os
import sys
import glob
import pickle
import multiprocessing
from multiprocessing import Process
import numpy as np
import scipy.stats as st
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('..')
from pipeline import Experiment


def load(id_, cfg_id, base_cfg_id, return_dict):
    return_dict[id_] = (-1, (0, 0))
    # base_paths = ['models', 'modelsDec10-14']
    base_paths = ['models']
    res = []
    for base_path in base_paths:
        ids_ = [
            '../' + base_path + "/%s-*-F-m1",
            '../' + base_path + "/%s-*-6-m1"
        ]
        for id__ in ids_:
            print(id__ % cfg_id)
            res.extend(list(glob.glob(id__ % cfg_id)))
    print('####################')
    print(id_)
    print(res)

    coop_ratios = []
    for fname in res:
        fname = fname[:-3]
        for _ in range(20):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            exp.load_eval(fname, steps=3000, initial_survival_time=3000)
            if exp.env.dead_fishes != 0:
                print(exp.env.coop_kills, exp.env.dead_fishes)
                coop_ratios.append(exp.env.coop_kills / exp.env.dead_fishes)
            else:
                print('####### NO dead fishes! #######')

    if coop_ratios:
        ci = st.t.interval(0.95, len(coop_ratios)-1, loc=np.mean(coop_ratios), scale=st.sem(coop_ratios))
        print('avg_coop_ratio:%d' % np.mean(coop_ratios), coop_ratios)
        return_dict[id_] = (np.mean(coop_ratios), ci)
    else:
        return_dict[id_] = (0, (0, 0))


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_i5 = {
        't2000_i5_r4_s03': ['ma9_t2000_i5_p150_r4_s03_stun_ext', 'ma9_t3000_i5_p150_r4_s03_stun_ext'],
        't2000_i5_r4_s04': ['ma9_t2000_i5_p150_r4_s04_stun_ext', 'ma9_t3000_i5_p150_r4_s04_stun_ext'],
        't2000_i5_r4_s05': ['ma9_t2000_i5_p150_r4_s05_stun_ext', 'ma9_t3000_i5_p150_r4_s05_stun_ext'],
        't2000_i5_r6_s03': ['ma9_t2000_i5_p150_r6_s03_stun_ext', 'ma9_t3000_i5_p150_r6_s03_stun_ext'],
        't2000_i5_r6_s04': ['ma9_t2000_i5_p150_r6_s04_stun_ext', 'ma9_t3000_i5_p150_r6_s04_stun_ext'],
        't2000_i5_r6_s05': ['ma9_t2000_i5_p150_r6_s05_stun_ext', 'ma9_t3000_i5_p150_r6_s05_stun_ext'],
        't2000_i5_r10_s03': ['ma9_t2000_i5_p150_r10_s03_stun_ext', 'ma9_t3000_i5_p150_r10_s03_stun_ext'],
        't2000_i5_r10_s04': ['ma9_t2000_i5_p150_r10_s04_stun_ext', 'ma9_t3000_i5_p150_r10_s04_stun_ext'],
        't2000_i5_r10_s05': ['ma9_t2000_i5_p150_r10_s05_stun_ext', 'ma9_t3000_i5_p150_r10_s05_stun_ext'],  # was failing

        't1500_i5_r4_s03': ['ma9_t1500_i5_p150_r4_s03_stun_ext', 'ma9_t3000_i5_p150_r4_s03_stun_ext'],
        't1500_i5_r4_s04': ['ma9_t1500_i5_p150_r4_s04_stun_ext', 'ma9_t3000_i5_p150_r4_s04_stun_ext'],
        't1500_i5_r4_s05': ['ma9_t1500_i5_p150_r4_s05_stun_ext', 'ma9_t3000_i5_p150_r4_s05_stun_ext'],
        't1500_i5_r6_s03': ['ma9_t1500_i5_p150_r6_s03_stun_ext', 'ma9_t3000_i5_p150_r6_s03_stun_ext'],
        't1500_i5_r6_s04': ['ma9_t1500_i5_p150_r6_s04_stun_ext', 'ma9_t3000_i5_p150_r6_s04_stun_ext'],
        't1500_i5_r6_s05': ['ma9_t1500_i5_p150_r6_s05_stun_ext', 'ma9_t3000_i5_p150_r6_s05_stun_ext'],
        't1500_i5_r10_s03': ['ma9_t1500_i5_p150_r10_s03_stun_ext', 'ma9_t3000_i5_p150_r10_s03_stun_ext'],
        't1500_i5_r10_s04': ['ma9_t1500_i5_p150_r10_s04_stun_ext', 'ma9_t3000_i5_p150_r10_s04_stun_ext'],
        't1500_i5_r10_s05': ['ma9_t1500_i5_p150_r10_s05_stun_ext', 'ma9_t3000_i5_p150_r10_s05_stun_ext'],

        't1000_i5_r4_s03': ['ma9_t1000_i5_p150_r4_s03_stun_ext', 'ma9_t3000_i5_p150_r4_s03_stun_ext'],
        't1000_i5_r4_s04': ['ma9_t1000_i5_p150_r4_s04_stun_ext', 'ma9_t3000_i5_p150_r4_s04_stun_ext'],
        't1000_i5_r4_s05': ['ma9_t1000_i5_p150_r4_s05_stun_ext', 'ma9_t3000_i5_p150_r4_s05_stun_ext'],
        't1000_i5_r6_s03': ['ma9_t1000_i5_p150_r6_s03_stun_ext', 'ma9_t3000_i5_p150_r6_s03_stun_ext'],
        't1000_i5_r6_s04': ['ma9_t1000_i5_p150_r6_s04_stun_ext', 'ma9_t3000_i5_p150_r6_s04_stun_ext'],
        't1000_i5_r6_s05': ['ma9_t1000_i5_p150_r6_s05_stun_ext', 'ma9_t3000_i5_p150_r6_s05_stun_ext'],
        't1000_i5_r10_s03': ['ma9_t1000_i5_p150_r10_s03_stun_ext', 'ma9_t3000_i5_p150_r10_s03_stun_ext'],
        't1000_i5_r10_s04': ['ma9_t1000_i5_p150_r10_s04_stun_ext', 'ma9_t3000_i5_p150_r10_s04_stun_ext'],
        't1000_i5_r10_s05': ['ma9_t1000_i5_p150_r10_s05_stun_ext', 'ma9_t3000_i5_p150_r10_s05_stun_ext'],

        't500_i5_r4_s03': ['ma9_t500_i5_p150_r4_s03_stun_ext', 'ma9_t3000_i5_p150_r4_s03_stun_ext'],
        't500_i5_r4_s04': ['ma9_t500_i5_p150_r4_s04_stun_ext', 'ma9_t3000_i5_p150_r4_s04_stun_ext'],
        't500_i5_r4_s05': ['ma9_t500_i5_p150_r4_s05_stun_ext', 'ma9_t3000_i5_p150_r4_s05_stun_ext'],
        't500_i5_r6_s03': ['ma9_t500_i5_p150_r6_s03_stun_ext', 'ma9_t3000_i5_p150_r6_s03_stun_ext'],
        't500_i5_r6_s04': ['ma9_t500_i5_p150_r6_s04_stun_ext', 'ma9_t3000_i5_p150_r6_s04_stun_ext'],
        't500_i5_r6_s05': ['ma9_t500_i5_p150_r6_s05_stun_ext', 'ma9_t3000_i5_p150_r6_s05_stun_ext'],
        't500_i5_r10_s03': ['ma9_t500_i5_p150_r10_s03_stun_ext', 'ma9_t3000_i5_p150_r10_s03_stun_ext'],
        't500_i5_r10_s04': ['ma9_t500_i5_p150_r10_s04_stun_ext', 'ma9_t3000_i5_p150_r10_s04_stun_ext'],
        't500_i5_r10_s05': ['ma9_t500_i5_p150_r10_s05_stun_ext', 'ma9_t3000_i5_p150_r10_s05_stun_ext']
    }

    cfg_ids_i10 = {
        't2000_i10_r4_s03': ['ma9_t2000_i10_p150_r4_s03_stun_ext', 'ma9_t3000_i10_p150_r4_s03_stun_ext'],
        't2000_i10_r4_s04': ['ma9_t2000_i10_p150_r4_s04_stun_ext', 'ma9_t3000_i10_p150_r4_s04_stun_ext'],  # ggg
        't2000_i10_r4_s05': ['ma9_t2000_i10_p150_r4_s05_stun_ext', 'ma9_t3000_i10_p150_r4_s05_stun_ext'],
        't2000_i10_r6_s03': ['ma9_t2000_i10_p150_r6_s03_stun_ext', 'ma9_t3000_i10_p150_r6_s03_stun_ext'],
        't2000_i10_r6_s04': ['ma9_t2000_i10_p150_r6_s04_stun_ext', 'ma9_t3000_i10_p150_r6_s04_stun_ext'],
        't2000_i10_r6_s05': ['ma9_t2000_i10_p150_r6_s05_stun_ext', 'ma9_t3000_i10_p150_r6_s05_stun_ext'],
        't2000_i10_r10_s03': ['ma9_t2000_i10_p150_r10_s03_stun_ext', 'ma9_t3000_i10_p150_r10_s03_stun_ext'],
        't2000_i10_r10_s04': ['ma9_t2000_i10_p150_r10_s04_stun_ext', 'ma9_t3000_i10_p150_r10_s04_stun_ext'],
        't2000_i10_r10_s05': ['ma9_t2000_i10_p150_r10_s05_stun_ext', 'ma9_t3000_i10_p150_r10_s05_stun_ext'],

        't1500_i10_r4_s03': ['ma9_t1500_i10_p150_r4_s03_stun_ext', 'ma9_t3000_i10_p150_r4_s03_stun_ext'],
        't1500_i10_r4_s04': ['ma9_t1500_i10_p150_r4_s04_stun_ext', 'ma9_t3000_i10_p150_r4_s04_stun_ext'],
        't1500_i10_r4_s05': ['ma9_t1500_i10_p150_r4_s05_stun_ext', 'ma9_t3000_i10_p150_r4_s05_stun_ext'],  # hhh
        't1500_i10_r6_s03': ['ma9_t1500_i10_p150_r6_s03_stun_ext', 'ma9_t3000_i10_p150_r6_s03_stun_ext'],
        't1500_i10_r6_s04': ['ma9_t1500_i10_p150_r6_s04_stun_ext', 'ma9_t3000_i10_p150_r6_s04_stun_ext'],
        't1500_i10_r6_s05': ['ma9_t1500_i10_p150_r6_s05_stun_ext', 'ma9_t3000_i10_p150_r6_s05_stun_ext'],
        't1500_i10_r10_s03': ['ma9_t1500_i10_p150_r10_s03_stun_ext', 'ma9_t3000_i10_p150_r10_s03_stun_ext'],
        't1500_i10_r10_s04': ['ma9_t1500_i10_p150_r10_s04_stun_ext', 'ma9_t3000_i10_p150_r10_s04_stun_ext'],
        't1500_i10_r10_s05': ['ma9_t1500_i10_p150_r10_s05_stun_ext', 'ma9_t3000_i10_p150_r10_s05_stun_ext'],

        't1000_i10_r4_s03': ['ma9_t1000_i10_p150_r4_s03_stun_ext', 'ma9_t3000_i10_p150_r4_s03_stun_ext'],
        't1000_i10_r4_s04': ['ma9_t1000_i10_p150_r4_s04_stun_ext', 'ma9_t3000_i10_p150_r4_s04_stun_ext'],
        't1000_i10_r4_s05': ['ma9_t1000_i10_p150_r4_s05_stun_ext', 'ma9_t3000_i10_p150_r4_s05_stun_ext'],
        't1000_i10_r6_s03': ['ma9_t1000_i10_p150_r6_s03_stun_ext', 'ma9_t3000_i10_p150_r6_s03_stun_ext'],
        't1000_i10_r6_s04': ['ma9_t1000_i10_p150_r6_s04_stun_ext', 'ma9_t3000_i10_p150_r6_s04_stun_ext'],
        't1000_i10_r6_s05': ['ma9_t1000_i10_p150_r6_s05_stun_ext', 'ma9_t3000_i10_p150_r6_s05_stun_ext'],
        't1000_i10_r10_s03': ['ma9_t1000_i10_p150_r10_s03_stun_ext', 'ma9_t3000_i10_p150_r10_s03_stun_ext'],
        't1000_i10_r10_s04': ['ma9_t1000_i10_p150_r10_s04_stun_ext', 'ma9_t3000_i10_p150_r10_s04_stun_ext'],
        't1000_i10_r10_s05': ['ma9_t1000_i10_p150_r10_s05_stun_ext', 'ma9_t3000_i10_p150_r10_s05_stun_ext'],

        't500_i10_r4_s03': ['ma9_t500_i10_p150_r4_s03_stun_ext', 'ma9_t3000_i10_p150_r4_s03_stun_ext'],
        't500_i10_r4_s04': ['ma9_t500_i10_p150_r4_s04_stun_ext', 'ma9_t3000_i10_p150_r4_s04_stun_ext'],
        't500_i10_r4_s05': ['ma9_t500_i10_p150_r4_s05_stun_ext', 'ma9_t3000_i10_p150_r4_s05_stun_ext'],
        't500_i10_r6_s03': ['ma9_t500_i10_p150_r6_s03_stun_ext', 'ma9_t3000_i10_p150_r6_s03_stun_ext'],
        't500_i10_r6_s04': ['ma9_t500_i10_p150_r6_s04_stun_ext', 'ma9_t3000_i10_p150_r6_s04_stun_ext'],
        't500_i10_r6_s05': ['ma9_t500_i10_p150_r6_s05_stun_ext', 'ma9_t3000_i10_p150_r6_s05_stun_ext'],
        't500_i10_r10_s03': ['ma9_t500_i10_p150_r10_s03_stun_ext', 'ma9_t3000_i10_p150_r10_s03_stun_ext'],
        't500_i10_r10_s04': ['ma9_t500_i10_p150_r10_s04_stun_ext', 'ma9_t3000_i10_p150_r10_s04_stun_ext'],
        't500_i10_r10_s05': ['ma9_t500_i10_p150_r10_s05_stun_ext', 'ma9_t3000_i10_p150_r10_s05_stun_ext']
    }

    kv = None
    if id_ == 'i5':
        kv = cfg_ids_i5
    elif id_ == 'i10':
        kv = cfg_ids_i10

    # What the fuck. Pool doesn't work.
    # Didn't have the time to investigate so I went for the hacky solution.
    # TODO: At some point check this out, might learn something.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for k, v in kv.items():
        p = Process(target=load, args=(k, v[0], v[1], return_dict))
        print('starting', k, v)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print(return_dict)
    result_kv_tuples = return_dict

    names = []
    values = []
    for k in kv:
        names.append(k)
        if k not in result_kv_tuples:
            print('ERR k not found!', k)
            v = kv[k]
            p = Process(target=load, args=(k, v[0], v[1], return_dict))
            p.start()
            p.join()
            values.append(return_dict[k])
        else:
            values.append(result_kv_tuples[k])

    print(names)
    print(values)
    fname = '../pickles/' + id_ + '_coop_starve.pickle'
    print(fname)
    with open(fname, 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
