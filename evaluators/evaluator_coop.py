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
    base_paths = ['models', 'modelsDec10-14', 'modelsDec15-20']
    base_paths = ['models']
    res = []
    for base_path in base_paths:
        ids_ = [
            '../' + base_path + "/%s-*-F-m1",
            '../' + base_path + "/%s_sp200-*-F-m1",
            '../' + base_path + "/%s-*-6-m1",
            '../' + base_path + "/%s_sp200-*-6-m1"
        ]
        for id__ in ids_:
            print(id__ % cfg_id)
            res.extend(list(glob.glob(id__ % cfg_id)))
    print('####################')
    print(id_)
    print(res)

    failures = 0
    total = 0
    coop_ratios = []
    for fname in res:
        fname = fname[:-3]
        print(fname)
        try:
            for _ in range(20):
                exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
                exp.load_eval(fname, steps=3000, initial_survival_time=3000)
                if exp.env.dead_fishes != 0:
                    coop_ratios.append(exp.env.coop_kills / exp.env.dead_fishes)
                else:
                    failures += 1
                    print('#### NO dead fishes! ####')
                total += 1
        except Exception as ex:
            print('wtf.', ex)
            return_dict['fails'] += 1

    if coop_ratios:
        ci = st.t.interval(0.95, len(coop_ratios)-1, loc=np.mean(coop_ratios), scale=st.sem(coop_ratios))
        print('avg_coop_ratio:%d' % np.mean(coop_ratios))
        return_dict[id_] = (np.mean(coop_ratios), ci, failures / float(total))
    else:
        return_dict[id_] = (0, (0, 0), 0)
    print('Saving', id_)
    print(return_dict)


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_i5 = {
        'i5_r4_s03': ['ma9_t*_i5_p150_r4_s03', 'ma9_t2000_i5_p150_r4_s03_sp200'],
        'i5_r4_s035': ['ma9_t*_i5_p150_r4_s035', 'ma9_t3000_i5_p150_r4_s035_sp200'],
        'i5_r4_s04': ['ma9_t3000_i5_p150_r4_s04', 'ma9_t3000_i5_p150_r4_s04_sp200'],
        'i5_r4_s05': ['ma9_t*_i5_p150_r4_s05', 'ma9_t2000_i5_p150_r4_s05_sp200'],

        'i5_r6_s03': ['ma9_t*_i5_p150_r6_s03', 'ma9_t2000_i5_p150_r6_s03_sp200'],
        'i5_r6_s035': ['ma9_t*_i5_p150_r6_s035', 'ma9_t3000_i5_p150_r6_s035_sp200'],
        'i5_r6_s04': ['ma9_t3000_i5_p150_r6_s04', 'ma9_t3000_i5_p150_r6_s04_sp200'],
        'i5_r6_s05': ['ma9_t*_i5_p150_r6_s05', 'ma9_t2000_i5_p150_r6_s05_sp200'],

        'i5_r10_s03': ['ma9_t*_i5_p150_r10_s03', 'ma9_t2000_i5_p150_r10_s03_sp200'],
        'i5_r10_s035': ['ma9_t*_i5_p150_r10_s035', 'ma9_t3000_i5_p150_r10_s035_sp200'],
        'i5_r10_s04': ['ma9_t3000_i5_p150_r10_s04', 'ma9_t3000_i5_p150_r10_s04_sp200'],
        'i5_r10_s05': ['ma9_t*_i5_p150_r10_s05', 'ma9_t2000_i5_p150_r10_s05_sp200']
    }

    cfg_ids_i10 = {
        'i10_r4_s03': ['ma9_t*_i10_p150_r4_s03', 'ma9_t2000_i10_p150_r4_s03_sp200'],
        'i10_r4_s035': ['ma9_t*_i10_p150_r4_s035', 'ma9_t3000_i10_p150_r4_s035_sp200'],
        'i10_r4_s04': ['ma9_t3000_i10_p150_r4_s04', 'ma9_t3000_i10_p150_r4_s04_sp200'],
        'i10_r4_s05': ['ma9_t*_i10_p150_r4_s05', 'ma9_t2000_i10_p150_r4_s05_sp200'],

        'i10_r6_s03': ['ma9_t*_i10_p150_r6_s03', 'ma9_t2000_i10_p150_r6_s03_sp200'],
        'i10_r6_s035': ['ma9_t*_i10_p150_r6_s035', 'ma9_t3000_i10_p150_r6_s035_sp200'],
        'i10_r6_s04': ['ma9_t3000_i10_p150_r6_s04', 'ma9_t3000_i10_p150_r6_s04_sp200'],
        'i10_r6_s05': ['ma9_t*_i10_p150_r6_s05', 'ma9_t2000_i10_p150_r6_s05_sp200'],

        'i10_r10_s03': ['ma9_t*_i10_p150_r10_s03', 'ma9_t2000_i10_p150_r10_s03_sp200'],
        'i10_r10_s035': ['ma9_t*_i10_p150_r10_s035', 'ma9_t3000_i10_p150_r10_s035_sp200'],
        'i10_r10_s04': ['ma9_t3000_i10_p150_r10_s04', 'ma9_t3000_i10_p150_r10_s04_sp200'],
        'i10_r10_s05': ['ma9_t*_i10_p150_r10_s05', 'ma9_t2000_i10_p150_r10_s05_sp200']
    }

    cfg_ids_i5_stun = {
        'i5_r4_s03': ['ma9_t*_i5_p150_r4_s03_stun_ext', 'ma9_t2000_i5_p150_r4_s03_stun_ext_sp200'],
        'i5_r4_s035': ['ma9_t*_i5_p150_r4_s035_stun_ext', 'ma9_t3000_i5_p150_r4_s035_stun_ext_sp200'],
        'i5_r4_s04': ['ma9_t3000_i5_p150_r4_s04_stun_ext', 'ma9_t3000_i5_p150_r4_s04_stun_ext_sp200'],
        'i5_r4_s05': ['ma9_t*_i5_p150_r4_s05_stun_ext', 'ma9_t2000_i5_p150_r4_s05_stun_ext_sp200'],

        'i5_r6_s03': ['ma9_t*_i5_p150_r6_s03_stun_ext', 'ma9_t2000_i5_p150_r6_s03_stun_ext_sp200'],
        'i5_r6_s035': ['ma9_t*_i5_p150_r6_s035_stun_ext', 'ma9_t3000_i5_p150_r6_s035_stun_ext_sp200'],
        'i5_r6_s04': ['ma9_t3000_i5_p150_r6_s04_stun_ext', 'ma9_t3000_i5_p150_r6_s04_stun_ext_sp200'],
        'i5_r6_s05': ['ma9_t*_i5_p150_r6_s05_stun_ext', 'ma9_t2000_i5_p150_r6_s05_stun_ext_sp200'],

        'i5_r10_s03': ['ma9_t*_i5_p150_r10_s03_stun_ext', 'ma9_t2000_i5_p150_r10_s03_stun_ext_sp200'],
        'i5_r10_s035': ['ma9_t*_i5_p150_r10_s035_stun_ext', 'ma9_t3000_i5_p150_r10_s035_stun_ext_sp200'],
        'i5_r10_s04': ['ma9_t3000_i5_p150_r10_s04_stun_ext', 'ma9_t3000_i5_p150_r10_s04_stun_ext_sp200'],
        'i5_r10_s05': ['ma9_t*_i5_p150_r10_s05_stun_ext', 'ma9_t2000_i5_p150_r10_s05_stun_ext_sp200']
    }

    cfg_ids_i10_stun = {
        'i10_r4_s03': ['ma9_t*_i10_p150_r4_s03_stun_ext', 'ma9_t2000_i10_p150_r4_s03_stun_ext_sp200'],
        'i10_r4_s035': ['ma9_t*_i10_p150_r4_s035_stun_ext', 'ma9_t3000_i10_p150_r4_s035_stun_ext_sp200'],
        'i10_r4_s04': ['ma9_t3000_i10_p150_r4_s04_stun_ext', 'ma9_t3000_i10_p150_r4_s04_stun_ext_sp200'],
        'i10_r4_s05': ['ma9_t*_i10_p150_r4_s05_stun_ext', 'ma9_t2000_i10_p150_r4_s05_stun_ext_sp200'],

        'i10_r6_s03': ['ma9_t*_i10_p150_r6_s03_stun_ext', 'ma9_t2000_i10_p150_r6_s03_stun_ext_sp200'],
        'i10_r6_s035': ['ma9_t*_i10_p150_r6_s035_stun_ext', 'ma9_t3000_i10_p150_r6_s035_stun_ext_sp200'],
        'i10_r6_s04': ['ma9_t3000_i10_p150_r6_s04_stun_ext', 'ma9_t3000_i10_p150_r6_s04_stun_ext_sp200'],
        'i10_r6_s05': ['ma9_t*_i10_p150_r6_s05_stun_ext', 'ma9_t2000_i10_p150_r6_s05_stun_ext_sp200'],

        'i10_r10_s03': ['ma9_t*_i10_p150_r10_s03_stun_ext', 'ma9_t2000_i10_p150_r10_s03_stun_ext_sp200'],
        'i10_r10_s035': ['ma9_t*_i10_p150_r10_s035_stun_ext', 'ma9_t3000_i10_p150_r10_s035_stun_ext_sp200'],
        'i10_r10_s04': ['ma9_t3000_i10_p150_r10_s04_stun_ext', 'ma9_t3000_i10_p150_r10_s04_stun_ext_sp200'],
        'i10_r10_s05': ['ma9_t*_i10_p150_r10_s05_stun_ext', 'ma9_t2000_i10_p150_r10_s05_stun_ext_sp200']
    }

    cfg_ids_i5_vd20 = {
        'i5_r4_s03': ['ma9_i5_p150_r4_s03_sp200_two_net_vd20_f', 'ma9_i5_p150_r4_s03_sp200_two_net_vd20_f'],
        'i5_r4_s035': ['ma9_i5_p150_r4_s03_sp200_two_net_vd20_f', 'ma9_i5_p150_r4_s035_sp200_two_net_vd20_f'],
        'i5_r4_s04': ['ma9_i5_p150_r4_s04_sp200_two_net_vd20_f', 'ma9_i5_p150_r4_s04_sp200_two_net_vd20_f'],
        'i5_r4_s05': ['ma9_i5_p150_r4_s05_sp200_two_net_vd20_f', 'ma9_i5_p150_r4_s05_sp200_two_net_vd20_f'],

        'i5_r6_s03': ['ma9_i5_p150_r6_s03_sp200_two_net_vd20_f', 'ma9_i5_p150_r6_s03_sp200_two_net_vd20_f'],
        'i5_r6_s035': ['ma9_i5_p150_r6_s035_sp200_two_net_vd20_f', 'ma9_i5_p150_r6_s035_sp200_two_net_vd20_f'],
        'i5_r6_s04': ['ma9_i5_p150_r6_s04_sp200_two_net_vd20_f', 'ma9_i5_p150_r6_s04_sp200_two_net_vd20_f'],
        'i5_r6_s05': ['ma9_i5_p150_r6_s05_sp200_two_net_vd20_f', 'ma9_i5_p150_r6_s05_sp200_two_net_vd20_f'],

        'i5_r10_s03': ['ma9_i5_p150_r10_s03_sp200_two_net_vd20_f', 'ma9_i5_p150_r10_s03_sp200_two_net_vd20_f'],
        'i5_r10_s035': ['ma9_i5_p150_r10_s035_sp200_two_net_vd20_f', 'ma9_i5_p150_r10_s035_sp200_two_net_vd20_f'],
        'i5_r10_s04': ['ma9_i5_p150_r10_s04_sp200_two_net_vd20_f', 'ma9_i5_p150_r10_s04_sp200_two_net_vd20_f'],
        'i5_r10_s05': ['ma9_i5_p150_r10_s05_sp200_two_net_vd20_f', 'ma9_i5_p150_r10_s05_sp200_two_net_vd20_f']
    }

    cfg_ids_i5_vd25 = {
        'i5_r4_s03': ['ma9_i5_p150_r4_s03_sp200_two_net_vd25_f', 'ma9_i5_p150_r4_s03_sp200_two_net_vd25_f'],
        'i5_r4_s035': ['ma9_i5_p150_r4_s03_sp200_two_net_vd25_f', 'ma9_i5_p150_r4_s035_sp200_two_net_vd25_f'],
        'i5_r4_s04': ['ma9_i5_p150_r4_s04_sp200_two_net_vd25_f', 'ma9_i5_p150_r4_s04_sp200_two_net_vd25_f'],
        'i5_r4_s05': ['ma9_i5_p150_r4_s05_sp200_two_net_vd25_f', 'ma9_i5_p150_r4_s05_sp200_two_net_vd25_f'],

        'i5_r6_s03': ['ma9_i5_p150_r6_s03_sp200_two_net_vd25_f', 'ma9_i5_p150_r6_s03_sp200_two_net_vd25_f'],
        'i5_r6_s035': ['ma9_i5_p150_r6_s035_sp200_two_net_vd25_f', 'ma9_i5_p150_r6_s035_sp200_two_net_vd25_f'],
        'i5_r6_s04': ['ma9_i5_p150_r6_s04_sp200_two_net_vd25_f', 'ma9_i5_p150_r6_s04_sp200_two_net_vd25_f'],
        'i5_r6_s05': ['ma9_i5_p150_r6_s05_sp200_two_net_vd25_f', 'ma9_i5_p150_r6_s05_sp200_two_net_vd25_f'],

        'i5_r10_s03': ['ma9_i5_p150_r10_s03_sp200_two_net_vd25_f', 'ma9_i5_p150_r10_s03_sp200_two_net_vd25_f'],
        'i5_r10_s035': ['ma9_i5_p150_r10_s035_sp200_two_net_vd25_f', 'ma9_i5_p150_r10_s035_sp200_two_net_vd25_f'],
        'i5_r10_s04': ['ma9_i5_p150_r10_s04_sp200_two_net_vd25_f', 'ma9_i5_p150_r10_s04_sp200_two_net_vd25_f'],
        'i5_r10_s05': ['ma9_i5_p150_r10_s05_sp200_two_net_vd25_f', 'ma9_i5_p150_r10_s05_sp200_two_net_vd25_f']
    }

    cfg_ids_i5_vd30 = {
        'i5_r4_s03': ['ma9_i5_p150_r4_s03_sp200_two_net_vd30_f', 'ma9_i5_p150_r4_s03_sp200_two_net_vd30_f'],
        'i5_r4_s035': ['ma9_i5_p150_r4_s03_sp200_two_net_vd30_f', 'ma9_i5_p150_r4_s035_sp200_two_net_vd30_f'],
        'i5_r4_s04': ['ma9_i5_p150_r4_s04_sp200_two_net_vd30_f', 'ma9_i5_p150_r4_s04_sp200_two_net_vd30_f'],
        'i5_r4_s05': ['ma9_i5_p150_r4_s05_sp200_two_net_vd30_f', 'ma9_i5_p150_r4_s05_sp200_two_net_vd30_f'],

        'i5_r6_s03': ['ma9_i5_p150_r6_s03_sp200_two_net_vd30_f', 'ma9_i5_p150_r6_s03_sp200_two_net_vd30_f'],
        'i5_r6_s035': ['ma9_i5_p150_r6_s035_sp200_two_net_vd30_f', 'ma9_i5_p150_r6_s035_sp200_two_net_vd30_f'],
        'i5_r6_s04': ['ma9_i5_p150_r6_s04_sp200_two_net_vd30_f', 'ma9_i5_p150_r6_s04_sp200_two_net_vd30_f'],
        'i5_r6_s05': ['ma9_i5_p150_r6_s05_sp200_two_net_vd30_f', 'ma9_i5_p150_r6_s05_sp200_two_net_vd30_f'],

        'i5_r10_s03': ['ma9_i5_p150_r10_s03_sp200_two_net_vd30_f', 'ma9_i5_p150_r10_s03_sp200_two_net_vd30_f'],
        'i5_r10_s035': ['ma9_i5_p150_r10_s035_sp200_two_net_vd30_f', 'ma9_i5_p150_r10_s035_sp200_two_net_vd30_f'],
        'i5_r10_s04': ['ma9_i5_p150_r10_s04_sp200_two_net_vd30_f', 'ma9_i5_p150_r10_s04_sp200_two_net_vd30_f'],
        'i5_r10_s05': ['ma9_i5_p150_r10_s05_sp200_two_net_vd30_f', 'ma9_i5_p150_r10_s05_sp200_two_net_vd30_f']
    }

    cfg_ids_i5_vd35 = {
        'i5_r4_s03': ['ma9_i5_p150_r4_s03_sp200_two_net_vd35_f', 'ma9_i5_p150_r4_s03_sp200_two_net_vd35_f'],
        'i5_r4_s035': ['ma9_i5_p150_r4_s03_sp200_two_net_vd35_f', 'ma9_i5_p150_r4_s035_sp200_two_net_vd35_f'],
        'i5_r4_s04': ['ma9_i5_p150_r4_s04_sp200_two_net_vd35_f', 'ma9_i5_p150_r4_s04_sp200_two_net_vd35_f'],
        'i5_r4_s05': ['ma9_i5_p150_r4_s05_sp200_two_net_vd35_f', 'ma9_i5_p150_r4_s05_sp200_two_net_vd35_f'],

        'i5_r6_s03': ['ma9_i5_p150_r6_s03_sp200_two_net_vd35_f', 'ma9_i5_p150_r6_s03_sp200_two_net_vd35_f'],
        'i5_r6_s035': ['ma9_i5_p150_r6_s035_sp200_two_net_vd35_f', 'ma9_i5_p150_r6_s035_sp200_two_net_vd35_f'],
        'i5_r6_s04': ['ma9_i5_p150_r6_s04_sp200_two_net_vd35_f', 'ma9_i5_p150_r6_s04_sp200_two_net_vd35_f'],
        'i5_r6_s05': ['ma9_i5_p150_r6_s05_sp200_two_net_vd35_f', 'ma9_i5_p150_r6_s05_sp200_two_net_vd35_f'],

        'i5_r10_s03': ['ma9_i5_p150_r10_s03_sp200_two_net_vd35_f', 'ma9_i5_p150_r10_s03_sp200_two_net_vd35_f'],
        'i5_r10_s035': ['ma9_i5_p150_r10_s035_sp200_two_net_vd35_f', 'ma9_i5_p150_r10_s035_sp200_two_net_vd35_f'],
        'i5_r10_s04': ['ma9_i5_p150_r10_s04_sp200_two_net_vd35_f', 'ma9_i5_p150_r10_s04_sp200_two_net_vd35_f'],
        'i5_r10_s05': ['ma9_i5_p150_r10_s05_sp200_two_net_vd35_f', 'ma9_i5_p150_r10_s05_sp200_two_net_vd35_f']
    }

    kv = None
    if id_ == 'i5':
        kv = cfg_ids_i5
    elif id_ == 'i10':
        kv = cfg_ids_i10
    elif id_ == 'i5_stun':
        kv = cfg_ids_i5_stun
    elif id_ == 'i10_stun':
        kv = cfg_ids_i10_stun
    elif id_ == 'vd20' or id_ == 'vd20_fast':
        kv = cfg_ids_i5_vd20
    elif id_ == 'vd25':
        kv = cfg_ids_i5_vd25
    elif id_ == 'vd30':
        kv = cfg_ids_i5_vd30
    elif id_ == 'vd35':
        kv = cfg_ids_i5_vd35

    # What the fuck. Pool doesn't work.
    # Didn't have the time to investigate so I went for the hacky solution.
    # TODO: At some point check this out, might learn something.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict['fails'] = 0
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
        values.append(result_kv_tuples[k])

    print('TOTAL EXCEPTIONS', return_dict['fails'])
    print(names)
    print(values)
    fname = '../pickles/' + id_ + '_coop.pickle'
    print(fname)
    with open(fname, 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
