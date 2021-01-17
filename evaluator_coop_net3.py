import os
import sys
import glob
import pickle
import multiprocessing
from multiprocessing import Process
import numpy as np
import scipy.stats as st
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pipeline import Experiment


def load(id_, cfg_id, base_cfg_id, return_dict):
    return_dict[id_] = (-1, (0, 0))
    # base_paths = ['models', 'modelsDec10-14']
    base_paths = ['models']
    res = []
    for base_path in base_paths:
        ids_ = [
            base_path + "/%s-*-F-m1",
            base_path + "/%s-*-6-m1"
        ]
        for id__ in ids_:
            print(id__ % cfg_id)
            res.extend(list(glob.glob(id__ % cfg_id)))
    print('####################')
    print(id_)
    print(res)

    coop_ratios = []
    failures = []
    for fname in res:
        fname = fname[:-3]
        # print(fname)
        for _ in range(20):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            exp.load_eval(fname, steps=3000, initial_survival_time=3000)
            if exp.env.dead_fishes != 0:
                print(exp.env.full_coop_kills, exp.env.dead_fishes)
                coop_ratios.append(exp.env.full_coop_kills / exp.env.dead_fishes)
                failures.append(0.)
            else:
                failures.append(1.)
                print('####### NO dead fishes! #######')
                # TODO DUBIOUS! Should we maybe add a [0] if there's no dead fishes?
                # lets try.
                # coop_ratios.append(0)

    if coop_ratios:
        ci = st.t.interval(0.95, len(coop_ratios)-1, loc=np.mean(coop_ratios), scale=st.sem(coop_ratios))
        print('avg_coop_ratio:%d' % np.mean(coop_ratios))
        ci_fail = st.t.interval(0.95, len(failures)-1, loc=np.mean(failures), scale=st.sem(failures))
        print('avg_fail_ratio:%d' % np.mean(failures))
        return_dict[id_] = (np.mean(coop_ratios), ci, np.mean(failures), ci_fail)
    else:
        return_dict[id_] = (0, (0, 0), 0)


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_vd15 = {
        'r4_s025': ['ma9_i5_p150_r4_s025_sp200_n_net3_vd15'],
        'r4_s03': ['ma9_i5_p150_r4_s03_sp200_n_net3_vd15'],
        'r4_s035': ['ma9_i5_p150_r4_s035_sp200_n_net3_vd15'],
        'r4_s04': ['ma9_i5_p150_r4_s04_sp200_n_net3_vd15'],
        'r4_s05': ['ma9_i5_p150_r4_s05_sp200_n_net3_vd15'],
        'r6_s025': ['ma9_i5_p150_r6_s025_sp200_n_net3_vd15'],
        'r6_s03': ['ma9_i5_p150_r6_s03_sp200_n_net3_vd15'],
        'r6_s035': ['ma9_i5_p150_r6_s035_sp200_n_net3_vd15'],
        'r6_s04': ['ma9_i5_p150_r6_s04_sp200_n_net3_vd15'],
        'r6_s05': ['ma9_i5_p150_r6_s05_sp200_n_net3_vd15'],
        'r10_s025': ['ma9_i5_p150_r10_s025_sp200_n_net3_vd15'],
        'r10_s03': ['ma9_i5_p150_r10_s03_sp200_n_net3_vd15'],
        'r10_s035': ['ma9_i5_p150_r10_s035_sp200_n_net3_vd15'],
        'r10_s04': ['ma9_i5_p150_r10_s04_sp200_n_net3_vd15'],
        'r10_s05': ['ma9_i5_p150_r10_s05_sp200_n_net3_vd15']
    }

    cfg_ids_vd20 = {
        'r4_s025': ['ma9_i5_p150_r4_s025_sp200_n_net3_vd20'],
        'r4_s03': ['ma9_i5_p150_r4_s03_sp200_n_net3_vd20'],
        'r4_s035': ['ma9_i5_p150_r4_s035_sp200_n_net3_vd20'],
        'r4_s04': ['ma9_i5_p150_r4_s04_sp200_n_net3_vd20'],
        'r4_s05': ['ma9_i5_p150_r4_s05_sp200_n_net3_vd20'],
        'r6_s025': ['ma9_i5_p150_r6_s025_sp200_n_net3_vd20'],
        'r6_s03': ['ma9_i5_p150_r6_s03_sp200_n_net3_vd20'],
        'r6_s035': ['ma9_i5_p150_r6_s035_sp200_n_net3_vd20'],
        'r6_s04': ['ma9_i5_p150_r6_s04_sp200_n_net3_vd20'],
        'r6_s05': ['ma9_i5_p150_r6_s05_sp200_n_net3_vd20'],
        'r10_s025': ['ma9_i5_p150_r10_s025_sp200_n_net3_vd20'],
        'r10_s03': ['ma9_i5_p150_r10_s03_sp200_n_net3_vd20'],
        'r10_s035': ['ma9_i5_p150_r10_s035_sp200_n_net3_vd20'],
        'r10_s04': ['ma9_i5_p150_r10_s04_sp200_n_net3_vd20'],
        'r10_s05': ['ma9_i5_p150_r10_s05_sp200_n_net3_vd20']
    }

    cfg_ids_vd25 = {
        'r4_s025': ['ma9_i5_p150_r4_s025_sp200_n_net3_vd25'],
        'r4_s03': ['ma9_i5_p150_r4_s03_sp200_n_net3_vd25'],
        'r4_s035': ['ma9_i5_p150_r4_s035_sp200_n_net3_vd25'],
        'r4_s04': ['ma9_i5_p150_r4_s04_sp200_n_net3_vd25'],
        'r4_s05': ['ma9_i5_p150_r4_s05_sp200_n_net3_vd25'],
        'r6_s025': ['ma9_i5_p150_r6_s025_sp200_n_net3_vd25'],
        'r6_s03': ['ma9_i5_p150_r6_s03_sp200_n_net3_vd25'],
        'r6_s035': ['ma9_i5_p150_r6_s035_sp200_n_net3_vd25'],
        'r6_s04': ['ma9_i5_p150_r6_s04_sp200_n_net3_vd25'],
        'r6_s05': ['ma9_i5_p150_r6_s05_sp200_n_net3_vd25'],
        'r10_s025': ['ma9_i5_p150_r10_s025_sp200_n_net3_vd25'],
        'r10_s03': ['ma9_i5_p150_r10_s03_sp200_n_net3_vd25'],
        'r10_s035': ['ma9_i5_p150_r10_s035_sp200_n_net3_vd25'],
        'r10_s04': ['ma9_i5_p150_r10_s04_sp200_n_net3_vd25'],
        'r10_s05': ['ma9_i5_p150_r10_s05_sp200_n_net3_vd25']
    }

    cfg_ids_vd30 = {
        'r4_s025': ['ma9_i5_p150_r4_s025_sp200_n_net3_vd30'],
        'r4_s03': ['ma9_i5_p150_r4_s03_sp200_n_net3_vd30'],
        'r4_s035': ['ma9_i5_p150_r4_s035_sp200_n_net3_vd30'],
        'r4_s04': ['ma9_i5_p150_r4_s04_sp200_n_net3_vd30'],
        'r4_s05': ['ma9_i5_p150_r4_s05_sp200_n_net3_vd30'],
        'r6_s025': ['ma9_i5_p150_r6_s025_sp200_n_net3_vd30'],
        'r6_s03': ['ma9_i5_p150_r6_s03_sp200_n_net3_vd30'],
        'r6_s035': ['ma9_i5_p150_r6_s035_sp200_n_net3_vd30'],
        'r6_s04': ['ma9_i5_p150_r6_s04_sp200_n_net3_vd30'],
        'r6_s05': ['ma9_i5_p150_r6_s05_sp200_n_net3_vd30'],
        'r10_s025': ['ma9_i5_p150_r10_s025_sp200_n_net3_vd30'],
        'r10_s03': ['ma9_i5_p150_r10_s03_sp200_n_net3_vd30'],
        'r10_s035': ['ma9_i5_p150_r10_s035_sp200_n_net3_vd30'],
        'r10_s04': ['ma9_i5_p150_r10_s04_sp200_n_net3_vd30'],
        'r10_s05': ['ma9_i5_p150_r10_s05_sp200_n_net3_vd30']
    }

    cfg_ids_vd35 = {
        'r4_s025': ['ma9_i5_p150_r4_s025_sp200_n_net3_vd35'],
        'r4_s03': ['ma9_i5_p150_r4_s03_sp200_n_net3_vd35'],
        'r4_s035': ['ma9_i5_p150_r4_s035_sp200_n_net3_vd35'],
        'r4_s04': ['ma9_i5_p150_r4_s04_sp200_n_net3_vd35'],
        'r4_s05': ['ma9_i5_p150_r4_s05_sp200_n_net3_vd35'],
        'r6_s025': ['ma9_i5_p150_r6_s025_sp200_n_net3_vd35'],
        'r6_s03': ['ma9_i5_p150_r6_s03_sp200_n_net3_vd35'],
        'r6_s035': ['ma9_i5_p150_r6_s035_sp200_n_net3_vd35'],
        'r6_s04': ['ma9_i5_p150_r6_s04_sp200_n_net3_vd35'],
        'r6_s05': ['ma9_i5_p150_r6_s05_sp200_n_net3_vd35'],
        'r10_s025': ['ma9_i5_p150_r10_s025_sp200_n_net3_vd35'],
        'r10_s03': ['ma9_i5_p150_r10_s03_sp200_n_net3_vd35'],
        'r10_s035': ['ma9_i5_p150_r10_s035_sp200_n_net3_vd35'],
        'r10_s04': ['ma9_i5_p150_r10_s04_sp200_n_net3_vd35'],
        'r10_s05': ['ma9_i5_p150_r10_s05_sp200_n_net3_vd35']
    }

    kv = None
    if id_ == 'vd15':
        kv = cfg_ids_vd15
    elif id_ == 'vd20':
        kv = cfg_ids_vd20
    elif id_ == 'vd25':
        kv = cfg_ids_vd25
    elif id_ == 'vd30':
        kv = cfg_ids_vd30
    elif id_ == 'vd35':
        kv = cfg_ids_vd35

    # What the fuck. Pool doesn't work.
    # Didn't have the time to investigate so I went for the hacky solution.
    # TODO: At some point check this out, might learn something.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for k, v in kv.items():
        p = Process(target=load, args=(k, v[0], v[0], return_dict))
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
            p = Process(target=load, args=(k, v[0], v[0], return_dict))
            p.start()
            p.join()
            values.append(return_dict[k])
        else:
            values.append(result_kv_tuples[k])

    # TODO Unfinished, could be a 'continue' feature.
    # if os.path.isfile('pickles/' + id_ + '_coop_net3.pickle'):
    #     with open('pickles/' + id_ + '_coop_net3.pickle', 'rb') as f:
    #         names_old, values_old = pickle.laod(f)
    #     names_old.extend(names)
    #     valu

    print(names)
    print(values)
    print('pickles/' + id_ + '_coop_net3.pickle')
    with open('pickles/' + id_ + '_coop_net3.pickle', 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
