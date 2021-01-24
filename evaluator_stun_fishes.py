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

    stuns = []
    for fname in res:
        fname = fname[:-3]
        # print(fname)
        for _ in range(20):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            exp.load_eval(fname, steps=3000, initial_survival_time=3000)
            stuns.append(exp.env.n_stuns)

    if stuns:
        ci = st.t.interval(0.95, len(stuns)-1, loc=np.mean(stuns), scale=st.sem(stuns))
        print('avg_stuns:%d' % np.mean(stuns), stuns)
        return_dict[id_] = (np.mean(stuns), ci)
    else:
        return_dict[id_] = (0, (0, 0))


def main(id_):
    multiprocessing.set_start_method('spawn')

    cfg_ids_t3000 = {
        't3000_i2_d100': ['ma9_t3000_i2_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i2_p150_r6_s06_stun_ext_d100'],
        't3000_i4_d100': ['ma9_t3000_i4_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i4_p150_r6_s06_stun_ext_d100'],
        't3000_i6_d100': ['ma9_t3000_i6_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i6_p150_r6_s06_stun_ext_d100'],
        't3000_i8_d100': ['ma9_t3000_i8_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i8_p150_r6_s06_stun_ext_d100'],
        't3000_i10_d100': ['ma9_t3000_i10_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i10_p150_r6_s06_stun_ext_d100'],

        't3000_i2_d300': ['ma9_t3000_i2_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i2_p150_r6_s06_stun_ext_d300'],
        't3000_i4_d300': ['ma9_t3000_i4_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i4_p150_r6_s06_stun_ext_d300'],
        't3000_i6_d300': ['ma9_t3000_i6_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i6_p150_r6_s06_stun_ext_d300'],
        't3000_i8_d300': ['ma9_t3000_i8_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i8_p150_r6_s06_stun_ext_d300'],
        't3000_i10_d300': ['ma9_t3000_i10_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i10_p150_r6_s06_stun_ext_d300']
    }

    cfg_ids_t1000 = {
        't1000_i2_d100': ['ma9_t1000_i2_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i2_p150_r6_s06_stun_ext_d100'],
        't1000_i4_d100': ['ma9_t1000_i4_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i4_p150_r6_s06_stun_ext_d100'],
        't1000_i6_d100': ['ma9_t1000_i6_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i6_p150_r6_s06_stun_ext_d100'],
        't1000_i8_d100': ['ma9_t1000_i8_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i8_p150_r6_s06_stun_ext_d100'],
        't1000_i10_d100': ['ma9_t1000_i10_p150_r6_s06_stun_ext_d100', 'ma9_t3000_i10_p150_r6_s06_stun_ext_d100'],

        't1000_i2_d300': ['ma9_t1000_i2_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i2_p150_r6_s06_stun_ext_d300'],
        't1000_i4_d300': ['ma9_t1000_i4_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i4_p150_r6_s06_stun_ext_d300'],
        't1000_i6_d300': ['ma9_t1000_i6_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i6_p150_r6_s06_stun_ext_d300'],
        't1000_i8_d300': ['ma9_t1000_i8_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i8_p150_r6_s06_stun_ext_d300'],
        't1000_i10_d300': ['ma9_t1000_i10_p150_r6_s06_stun_ext_d300', 'ma9_t3000_i10_p150_r6_s06_stun_ext_d300']
    }

    kv = None
    if id_ == 't3000':
        kv = cfg_ids_t3000
    elif id_ == 't1000':
        kv = cfg_ids_t1000

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
    print('pickles/' + id_ + '_stuns_fishes.pickle')
    with open('pickles/' + id_ + '_stuns_fishes.pickle', 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
