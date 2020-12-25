import sys
import glob
import pickle
import multiprocessing
from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import scipy.stats as st


def load(id_, cfg_id, base_cfg_id, return_dict):
    from pipeline import Experiment
    base_paths = ['models', 'modelsDec10-14']
    res = []
    for base_path in base_paths:
        ids_ = [
            base_path + "/%s-*-F-m1",
            base_path + "/%s_sp200-*-F-m1",
            base_path + "/%s-*-6-m1",
            base_path + "/%s_sp200-*-6-m1"
        ]
        for id__ in ids_:
            print(id__ % cfg_id)
            res.extend(list(glob.glob(id__ % cfg_id)))
    print('####################')
    print(res)

    coop_ratios = []
    for fname in res:
        fname = fname[:-3]
        print(fname)
        for _ in range(20):
            exp = Experiment(base_cfg_id, show_gui=False, dump_cfg=False)
            exp.load_eval(fname, steps=3000, initial_survival_time=3000)
            if exp.env.dead_fishes != 0:
                coop_ratios.append(exp.env.coop_kills / exp.env.dead_fishes)
            else:
                print('####################')
                print('NO dead fishes!')
                print('####################')
            # TODO DUBIOUS! Should we maybe add a [0] if there's no dead fishes?

    ci = st.t.interval(0.95, len(coop_ratios)-1, loc=np.mean(coop_ratios), scale=st.sem(coop_ratios))
    print('avg_coop_ratio:%d' % np.mean(coop_ratios))
    return_dict[id_] = (np.mean(coop_ratios), ci)


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
        values.append(result_kv_tuples[k])

    print(names)
    print(values)
    print('pickles/' + id_ + '_coop.pickle')
    with open('pickles/' + id_ + '_coop.pickle', 'wb+') as f:
        pickle.dump((names, values), f)


if __name__ == '__main__':
    id_ = sys.argv[1]
    main(id_)
