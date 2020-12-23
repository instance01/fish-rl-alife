import cProfile
from main import Experiment


PROFILE_FILE = '1.profile'


def run():
    cfg_id = 'dbg_n_net3_s035_i25_50x50'
    Experiment(cfg_id).train()


cProfile.runctx("run()", globals(), locals(), PROFILE_FILE)
