import os
import sys
from framework.buildingblocks import Experiment
os.chdir(os.path.abspath(os.path.dirname(__file__)))

if __name__ == '__main__':
    config_file = sys.argv[1]
    e = Experiment(config_file)
    e.start_phase_one()
    e.start_phase_two()
    e.flag_done()



