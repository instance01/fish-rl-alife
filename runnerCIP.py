from __future__ import print_function
import itertools
import random
import time
import numpy as np
import os
from threading import Thread, Semaphore
import subprocess
import sys

def errorPrint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


seeds = range(int(sys.argv[2]), int(sys.argv[3]))
steps = [5000000]
hiddenL = [10]
hiddenN = [16]
gamma = [.999999]
lr = [1e-3]
memory = [50000]
batch_size = [64]
warmup = [10]
fish = [10]
observableFish = [1, 2, 5]

# seeds = range(5)
# steps = [500000, 1000000]
# hiddenL = [5, 10]
# hiddenN = [10, 16]
# gamma = [.95, .999999]
# lr = [1e-3]
# memory = [50000]
# batch_size = [32, 64, 128]
# warmup = [10]
# fish = [5, 10]
# observableFish = [1, 2, 5]


l = list(itertools.product(seeds, steps, hiddenL, hiddenN, gamma, lr, memory, batch_size, warmup, fish, observableFish))

random.seed(seeds[0])
random.shuffle(l)

for t in l:
    errorPrint(t)

parallel = int(sys.argv[1])
for args in l:
    running = int(subprocess.check_output("squeue | grep hahnca | wc -l", shell=True))
    while running >= parallel:
        time.sleep(5*60)
        running = int(subprocess.check_output("squeue | grep hahnca | wc -l", shell=True))
    os.system("sbatch --job-name=multiagentDqn_" + "_".join(map(str, args)) + " submit.sh python multiagentDqnLearner.py " + " ".join(map(str, args)))
    
    
            
