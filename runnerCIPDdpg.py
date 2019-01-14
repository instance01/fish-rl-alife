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
steps = [500000]
hiddenL = [5, 10]
hiddenNActor = [16, 200]
hiddenNCritic = [32, 200]
gamma = [.999999]
lr = [1e-3]
memory = [100000]
batch_size = [64, 128, 512]
warmup = [100]
fish = [10,20]
observableFish = [2,5]



l = list(itertools.product(seeds, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, memory, batch_size, warmup, fish, observableFish))
l = [t for t in l if t[3] == 16 and t[4] == 32 or t[3] == 200 and t[4] == 200]

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
    os.system("sbatch --job-name=multiagentDdpq_" + "_".join(map(str, args)) + " submit.sh python multiagentDdpgLearner.py " + " ".join(map(str, args)))
    
    
            
