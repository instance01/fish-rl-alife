from __future__ import print_function
import itertools
import random
import time
import numpy as np
import os
from threading import Thread, Semaphore
import sys

def errorPrint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def launch(semaphore, args):
    os.system("python multiagentDqnLearner.py " + " ".join(map(str, args)))
    #print("sbatch submit.sh python multiagentDqnLearner.py " + " ".join(map(str, args)))
    semaphore.release()

seeds = range(int(sys.argv[2]), int(sys.argv[3]))
steps = [1000000]
hiddenL = [10]
hiddenN = [16]
gamma = [.999999]
lr = [1e-3]
memory = [50000]
batch_size = [64]
warmup = [10]
fish = [10, 20]
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
cores = Semaphore(parallel)
for t in l:
    cores.acquire()
    Thread(target=launch, args=(cores, t)).start()

    if parallel > 0:
        time.sleep(np.random.uniform(1, 3))
        parallel -= 1
