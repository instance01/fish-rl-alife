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
    os.system("python multiagentDdpgLearner.py " + " ".join(map(str, args)))
    semaphore.release()

# seeds = range(3)
# steps = [100000, 500000]
# hiddenL = [1, 3, 5]
# hiddenNActor = [16, 200]
# hiddenNCritic = [32, 200]
# gamma = [.99]
# lr = [1e-3]
# memory = [100000]
# batch_size = [64, 128, 512]
# warmup = [100]
# fish = [3]
# observableFish = [3]

seeds = range(int(sys.argv[2]), int(sys.argv[3]))
steps = [100000]
hiddenL = [1]
hiddenNActor = [16]
hiddenNCritic = [32]
gamma = [.99]
lr = [1e-3]
memory = [100000]
batch_size = [64]
warmup = [100]
fish = [3]
observableFish = [3]

# seeds = range(1)
# steps = [10000, 50000]
# hiddenL = [1, 3]
# hiddenN = [16, 32]
# gamma = [.95, .99]
# lr = [1e-1,1e-3]
# memory = [10000, 50000]
# batch_size = [32, 1024]
# warmup = [10, 1000]


l = list(itertools.product(seeds, steps, hiddenL, hiddenNActor, hiddenNCritic, gamma, lr, memory, batch_size, warmup, fish, observableFish))
l = [t for t in l if t[3] == 16 and t[4] == 32 or t[3] == 200 and t[4] == 200]

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
