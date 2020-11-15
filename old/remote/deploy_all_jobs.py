import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import subprocess
from datetime import datetime
from projekt_parser.parser import SafeConfigParser
os.chdir(os.path.abspath(os.path.dirname(__file__)))

GOBI_CLUSTER = ['aquamarin', 'benitoit', 'beryll', 'brasilianit', 'buergerit', 'cordierit', 'danburit', 'datolith', 'diamant', 'dioptas', 'dravit', 'elbait', 'euklas', 'falkenauge', 'feuerachat', 'feueropal', 'gagat', 'goshenit', 'hambergit', 'heliodor', 'indigiolith', 'almandin', 'citrin', 'hackmannit', 'amazonit']
LUNA_CLUSTER = ['peridot', 'petalit', 'rubin', 'sodalith', 'tansanit', 'thulit', 'tigerauge', 'topas', 'vesuvianit', 'zirkon', 'jaspis', 'karneol', 'katzenauge', 'labradorit', 'lapislazuli', 'leucit', 'pyrit', 'rhodonit', 'rubellit', 'saphir', 'smaragd']
SIBIRIEN_CLUSTER = ['mehnach', 'mindel', 'moosach', 'peitnach', 'pfettrach', 'pfreimd', 'ranna', 'rezat', 'sandrach', 'schlierach', 'selbitz', 'sempt', 'starzlach', 'mauch', 'naab', 'nassach', 'nau', 'osterbach', 'ostrach', 'oybach', 'partnach', 'rottach', 'arbach', 'aisch', 'breitach']
KALAHARI_CLUSTER = ['frommbach', 'gaissa', 'geltnach', 'gloett', 'guenz', 'haselgraben', 'hasslach', 'huehnerbach', 'ilm', 'ilz', 'inn', 'isar', 'jachen', 'kinsach', 'kirnach', 'krassach', 'kronach', 'leubas', 'loisach', 'luhe', 'mangfall']
ARKTIS_CLUSTER = ['abens', 'aiterach', 'altmuehl', 'amper', 'ampfrach', 'anlauter', 'blau', 'brenz', 'buxach', 'chamb', 'donau', 'dorfen', 'itz']
REMOTES_MACHINES = GOBI_CLUSTER + LUNA_CLUSTER + SIBIRIEN_CLUSTER + KALAHARI_CLUSTER + ARKTIS_CLUSTER

HOLDBACK_GPU_MEMORY = 300  # MB


def query_remote_machine(remote_name: str):
    cpu_answer = subprocess.run(['ssh', remote_name, 'nproc'], stdout=subprocess.PIPE)
    gpu_answer = subprocess.run(['ssh', remote_name, 'nvidia-smi', '--query-gpu=memory.free', '--format=csv'],
                                stdout=subprocess.PIPE)
    gpu_answer = gpu_answer.stdout.decode("utf-8")
    cpu_answer = cpu_answer.stdout.decode("utf-8")

    free_cpu_cores = 0
    free_gpu_memory = 0
    if'failed' not in gpu_answer and re.match('memory\\.free\\ \\[MiB\\]\\\n[0-9]+\\ MiB', gpu_answer):
        free_gpu_memory = int(re.findall('[0-9]+', gpu_answer)[0])
    else:
        print('ERROR: failed at querying gpu at {}\n{}\n'.format(remote_name, gpu_answer), file=sys.stdout)

    if re.match('[0-9]+', cpu_answer):
        free_cpu_cores = int(cpu_answer)
    else:
        print('ERROR: failed at querying cpu at {}\n{}\n'.format(remote_name, cpu_answer), file=sys.stdout)

    return free_cpu_cores, free_gpu_memory


def submit_jobs(remote_name: str, config_files: list):

    cpu_cores, free_gpu_memory = query_remote_machine(remote_name)
    free_gpu_memory -= HOLDBACK_GPU_MEMORY

    parser = SafeConfigParser()
    failed_tasks = []

    push_to_remote = True
    while config_files and push_to_remote:

        file = config_files.pop()
        parser.read(file)
        has_gpu_estimation = parser.has_option(section='RESOURCES', option='gpu_memory')
        has_cpu_estimation = parser.has_option(section='RESOURCES', option='cpu_cores')
        if has_gpu_estimation and has_cpu_estimation:
            needed_gpu_memory = parser.get_evaluated(section='RESOURCES', option='gpu_memory')
            needed_cpu_cores = parser.get_evaluated(section='RESOURCES', option='cpu_cores')
        else:
            failed_tasks.append(file)
            print("ERROR: No values found for [gpu_memory/cpu_cores] in config file {}!".
                  format(file), file=sys.stdout)
            continue

        if free_gpu_memory >= needed_gpu_memory and cpu_cores >= needed_cpu_cores:
            command = 'python3 ~/git/fish-environment-code-update/main.py {}'.format(file)
            output = file.replace('configs', 'outputs').replace('.ini', '.out')
            os.system('ssh {} \'source ~/git/fish-environment-code-update/venv/bin/activate && nohup {} > {} &\''.
                      format(remote_name, command, output))
            free_gpu_memory -= needed_gpu_memory
            cpu_cores -= needed_cpu_cores
            print('{} started...'.format(file))
            time.sleep(1)
        else:
            config_files.append(file)
            push_to_remote = False

    return config_files, failed_tasks


def start():

    tasks = [os.path.abspath(f.path) for f in os.scandir('./configs') if '.gitkeep' not in f.path]
    tasks.sort()
    print('\n##### tasks #####\n{} tasks found'.format(len(tasks)))

    dead_machines = subprocess.run(['sinfo', '-d', '-h',  '--format=\\"\\%N\\"'], stdout=subprocess.PIPE)
    dead_machines = dead_machines.stdout.decode("utf-8")
    dead_machines = dead_machines.split(',')
    dead_machines = [d.replace('\n', '').replace('\\"', '').replace('\\', '') for d in dead_machines]

    print('\n#####dead machines #####\n',
          str(dead_machines)[1:-1],)

    failed_tasks = []
    for pc in REMOTES_MACHINES:
        print('\nremote name:', pc)

        if pc in dead_machines:
            print('\nremote machine: {} is down'.format(pc), file=sys.stderr)
        else:
            rest, fails = submit_jobs(pc, tasks)
            failed_tasks += fails
            if not rest:
                print('\nno Tasks left to deploy')
                break
            else:
                tasks = rest

    if failed_tasks or tasks:
        with open('deploy_log_{}.txt'.format(datetime.now().replace(microsecond=0)), 'w') as log:
            for f in failed_tasks:
                log.write('failed at {}\n'.format(f))
            for t in tasks:
                log.write('not deployed {}\n'.format(t))
    exit()


if __name__ == '__main__':
    start()

