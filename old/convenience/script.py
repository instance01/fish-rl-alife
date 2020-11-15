import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import numpy as np
import pandas as pd
import datetime
from projekt_parser.parser import SafeConfigParser
from warnings import warn
os.chdir(os.path.abspath(os.path.dirname(__file__)))


def summarize():
    root = '../runs'
    sub_folders = [f.path for f in os.scandir(root) if os.path.isdir(f.path)]
    sub_folders.sort()
    parser = SafeConfigParser()

    rows = []
    skipped = 0
    for i, folder in enumerate(sub_folders):

        if os.path.exists(os.path.join(folder, 'not_done.txt')):
            print('Skipping: {}\n-not done |  index: {}'.format(folder, i))
            skipped += 1
            try:
                parser.read(os.path.join(folder, 'experiment_config.ini'))
                text = parser.get_evaluated(section='EXPERIMENT', option='description')
                print('-description:', text)
            except FileNotFoundError:
                continue
            continue

        if re.match('2020-[\d | \\- | _ |\s | :]+', os.path.basename(folder)) is None:
            print('Skipping: {}\n-not an experiment | index: {}'.format(folder, i))
            skipped += 1
            continue

        if not __check_folder(folder):
            print('Skipping: {}\n-check folder failed | index: {}'.format(folder, i))
            skipped += 1
            continue

        experiment = os.path.basename(folder)
        parser.read(os.path.join(folder, 'experiment_config.ini'))
        gamma = parser.get_evaluated(section='AGENT', option='gamma')
        trace_len = parser.get_evaluated(section='HISTORY', option='trace_length')

        with open(os.path.join(folder, 'profiling.txt')) as profiling_file:
            text = profiling_file.read()
            duration = re.findall('duration\\:\\ .+', text)[0]
            duration = duration.replace('duration: ', '')
            warn('Please implement a better datetime regex !')

            machine = re.findall('machine\\:\\ \w+', text)[0]
            machine = machine.replace('machine: ', '')

        evaluation_files = [f.path for f in os.scandir(os.path.join(folder, 'evaluation/'))]
        evaluation_files.sort()

        if not evaluation_files:
            print('Skipping: {}\n-no evaluation files found | index: {}'.format(folder, i))
            continue

        for folder in evaluation_files:
            name = os.path.basename(folder)
            phase = re.findall('phase_[a-z]+', name)[0]
            if phase == 'phase_one':
                phase = 'I'
            elif phase == 'phase_two':
                phase = 'II'
            checkpoint = re.findall('checkpoint_\d+', name)[0]
            checkpoint = checkpoint.replace('checkpoint_', '')

            df = pd.read_csv(folder)
            avg_kills = np.mean(df['killed_fish'])
            std_kills = np.std(df['killed_fish'])
            avg_procreations = np.mean(df['procreations'])
            std_procreations = np.std(df['procreations'])
            rows.append([experiment, trace_len, gamma, phase, checkpoint,  duration, machine,
                         avg_kills, std_kills, avg_procreations, std_procreations])

    header = ["experiment", "trace_length", "gamma", "phase", "checkpoint", "duration", "machine",
              "avg_kills", "std_kills", "avg_procreations", "std_procreations"]

    pd.DataFrame(dict(zip(header, zip(*rows)))).to_csv('./summary.csv')
    print('Skipped {} files'.format(skipped))


def list_experiment_description():
    root = '../runs/'
    sub_folders = [f.path for f in os.scandir(root) if os.path.isdir(f.path)]
    sub_folders.sort()

    parser = SafeConfigParser()

    experiments = []
    for i, folder in enumerate(sub_folders):
        path = os.path.join(folder, 'experiment_config.ini')
        if os.path.exists(path):
            parser.read(path)
            info = parser.get_evaluated(section='EXPERIMENT', option='description')
            status = 'done' if __check_folder(folder) else 'failed'
            experiments.append((i, folder, info, status))
        else:
            experiments.append((i, folder, 'Error while reading:' + path, 'failed'))
    experiments.sort(key=lambda index_folder_text:  index_folder_text[2])

    for index, folder, info, status in experiments:
        print(index, " " * (5 - len(str(index))), folder, " " * (40 - len(str(folder))), info, " " * (30 - len(info)), status)


def remove_unfinished_runs():
    root = '../runs/'
    sub_dirs = [f.path for f in os.scandir(root) if f.is_dir()]

    for sd in sub_dirs:
        files = [file for file in os.listdir(sd)]
        if 'not_done.txt' in files:
            shutil.rmtree(sd)


def __check_folder(folder_path:str):
    if not os.path.exists(os.path.join(folder_path, 'experiment_config.ini')):
        return False
    if not os.path.exists(os.path.join(folder_path, 'evaluation/')):
        return False
    if len(os.listdir(os.path.join(folder_path, 'evaluation/'))) != 4:
        return False
    if os.path.exists(os.path.join(folder_path, 'not_done.txt')):
        return False
    return True


def seed__main_config():

    counter = int(open(r'.experiment_counter').read())

    for _ in range(int(sys.argv[2])):
        str_counter = '0' * (4 - len(str(counter))) + str(counter)
        seed = np.random.randint(0, 999)
        parser = SafeConfigParser()
        parser.read("../main_config.ini")
        parser['AGENT']['seed'] = str(seed)
        parser['ENVIRONMENT']['seed'] = str(seed)
        file_path = '../remote/configs/experiment_{}.ini'.format(str_counter)
        with open(file_path, 'w') as config_file:
            parser.write(config_file)
        counter += 1

    with open(r'.experiment_counter', 'w') as file:
        file.write(str(counter))


######################################################################################################################

functions_dict = {'list': list_experiment_description,
                  'clear': remove_unfinished_runs,
                  'seed': seed__main_config,
                  'summary': summarize,
                  }

######################################################################################################################

if __name__ == '__main__':
    call_to = sys.argv[1]
    try:
        functions_dict[call_to]()
    except KeyError as e:
        raise NotImplementedError("The keyword-argument {} could not "
                                  "be resolved to a function call !".format(call_to), e)





