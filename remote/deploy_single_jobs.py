import os
import sys
import time

if __name__ == '__main__':
    start = int(sys.argv[1])
    config_files = [s.path for s in os.scandir('remote/configs') if not'.gitkeep' in s.path]
    config_files.sort()
    for f in config_files[start:]:
        print('up now:  ', f)
        signal = input('[y/n] >>> ')
        if signal == 'y':
            command = 'python3 main.py {}'.format(f)
            log_file = 'remote/outputs/{}'.format(os.path.basename(f).replace('.ini', '.out'))
            os.system('nohup {} > {} & '.format(command, log_file))
        elif signal == 'n':
            exit()
        else:
            print('input unknown!')
        time.sleep(1)  # wait for answer

    print('\nNo jobs left!')
