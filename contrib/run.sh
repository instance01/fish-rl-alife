#!/bin/bash
#declare -a arr=("lunar1" "lunar2" "lunar3" "lunar4" "lunar5")
#declare -a arr=("ma6" "ma6" "ma6" "ma6")

threads="1"
partition="All"
# pyrit: We are segfaulting on there. (torch)
# jaspis: Has no fucking home directories.. wtf
#excluded="pyrit,vesuvianit,jaspis"
#excluded="citrin,amazonit,vesuvianit,jaspis,dravit"
#excluded="citrin,amazonit,jaspis,dravit"
#excluded="ecknach,aquamarin,citrin,amazonit,jaspis,dravit,euklas"
excluded="buxach"

for i in "${arr[@]}"
do
   echo "fish-$i"
   #screen -dmS "fish-$i" bash -c 'source /home/r/ratke/PY3ENV/bin/activate && cd /home/r/ratke/fish/ && srun -v -x '$excluded' -s -p '$partition' -c '$threads' python3 main.py '$i'; exec bash' 2>&1 &
   screen -dmS "fish-$i" bash -c 'source /home/r/ratke/PY3ENV/bin/activate && cd /home/r/ratke/fish/ && python3 main.py '$i'; exec bash' 2>&1 &
done
