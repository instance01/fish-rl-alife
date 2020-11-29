#screen -ls | grep '(Detached)' | awk '{print $1}' | xargs -I % -t screen -X -S % quit
screen -ls | grep 'nm6' | awk '{print $1}' | xargs -I % -t screen -X -S % quit
