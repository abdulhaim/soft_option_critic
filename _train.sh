#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# ADD PYTHONPATH
export PYTHONPATH=$DIR/gym_env:$PYTHONPATH

# For MuJoCo
# NOTE Below MuJoCo path, Nvidia driver version, and GLEW path
# may differ depends on a computer setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Comment for using GPU
#export CUDA_VISIBLE_DEVICES=-1

# Begin experiment
cd $DIR
python3.6 main.py \
--model_type "SAC" \
--env_name "BugCrippled"
