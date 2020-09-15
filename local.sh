#!/bin/bash

FOLDER_NAME="soft_option_critic"
declare -a arr=(
    "ec2-34-202-165-150.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Documents/abdulhai.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Documents/abdulhai.pem -r ~/Documents/meng_repos/"soft_option_critic" ubuntu@$SSH_ADDRESS:/home/ubuntu/

done

