#!/bin/bash

FOLDER_NAME="soft_option_critic"
declare -a arr=(
    "ec2-3-237-37-222.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Downloads/dongki.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Downloads/dongki.pem -r ~/Desktop/git\ repos/"soft_option_critic" ubuntu@$SSH_ADDRESS:/home/ubuntu/

done

