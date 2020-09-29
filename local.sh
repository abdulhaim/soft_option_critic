#!/bin/bash

FOLDER_NAME="soft_option_critic"
declare -a arr=(
    "ec2-52-206-27-8.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Documents/abdulhai.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Documents/abdulhai.pem ~/Documents/meng_repos/$FOLDER_NAME/*.py ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem ~/Documents/meng_repos/$FOLDER_NAME/*.sh ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem ~/Documents/meng_repos/$FOLDER_NAME/.gitignore ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem ~/Documents/meng_repos/$FOLDER_NAME/*.md ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem ~/Documents/meng_repos/$FOLDER_NAME/requirements.txt ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/

    scp -i ~/Documents/abdulhai.pem -r ~/Documents/meng_repos/$FOLDER_NAME/agent_utils/ ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem -r ~/Documents/meng_repos/$FOLDER_NAME/gym_env/ ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/abdulhai.pem -r ~/Documents/meng_repos/$FOLDER_NAME/Model/ ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/


done

