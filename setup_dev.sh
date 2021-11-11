#!/bin/sh

curl --version >/dev/null 2>&1 || sudo apt-get install curl
conda --version >/dev/null 2>&1 || curl "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh" --output "/tmp/Anaconda3-2021.05-Linux-x86_64.sh"
conda --version >/dev/null 2>&1 || bash /tmp/Anaconda3-2021.05-Linux-x86_64.sh
conda --version >/dev/null 2>&1 || sudo apt-get update
conda --version >/dev/null 2>&1 || sudo apt-get upgrade

conda update --name base conda
# conda init
eval "$(conda shell.bash hook)"
conda activate gomoku_env || conda create -n gomoku_env python=3.9
echo "\n# Use conda activate gomoku_env to activate\n"