#! /bin/sh

#SBATCH --job-name=train
#SBATCH --output=/home/yandex/MLW2023/yuvalramot/Model/train.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/yuvalramot/Model/awesome.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=1 # CPU cores per process
#SBATCH --gres=gpu:4 # GPUs in total


python3 train.py