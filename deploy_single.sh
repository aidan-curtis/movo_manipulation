#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=12:00:00

python run_planner.py --seed=4 --algo=snowplow --env=simple_navigation --debug --save_dir="/home/gridsan/acurtis/movo_manipulation/results/"