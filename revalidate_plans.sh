#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=12:00:00


for env in "simple_navigation" "simple_vision" "simple_namo" "complex" "subgoal_obstructed" "attachment_obstructed"
do
    for algo in "a_star" "namo" "vamp" "lamb" "snowplow"
    do
        for seed in "1" "2" "3" "4" "5"
        do
            python run_planner.py --algo="$algo" --env="$env" --seed="$seed" --only_validate="/Users/aidancurtis/movo_manipulation/results/algo=${algo}_env=${env}_seed=${seed}.pkl" --save_dir="/Users/aidancurtis/movo_manipulation/reval_results/"
        done
    done
done
