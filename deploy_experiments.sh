#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --time=03:00:00

i=1
for env in "simple_navigation" "simple_vision" "simple_namo" "complex" "subgoal_obstructed" "attachment_obstructed"
do
    for algo in "a_star" "namo" "vamp" "lamb" "snowplow"
    do
        for seed in "1" "2" "3" "4" "5"
        do
            if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
                python run_planner.py --algo="$algo" --env="$env" --seed="$seed" --save_dir="/home/gridsan/acurtis/movo_manipulation/results/"
            fi
            i=$((i+1))
        done
    done
done