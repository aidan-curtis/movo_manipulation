#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 1
#SBATCH --time=02:00:00

for env in "simple_navigation" "simple_vision" "single_movable" "complex" "subgoal_obstructed" "attachment_obstructed"
do
    for algo in "a_star" "namo" # "vamp" "lamb" "snowplow"
    do
        for seed in "1" "2" "3" "4" "5"
        do
            if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
                python run_planner.py --algo="$algo" --env="$env" -v --save_dir="/home/gridsan/acurtis/movo_manipulation"
            fi
            i=$((i+1))
        done
    done
done