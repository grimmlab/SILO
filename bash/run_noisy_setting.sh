TASKS=(E4B AMIE)
SEEDS=(1 2 3 4 5)
NOISE_LVL=(-5 -15 -25)
NOISE_MODE=TRUE 

for lvl in "${NOISE_LVL[@]}"; do
    RESULTS_DIRPATH="./results/noisy/${lvl}/"
    mkdir -p "${RESULTS_DIRPATH}"
    for TASK in "${TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "Running $TASK with seed=$SEED"
            python ./active_loop_noise.py \
                --device "cuda:0" \
                --task $TASK \
                --seed $SEED \
                --noise_mode $NOISE_MODE \
                --noise_level $lvl \
                --results "$RESULTS_DIRPATH" 
        done
    done
done