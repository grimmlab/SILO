TASKS=(TEM UBE2I)
SEEDS=(1 2 3 4 5)
LOW_DATA_SETTING=True
LOW_DATA_PERC=(0.1 0.2 0.5)

for perc in "${LOW_DATA_PERC[@]}"; do
    RESULTS_DIRPATH="./results/low_data_setting/${perc}/"
    mkdir -p "${RESULTS_DIRPATH}"
    for TASK in "${TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "Running $TASK with seed=$SEED"
            python ./active_loop.py \
                --device "cuda:0" \
                --task "$TASK" \
                --seed "$SEED" \
                --results "$RESULTS_DIRPATH" \
                --low_data_setting "$LOW_DATA_SETTING" \
                --low_data_perc "$perc"
        done
    done
done