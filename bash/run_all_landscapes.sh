TASKS=(UBE2I LGK Pab1 AMIE E4B GFP AAV TEM)
SEEDS=(1 2 3 4 5)
RESULTS_DIRPATH="./results/main"
mkdir -p "${RESULTS_DIRPATH}"

for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running $TASK with seed=$SEED"
        python ./active_loop.py \
            --device "cuda:0" \
            --task $TASK \
            --seed $SEED \
            --results $RESULTS_DIRPATH
    done
done