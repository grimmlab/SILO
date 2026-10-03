TASKS=(Pab1 AAV UBE2I LGK AMIE E4B GFP TEM)
SEEDS=(1 2 3 4 5)
RESULTS_DIRPATH="./results/"
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