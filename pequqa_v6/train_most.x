chemprop train \
  --remove-checkpoints \
  --data-path train_only.csv \
  --patience 40 \
  --task-type regression \
  --metric mae \
  --descriptors-path ./train_features.npz \
  --no-descriptor-scaling \
  --num-replicates 25  \
  --epochs 400 \
  --message-hidden-dim 700 \
  --depth 6 \
  --dropout 0.25 \
  --ffn-hidden-dim 2000 \
  --ffn-num-layers 2 \
  --init-lr 0.0001 \
  --final-lr 0.0000001 \
  --warmup-epochs 2 \
  --batch-size 32 \
  --split-sizes 0.7 0.1 0.2 \
  --data-seed 0 \
  --save-dir model_most/ \
  --from-foundation CheMeleon \
  --loss-function mae
chemprop predict --test-path ./test_only.csv --model-path ./model_most --preds-path ./pred_most.csv --descriptors-path ./test_features.npz --no-descriptor-scaling


