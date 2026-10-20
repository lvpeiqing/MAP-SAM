#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CHAOST2'
#DATASET='SABS'
#DATASET='CMR'
NWORKER=16
RUNS=1
ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[1,2,3,4]  ##CHAOST2
#TEST_LABEL=[1,2,3,6]  ##SABS
#TEST_LABEL=[1,2,3]  ##CMR
#EXCLUDE_LABEL=[1,2,3,4] ##setting2 CHAOST2
#EXCLUDE_LABEL=[1,2,3,6] ##setting2 SABS
EXCLUDE_LABEL=None ##setting1 CHAOST2、SABS、CMR
# TEST=1234  # flag for folder name, '1234' for not excluding any labels,  '14' for excluding label [2,3], '23' for excluding label [1,4]
USE_GT=False
###### Training configs ######
NSTEP=60000   #42000   #60000   #36000  #50000 #36000 #40000
DECAY=0.98

MAX_ITER=3000 #1000 # defines the size of an epoch #3000 #1000（5000）
SNAPSHOT_INTERVAL=3000 #5000 # interval for saving snapshot #6000 #10000（5000）
SEED=2022

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}_fewshot_setting1"
  echo $PREFIX
  LOGDIR="./exps_on_${DATASET}_setting1_MAPSAM"


  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi

  python3 train.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  test_label=$TEST_LABEL \
  exclude_label=$EXCLUDE_LABEL \
  use_gt=$USE_GT \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  lr_step_gamma=$DECAY \
  path.log_dir=$LOGDIR
done