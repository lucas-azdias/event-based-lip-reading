#!/usr/bin/env bash

# RUN TRAIN
# normal-no-compression
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=normal-no-compression

# normal-distillation
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=normal-distillation --distillation --teacher_weights=normal-no-compression

# normal-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=normal-prune --prune --skip_train --weights=normal-no-compression

# normal-distillation-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=normal-distillation-prune --distillation --teacher_weights=normal-no-compression --prune

# simple-no-compression
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=simple-no-compression --use_simple

# simple-distillation
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=simple-distillation --distillation --teacher_weights=normal-no-compression --use_simple

# simple-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=simple-prune --prune --skip_train --weights=simple-no-compression --use_simple

# simple-distillation-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=simple-distillation-prune --distillation --teacher_weights=normal-no-compression --prune --use_simple


# RUN TEST
# normal-no-compression
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=normal-no-compression --use_profiler --reps=5

# normal-distillation
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=normal-distillation --use_profiler --reps=5

# normal-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=normal-prune --use_profiler --reps=5

# normal-distillation-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=normal-distillation-prune --use_profiler --reps=5

# simple-no-compression
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=simple-no-compression --use_simple --use_profiler --reps=5

# simple-distillation
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=simple-distillation --use_simple --use_profiler --reps=5

# simple-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=simple-prune --use_simple --use_profiler --reps=5

# simple-distillation-prune
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --weights=simple-distillation-prune --use_simple --use_profiler --reps=5