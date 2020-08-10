#!/bin/bash

# tversky-BCE
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss BCE --threshold 0.3 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss BCE --threshold 0.4 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss BCE --threshold 0.5 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss BCE --threshold 0.6 --fp16

# tversky-tversky

python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss tversky --threshold 0.3 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss tversky --threshold 0.4 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss tversky --threshold 0.5 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss tversky --mantra-loss tversky --threshold 0.6 --fp16

# BCE-BCE
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss BCE --mantra-loss BCE --threshold 0.3 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss BCE --mantra-loss BCE --threshold 0.4 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss BCE --mantra-loss BCE --threshold 0.5 --fp16
python3 train_threshold.py --iter 50 --epoch 100 --aspp-loss BCE --mantra-loss BCE --threshold 0.6 --fp16
