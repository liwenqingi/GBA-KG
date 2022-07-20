#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd /home/liwenqing/liwenqing_hpc/2_software/BERN2/scripts

# For Linux and MacOS
bash run_bern2.sh

# For Windows
bash run_bern2_windows.sh