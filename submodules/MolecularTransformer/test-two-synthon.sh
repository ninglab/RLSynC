#!/bin/bash
#SBATCH --account PCON0041
#SBATCH --job-name two-synthonSS
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -c 1
#SBATCH --time 04:00:00
#SBATCH --gpus-per-task 1
source ~/.bashrc
conda activate retrosynthesis
python translate.py -model ../../data/mt/MIT_mixed_augm_model_average_20.pt \
                    -src ../../data/src-train_two_synthons-simplesynth.txt \
                    -output ../../data/train_two_synthons-output-MIT_mixed-simplesynth.txt \
                    -tgt ../../data/tgt-train_two_synthons-simplesynth.txt \
                    -n_best 5 -gpu 1 \
                    -batch_size 64 -replace_unk -max_length 200
sleep 1
python3 score_predictions.py \
    -targets ../../data/tgt-train_two_synthons-simplesynth.txt \
    -beam_size=5 \
    -predictions ../../data/train_two_synthons-output-MIT_mixed-simplesynth.txt \
    -invalid_smiles 2>&1 > ../../data/train_two_synthons-scores-MIT_mixed-simplesynth.txt
