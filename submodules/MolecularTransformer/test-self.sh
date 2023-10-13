#for num in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19; do
for dataset in $(ls ../../data/mt/mtdata); do
    for model_path in $(ls ../../data/mt/${dataset}_model*.pt); do
            echo -e "$dataset\t$model_path"
            cat << SCRIPTEND > $model_path-test.sbatch.sh
#!/bin/bash
#SBATCH --account PCON0041
#SBATCH --job-name $model_path
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -c 1
#SBATCH --time 1-00:00:00
#SBATCH --gpus-per-task 1
        source ~/.bashrc
        conda activate retrosynthesis
        python translate.py -model $model_path \
                            -src ../../data/mt/mtdata/$dataset/src-test.txt \
                            -output $model_path-test_output.txt \
                            -tgt ../../data/mt/mtdata/$dataset/tgt-test.txt \
                            -n_best 5 -gpu 1 \
                            -batch_size 64 -replace_unk -max_length 200
        sleep 1
wc -l ../../data/mt/mtdata/$dataset/src-test.txt $model_path-test_output.txt
        python3 score_predictions.py \
            -targets ../../data/mt/mtdata/$dataset/tgt-test.txt \
            -beam_size=5 \
            -predictions ./$model_path-test_output.txt \
            -invalid_smiles 2>&1 > $model_path-scores.txt
SCRIPTEND
            sbatch < $model_path-test.sbatch.sh
    done
done
