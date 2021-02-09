#!/bin/bash/
N=$1

# BELOW CODE IS TO REPRODUCE THE ODDS RESULTS
#declare -a datalist=("vowels" "pima" "letter" "cardio" "arrhythmia" "musk" "mnist" "satimage-2" "satellite" "mammography" "thyroid" "annthyroid" "ionosphere" "pendigits"  "glass" "shuttle")
for data in "${datalist[@]}"
  do
    for missing_ratio in 0.0
      do
                  for seed in 0 1 2 3 4
                      do
                         python3 trainVAE.py --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001  --max_epochs 100  --hidden_dim 128 --z_dim 10
                         if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_IF.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001 &
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 trainLOF.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001 &
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_oneclasssvm.py  --data $data --missing_ratio $missing_ratio --seed $seed --training_ratio 0.6 --validation_ratio 0.001&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_DAGMM.py  --data $data --seed $seed --batch_size 128 --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001  --max_epochs 100&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_SOGAAL.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001 --batch_size 128 --max_epochs 20 --z_dim 10&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_vae_coteaching.py  --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001  --max_epochs 200  --hidden_dim 128 --z_dim 10&
                        python3 TrimRealNVP.py --data $data --batch_size 128 --seed $seed --max_epochs 20 --training_ratio 0.6 --hidden_dim 256&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 RobustRealNVP.py --data $data --batch_size 128 --seed $seed --max_epochs 20 --training_ratio 0.6 --hidden_dim 256&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 RealNVP.py --data $data --batch_size 128 --seed $seed --max_epochs 20 --training_ratio 0.6 --hidden_dim 256&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 train_ae.py  --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001  --max_epochs 200  --hidden_dim 256 --z_dim 10&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                        python3 trainSVDD.py  --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.6 --validation_ratio 0.001  --max_epochs 100  --hidden_dim 128 --z_dim 10&
                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                        # wait only for first job
                          wait -n
                        fi
                      done
              done
#      done
  done


