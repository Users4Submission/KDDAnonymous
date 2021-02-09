#!/bin/bash/
N=$1

#  below is cifar10 0.1 anomalies DONT CHANGE THEM
for normal_class in 0 1 2 3 4 5 6 7 8 9
    do
        for seed in 0 1 2 3 4
            do
#              python3 trainIFCIFAR.py  --concentrated 1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 trainLOFCIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 trainOCSVMCIFAR.py  --concentrated 1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 trainSOGAALCIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#               python3 trainDAGMMCIFAR.py  --concentrated 1 --normal_class $normal_class  --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
               python3 trainSVDDCIFAR.py  --concentrated 0 --normal_class $normal_class  --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 trainVAECIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 trainAECIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 RobustRealNVPCIFAR.py  --concentrated 1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 RobustRealNVPCIFAR.py  --concentrated 1 --normal_class $normal_class --SN 0 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 RobustRealNVPCIFAR.py  --concentrated 1 --normal_class $normal_class --SN 0 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
              # wait only for first job
                wait -n
              fi
            done
    done


  #  below is sensitivity analysis for estimated epsilon
#for normal_class in 0 1 2 3 4 5 6 7 8 9
#    do
#        for seed in 0 1 2 3 4
#            do
##              python3 trainIFCIFAR.py  --concentrated 1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
##              python3 trainLOFCIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
##              python3 trainOCSVMCIFAR.py  --concentrated 1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
##              python3 trainSOGAALCIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
##              python3 trainVAECIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
##              python3 trainAECIFAR.py  --concentrated 0 --normal_class $normal_class --L 1.5 --SN 1 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              python3 RobustRealNVPCIFAR.py  --oe -0.05 --concentrated 0 --data_anomaly_ratio 0.1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#              python3 RobustRealNVPCIFAR.py  --oe -0.025 --concentrated 0 --data_anomaly_ratio 0.1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#              python3 RobustRealNVPCIFAR.py  --oe 0.025 --concentrated 0 --data_anomaly_ratio 0.1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#              python3 RobustRealNVPCIFAR.py  --oe 0.05 --concentrated 0 --data_anomaly_ratio 0.1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#              python3 RobustRealNVPCIFAR.py  --oe 0.1 --concentrated 0 --data_anomaly_ratio 0.1 --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#            done
#    done


#  below is sensitivity analysis for ground turth epsilon
#for normal_class in 0 1 2 3 4 5 6 7 8 9
#    do
#      for data_anomaly_ratio in 0.1 0.2 0.3 0.4
#        do
#          for seed in 0 1 2 3 4
#            do
#              python3 RobustRealNVPCIFAR.py  --oe 0 --concentrated 1 --data_anomaly_ratio $data_anomaly_ratio --normal_class $normal_class --L 1.5 --SN 1 --Trim 1 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#              python3 RobustRealNVPCIFAR.py  --oe 0 --concentrated 1 --data_anomaly_ratio $data_anomaly_ratio --normal_class $normal_class --L 1.5 --SN 0 --Trim 0 --batch_size 128 --seed $seed --max_epochs 200 --training_ratio 0.8  --hidden_dim 256&
#              if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#              # wait only for first job
#                wait -n
#              fi
#            done
#
#
#        done
#
#    done

#                        python3 train_beta_vae_pyod.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 --batch_size 128 --max_epochs 100 --z_dim 10
#                        python3 train_vae_pyod.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 --batch_size 128 --max_epochs 100 --z_dim 10&
# "vowels" "pima" "optdigits" "sensor" "letter" "cardio" "arrhythmia" "breastw" "musk" "mnist" "satimage-2" "satellite" "mammography" "thyroid" "annthyroid" "ionosphere" "pendigits" "shuttle" "glass"
#declare -a datalist=("cifar10_0.9")
#for data in "${datalist[@]}"
#  do
#    for missing_ratio in 0.1 0.2 0.3 0.4
#      do
#          for oe in 0.01 0.05 0.1 0.15 0.2
#              do
#                  for seed in 0 1 2 3 4
#                      do
#                        python3 train_ae_ct_ensemble.py  --oe $oe --batch_size 128 --data $data  --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 --max_epochs 100 --knn_impute True --hidden_dim 128 --z_dim 10&
#                        if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
#                        # wait only for first job
#                          wait -n
#                        fi
#                      done
#              done
#      done
#
#
#  done


####
#for seed in 0 1 2 3 4
#do
#  python3 train_ae.py  --data $data --batch_size 128 --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001  --max_epochs 100  --hidden_dim 128 --z_dim 10&
#done
#for seed in 0 1 2 3 4
#do
#  python3 train_IF.py  --data $data --seed $seed --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001 &
#done
##
#for seed in 0 1 2 3 4
#do
#  python3 train_oneclasssvm.py  --data $data --missing_ratio $missing_ratio --seed $seed --training_ratio 0.599 --validation_ratio 0.001 &
#done
##
#for seed in 0 1 2 3 4
#do
#  python3 train_DAGMM.py  --data $data --seed $seed --batch_size 128 --missing_ratio $missing_ratio --training_ratio 0.599 --validation_ratio 0.001  --max_epochs 200&
#done







