# KDDAnonymous
For Review Only, we will clean and publish the code if accepted.

## To reproduce all results
bash run.sh #parallel jobs can reproduce the ODDs results.\
run_cifar.sh #parallel jobs can reproduce the cifar10 results.

You can check the hyperparameter we uses in bash.sh and run_cifar.sh.

## To reproduce figure in introduction
> python3 IntroductionExp.py
## To reproduce figure for two moon experiment
> python3 TwoMoonExperiment.py

## Run RobustRealNVP
RobustRealNVP.py is RobustRealNVP for ODDS and RobustRealNVPCIFAR.py is RobustRealNVP for CIFAR. Their codes are basically the same, we will fuse them later if accpeted.\
In order to get CIFAR results, you need first run 
>python3 VGG-CIFAR10.py 

## Conda Requirements
RobustRealNVP.yml
