# MiCro
# MiCro: Modeling Cross-Image Semantic Relationship Dependencies for Class-Incremental Semantic Segmentation in Remote Sensing Images
This is the official PyTorch implementation of our work: "MiCro: Modeling Cross-Image Semantic Relationship Dependencies for Class-Incremental Semantic Segmentation in Remote Sensing Images".

In this paper, we present a novel approach and we define a new evaluation benchmark for class-incremental semantic segmentation in remote sensing images. We assess the performance of our method and previous state-of-the-art methods on ISPRS Vaihingen and Potsdam datasets. 

# Requirements
This repository uses the following libraries:
- Python (3.6)
- Pytorch (1.2)
- torchvision (0.4.0)
- tensorboardX (1.8)
- apex (0.1)
- matplotlib (3.3.1)
- numpy (1.17.2)
- [inplace-abn](https://github.com/mapillary/inplace_abn) (1.0.7) 

We also assume to have installed pytorch.distributed package.

To facilitate your work in installing all dependencies, we provide you the requirement (requirements.txt) file.

# How to perform training
The most important file is run.py, that is in charge to start the training or test procedure.
To run it, simpy use the following command:

> python -m torch.distributed.launch --nproc_per_node=\<num_GPUs\> run.py --data_root \<data_folder\> --name \<exp_name\> .. other args ..

# Code implement
Note that we have integrated HCISS and MiCro based on the framework provided by "Modeling the Background for Incremental Learning in Semantic Segmentation" which accepted at CVPR 2020. The original code address is: https://github.com/fcdl94/MiB


The default is to use a pretraining for the backbone used, that is searched in the pretrained folder of the project. 
We used the pretrained model released by the authors of In-place ABN (as said in the paper), that can be found here:
 [link](https://github.com/mapillary/inplace_abn#training-on-imagenet-1k). 
Since the pretrained are made on multiple-gpus, they contain a prefix "module." in each key of the network. Please, be sure to remove them to be compatible with this code (simply rename them using key = key\[7:\]).
If you don't want to use pretrained, please use --no-pretrained.

There are many options (you can see them all by using --help option), but we arranged the code to being straightforward to test the reported methods.
Leaving all the default parameters, you can replicate the experiments by setting the following options.
- please specify the data folder using: --data_root \<data_root\> 
- dataset: --dataset vaihingen (ISPRS Vaihingen) | potsdam (ISPRS Potsdam)
- task: --task \<task\>, where tasks are
    - 4-1, 3-1s, 2-1s, 1s (both for ISPRS Vaihingen and Potsdam datasets)
- step (each step is run separately): --step \<N\>, where N is the step number, starting from 0
- disjoint is default setup (need to use /process_data/data_process_vaihingen.py to divide incremental settings), to enable overlapped: --overlapped
- learning rate: --lr 0.01 (for step 0) | 0.001 (for step > 0) 
- batch size: --batch_size \<24/num_GPUs\>
- epochs: --epochs 40 (for both datasets)
- method: --method \<method name\>, where names are
    - FT, LWF, LWF-MC, ILT, EWC, RW, PI, MiB, HCISS, MiCro
    
For all details please follow the information provided using the help option.

#### Example commands

LwF on the 4-1 setting of ISPRS Vaihingen, 4-1LWF.sh:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset vaihingen --name LWF --task 4-1 --step 0 --lr 0.01 --epochs 40 --method LWF

MiB on the 3-1s setting of ISPRS Vaihingen, step 2:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset vaihingen --name MIB --task 3-1s --step 2 --lr 0.001 --epochs 40 --method MiB

HCISS on 2-1s setting of ISPRS Potsdam, step 1:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset potsdam --name HCISS --task 2-1s --step 1 --lr 0.001 --epochs 40 --method HCISS

MiCro on 1s setting of potsdam, step 4:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset potsdam --name MiCro --task 1s  --step 4 --lr 0.001 --epochs 40 --method MiCro


You can also write .sh file as follow:
LWF on 1s setting of vaihingen
1s-vaihingen_LWF.sh  which contains:
> CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 14785 run.py  --batch_size 24 --dataset vaihingen --name LWF --task 1s --step 0 --lr=0.01  --epochs 40 --method LWF
> CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 14785 run.py  --batch_size 24 --dataset vaihingen --name LWF --task 1s --step 1 --lr=0.001 --epochs 40 --method LWF
> CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 14785 run.py  --batch_size 24 --dataset vaihingen --name LWF --task 1s --step 2 --lr=0.001 --epochs 40 --method LWF
> CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 14785 run.py  --batch_size 24 --dataset vaihingen --name LWF --task 1s --step 3 --lr=0.001 --epochs 40 --method LWF
> CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 14785 run.py  --batch_size 24 --dataset vaihingen --name LWF --task 1s --step 4 --lr=0.001 --epochs 40 --method LWF

Then execute the command 
> bash 1s-vaihingen_LWF.sh  

Once you trained the model, you can see the result on tensorboard (we perform the test after the whole training)
 or you can test it by using the same script and parameters but using the command 
> --test

that will skip all the training procedure and test the model on test data.

