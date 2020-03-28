# RAVE - Reweighted autoencoded variational Bayes for enhanced sampling
RAVE is a machine learning-based framework that learns the reaction coordinates, which can be used in biased MD simulation to enhance the sampling.
Please read and cite these manuscripts when using RAVE:
https://aip.scitation.org/doi/abs/10.1063/1.5025487
https://www.nature.com/articles/s41467-019-11405-4
https://arxiv.org/abs/2002.06099

Overviews of each of the subdirectories in this repository are provided below. For detailed descriptions of the files within the subdirectories, refer to the README's within each subdirectory.

## P_rave.py

This file contains the main part of RAVE that uses a deep neural network to learn the reaction coordinates as linear functions of order parameters. The linear combination coefficients are written to a txt file.  

## COLVAR2npy.py

This file contains the data preprocessing part of the code. It reads the output of MD simulation and converts them into the format that can be read by P_rave.py. It will be called by P_rave.py automatically.

## Analyze_prave.py

This file contains the output analyzing part of the code. It reads the output of neural network and outputs the combination coefficients. It will be called by P_rave.py automatically. It is also useful to run this file separately in the situations that the main code is still running or stop before finishing due to running out of time on HPC. 


Usage:

```text
python P_rave.py 
```

All the parameters need to be claimed correctly inside the file.

Some important parameters include:

-system_name : the name of system. Input trajectories files are named as: system_name_<traj_index>, where <traj_index> is the index of a trajectory starts from 0.
-n_trajs: the number of trajectories
-bias: whether the trajectories are from biased MD.
-time_delay: predictive time delay.
-T: temperature in unit of Kelvin.
-op_dim: dimensionality of order parameters
-rc_dim: dimensionality of reaction coordinates
-training_size: the number of data points for training. Training size should be smaller than the total number of available data points.
-batch_size: number of data points in each batch. The total number of training data point n should be a multiple of batch_size.
-epochs: number of epochs to train the model. A rule of thumb is training_size*epochs/batch_size should be around 10^5.

By default, input trajectories are in the `input` directory and the output can be found in  `output` directory. Users should create these two directories before running the code.

  
