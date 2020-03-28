"""Data preparation part of RAVE.
Code maintained by Yihang.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/1.5025487
https://www.nature.com/articles/s41467-019-11405-4
https://arxiv.org/abs/2002.06099
"""
import numpy as np 
import os.path
from os import path

def COLVAR2npy( Name, Temperature, OP_dim, Dir ='input/', Bias = False ):
    ''' Covert the COLVAR (file that cotains the trajecotry and coressponding biasing potential)
        to npy file.
        Here we assume that the COLVAR is of the shape n*(1+d1+d2+d3)
        n is the number of data points printed by plumed
        the first column is the simualtion time, usually in unit of ps
        d1 is the dimensionality of order parameters
        d2 is the dimensionality of reaction coordinates
        d3 is the number of bias potentials that is added during the simulation
        
        Parameters
        ----------
        Name : string
            Name of the system.
            
        Temperature : float
            Temperature in unit of Kelvin.
        
        OP_dim : int
            Predictive time delay.
        
        Dir: string
            Directory of input and output files.
        
        Bias: 
            Whether the trajectory is from a biased MD.
            When false reweigting factors are set to 1. 
            When true, reweigting factors are calculated and save. 
    
        Returns
        -------
        None
        
    '''  
    n_bias_convert = 1  #number of biases that will be converted into reweighting factors
    t0 = 0              #initial MD step
    total_stpes = -1    #total number of MD step, so only the date from t0 to t0+total_stpes will be saved   
    if Bias:
        traj_save_name = Dir+'x_'+Name
        bias_save_name = Dir+'w_'+Name
        if path.exists( traj_save_name+'.npy' ) and path.exists( bias_save_name+'.npy'):
            print( 'npy files already exit, delete them if you want to generate new ones.')
        else:
            Data = np.loadtxt(Dir+ Name)    
            x = Data[::, 1:OP_dim +1]   
            w = np.sum( Data[::, -n_bias_convert :], axis = 1)    
            #only a part of the full trajectory will be saved
            x = x[t0:t0+total_stpes, :]
            w = w[t0:t0+total_stpes]
            np.save(Dir+'x_'+ Name, x)
            reweihgting_factor = np.exp(w/(0.008319*Temperature))   #Here we assume that unit if bias is kJ
            np.save(Dir+'w_'+ Name, reweihgting_factor)        
    else: 
        traj_save_name = Dir+'x_unbiased_'+Name
        bias_save_name = Dir+'w_unbiased_'+Name     
        if path.exists( traj_save_name+'.npy') and path.exists( bias_save_name+'.npy'):
            print( 'npy files already exit, delete them if you want to generate new ones.')
        else:
            Data = np.loadtxt(Dir+ Name)    
            x = Data[::, 1:OP_dim +1]    
            #only a part of the full trajectory will be saved
            x = x[t0:t0+total_stpes, :]
            np.save(Dir+'x_unbiased_'+ Name, x)
            reweihgting_factor = np.ones( np.shape(x[:,0]))     # trajecotries are treated as unbiased
            np.save(Dir+'w_unbiased_'+ Name, reweihgting_factor)

if __name__ == '__main__':
    
    system_name = '6e1u_1'
    n_trajs = 4    #number of trajectories
    save_path = 'output/'        #pth to the directory that saves output files
    T = 300     #Temperature in unit of Kelvin 
    bias = True #When false reweigting factors are set to 1. 
    op_dim = 3  #dimention of order parameters
    for traj_index in range(n_trajs):
        COLVAR2npy.COLVAR2npy( system_name+'_%i'%traj_index, T, op_dim, 'input/', bias )
      
    
    
        
        
    




