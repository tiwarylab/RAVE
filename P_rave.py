"""RAVE: using predicitve inforamtion bottleneck framework to learn RCs 
to enhance the sampling of MD simulation. Code maintained by Yihang.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/1.5025487
https://www.nature.com/articles/s41467-019-11405-4
https://arxiv.org/abs/2002.06099
"""
import numpy as np
import COLVAR2npy
import Analyze_prave
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.initializers import RandomUniform, Constant
from keras.optimizers import RMSprop
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import Callback
from keras.losses import mean_squared_error

########################

### Global Functions ###
def data_prep(system_name, number_trajs, predictive_step):
    """ Read the input trajectory files.
        Prepare x, x_t trajectory and corresponding reweighting factors
        
        Parameters
        ----------
        system_name : string
            Name of the sytem.
            
        number_trajs : int
            Number of trajectories.
        
        predictive_step : int
            Predictive time delay.
        
        Returns
        -------
        X : np.array
            present trajectory.
         
        Y : np.array
            future trajectory.
            
        W1 : np.array
            reweighting factores in objective function before P(X_t | \chi )
            
        W2 : np.array
            reweighting factores in objective function before P(X | \chi )
    """
    
    
    for j in range(number_trajs):
        traj_file_name = 'input/x_'+system_name+'_%i.npy'%j   #present trajecotry of the shape n*d, where n is the MD steps and d is the number of order parameters
        w_file_name = 'input/w_'+system_name+'_%i.npy'%j      #weights correspond to trajecotry in x. Calculated by exp(beta*V)   
        if predictive_step==0:
            x = np.load(traj_file_name)
            y = x[:,:]      
            w1 =  np.load(w_file_name)   
            w2 = np.zeros( np.shape(w1) )        
        else:
            x = np.load(traj_file_name)
            y = x[predictive_step: , :]
            x = x[:-predictive_step, :]
            w = np.load(w_file_name)
            w_x = w[:-predictive_step]
            w_y = w[predictive_step:]            
            w1 = ( w_x * w_y )**0.5
            w2 =  w_x**0.5*( w_x**0.5- w_y**0.5)
        try:
            X = np.append(X, x, axis = 0)
            Y = np.append(Y, y, axis = 0)
            W1 = np.append(W1, w1, axis = 0)
            W2 = np.append(W2, w2, axis = 0)
        except:
            X = x
            Y = y
            W1 = w1
            W2 = w2
    normaliztion_factor = np.sum(W1)/len(W1)  
    W1 /= normaliztion_factor
    W2 /= normaliztion_factor    
    print('length of data:%i'%np.shape(X)[0] )
    print('number of order parameters:%i'%np.shape(X)[1] )
    print('min reweighting factor:%f'%np.min(W1))
    print('max reweighting factor:%f'%np.max(W1))   
    return X, Y, W1, W2

def random_pick(x, x_dt, w1, w2, training_len):
    """ ramdomly pick (x, x_dt) pair from data set
    
        Parameters
        ----------
        x : np.array
            present trajectory.
         
        x_dt : np.array
            future trajectory.
            
        w1 : np.array
            reweighting factores in objective function before P(X_t | \chi )
            
        w2 : np.array
            reweighting factores in objective function before P(X | \chi )
            
        training_len: int
            length of the return data set
            
        
        Returns
        -------
        x1 : np.array
            ramdonly selected data pionts from present trajectory.
         
        x2 : np.array
            future trajectory corresponds to selected data points in x1.
            
        w1 : np.array
            coressponding reweighting factores in objective function before P(X_t | \chi )
            
        w1 : np.array
            coressponding reweighting factores in objective function before P(X | \chi )
    """   
    
    indices = np.arange( np.shape(x)[0])   
    np.random.shuffle(indices)
    indices = indices[:training_len]
    x = x[indices, :]
    x_dt = x_dt[indices, :]
    w1 = w1[indices]
    w2 = w2[indices]
    print('%i data points are used in this training'%len(indices))
    
    return x, x_dt, w1, w2

def scaling(x):
    """ make order parametes with mean 0 and variance 1
        return new order parameter and scaling factors
        
        Parameters
        ----------
        x : np.array
            order parameters
            
        Returns
        ----------
        x : np.array
            order parameters after rescaling
        
        std_x : np.array
            resclaing factors of each OPs
              
     """ 
   
    x = x-np.mean(x, axis =0)
    std_x = np.std(x, axis =0)
    return x/std_x, std_x

def sampling(args):
	"""Sample the latent variable
	from a Normal distribution."""
	s_mean= args
	epsilon = K.random_normal(shape=(batch_size,rc_dim), mean=0.0, stddev=s_vari )
	s_noise = s_mean +  epsilon
	return s_noise


def dynamic_correction_loss(x, w1, w2):
    """loss function with dynamic correction"""
    def custom_loss(y_true, y_pred ):
         ce1 = mean_squared_error(y_true, y_pred )
         ce2 = mean_squared_error(x, y_pred)  
         return (w1[:,0]*ce1+w2[:,0]*ce2)    
    return custom_loss

class WeightsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses_vali = []
        self.weights0 = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.losses_vali.append(logs.get('val_loss'))
        self.weights0.append( prave.layers[1].get_weights())
    

#########################

if __name__ == '__main__':
    
    ### Global Variables ###
    #system info   
    system_name = '6e1u_1'
    n_trajs = 4    #number of trajectories
    save_path = 'output/'        #pth to the directory that saves output files
    T = 300     #Temperature in unit of Kelvin 
    bias = True #When false reweigting factors are set to 1. 
                #When true, reweigting factors are calculated and save 
    
    ###predictive time delay ###
    time_delay= list(range(0, 100, 10)) #predictive time delay
    
    #network variables
    training_size = 10240000  # if training_size = n, only n data points will be randomly piked from the whole data set and used to do the training
    batch_size = 2048  #total number of training data point n should be a multiple of batch_size 
    op_dim = 3  #dimensionality of order parameters
    rc_dim = 2   #dimensionality of reaction coordinates 
    int_dim = 128  #number of cells in each layer
    s_vari = 0.005 
    learning_rate = 0.0002 
    decay = 0.0       
    trials = range(4) 
    epochs = 20 #Number of epochs to train the model
    random_uniform = RandomUniform(minval=-0.05, maxval=0.05) 
    set_constant = Constant(value = 0.5**0.5)
    if_whiten = True
    
    #convert COLVAR file to npy file
    for traj_index in range(n_trajs):
        COLVAR2npy.COLVAR2npy( system_name+'_%i'%traj_index, T, op_dim, 'input/', bias )
      
    ### set predictive time delay ###
    if not bias:
        system_name = 'unbiased_' + system_name       
   
    ######################## 
    for dt in time_delay:      
        ######################## 
        ### load the dataset ###
        (x, y, w1, w2) = data_prep( system_name, n_trajs, dt )
        if if_whiten:
            x, scaling_factors = scaling(x) 
            y -= np.mean( y, axis =0)
            y /= scaling_factors
        else:
            scaling_factors = np.ones( op_dim )
        ############################   
        ### run different trials ###
        for trial in trials:     
            Result = []
            ############################################  
            ### Variational Autoencoder architecture ###
            input_Data = Input(batch_shape=(batch_size, op_dim))  
            input_w1 = Input(shape=(1,))    
            input_w2 = Input(shape=(1,))  
            linear_encoder = Dense( rc_dim, activation=None, use_bias=None,  kernel_regularizer=regularizers.l1(0.0), kernel_initializer='random_uniform',  kernel_constraint = unit_norm(axis=0))(input_Data)
                   
            s = Lambda(sampling)(linear_encoder)
            hidden_a = Dense(int_dim, activation='elu', kernel_initializer='random_uniform')(s)
            hidden_b = Dense(int_dim, activation='elu', kernel_initializer='random_uniform')(hidden_a)
            y_reconstruction = Dense( op_dim, activation=None, kernel_initializer='random_uniform')(hidden_b)
            
            ######################################### 
            ### Randomly pick samples from dataset ###
            #data for training
            train_x, train_y, train_w1, train_w2 = random_pick(x, y, w1, w2,training_size)
            #data for validation
            vali_x, vali_y, vali_w1, vali_w2 = random_pick(x , y, w1, w2, training_size)
           
            #############################################
            ### Prepare the PRAVE and train the PRVAE ###
            prave = Model([input_Data, input_w1 , input_w2] ,y_reconstruction)            
            rmsprop = RMSprop(lr=learning_rate, decay = decay)
            prave.compile(optimizer=rmsprop,loss=dynamic_correction_loss(input_Data, input_w1, input_w2))
            history = WeightsHistory()
            History = prave.fit( [train_x,train_w1,train_w2], train_y,
        	    shuffle=True,
        	    epochs=epochs,
        	    batch_size=batch_size,
            validation_data=([vali_x,vali_w1,vali_w2], vali_y),
            callbacks = [history ] )
                    
            ####################
       
            ### Save results ###
            Loss = np.array(  history.losses  )
            Val_Loss = np.array(  history.losses_vali  )          
            Weights0=np.array( history.weights0 )[:,0,:,:] 
            
            #w_norm = np.linalg.norm(Weights0,  axis=1)
            for op_index in range( op_dim ):
                Weights0[:,op_index,:]/=scaling_factors[op_index] #recale back to rc weights of non-whitenting ops
            for rc_index in range( rc_dim ):
                Weights0[:, :, rc_index]= np.transpose( np.transpose( Weights0[:, :, rc_index] )/np.linalg.norm(Weights0[:, :, rc_index], axis=1)) #normalize the rc weights
                 
            Loss = np.expand_dims(Loss, axis=-1)
            Val_Loss = np.expand_dims(Val_Loss, axis=-1)
            result_loss = np.concatenate((Loss, Val_Loss) , axis =-1)
            result_weights = Weights0           
            K.clear_session()
            print('!!!!')
            print(np.shape(result_weights))
            network_info = '_int_dim'+str(int_dim)+'_lr'+str(learning_rate)+'_decay'+str(decay)+'_batch_size'+str(batch_size)
            save_info = system_name+'_dt'+str(dt)+'_trail'+str(trial)+'_svar'+str(s_vari)+'_train_size'+str(training_size)+network_info 
            np.save(save_path+'Loss_'+save_info, result_loss)
            np.save(save_path+'Weights_'+save_info, result_weights)
    
    ### analyze and svae the results ###
    Analyze_prave.save_result(system_name, op_dim, time_delay, trials, s_vari, training_size, network_info, save_path)  
        

