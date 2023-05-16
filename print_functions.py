import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

def print_results(matchdata, injury_time, scale_factor, nbl, p_original, r_original, pv_original):
    
    r_final = nbl.get_r().numpy()
    p_final = nbl.get_p().numpy()
    pv1 = nbl.get_pv1().numpy()
    pv0 = nbl.get_pv0().numpy()
 
    print('p original: ', p_original)
    print('p final: ', p_final, '\n')
    
    print('r original: ', r_original)
    print('r final: ', r_final, '\n')
    
    print('pv original: ', pv_original)
    print('pv final: ', pv0, '\n')

    
    counters_mean = []
    counters_var = []
    # Mean trained, Variance trained, Mean original, Variance original 
    print('MeanTrain    VarTrain    MeanOrig    VarOrig \n')

    for i, a in enumerate(zip(r_final)):
        mean_i = mean(pv0[i]*a[0], p_final[0]) + mean(pv1[i]*a[0], p_final[1])
        var_i = var(pv0[i]*a[0], p_final[0]) + var(pv1[i]*a[0], p_final[1])
        
        mean_i_orig = mean(r_original[i], p_original[pv_original[i]])
        var_i_orig = var(r_original[i], p_original[pv_original[i]])
        
        print(f"{mean_i/scale_factor:.2f},      {var_i/(scale_factor**2):.2f},      {mean_i_orig:.2f},      {var_i_orig:.2f} ")
        
        counters_mean.append((mean_i/scale_factor))
        counters_var.append((var_i/scale_factor**2))
  
    
    mean_per_game = matchdata @ counters_mean
    var_per_game = matchdata @ counters_var
    average_mean = np.mean(mean_per_game)
    average_var = np.mean(var_per_game)
    
    print('\n Average mean of model: ', average_mean)
    print('\n Average variance of model: ', average_var)
    
   # print('mean injury time: ', np.average(injury_time)/scale_factor)
   # print('var injury time: ', np.var(injury_time)/(scale_factor**2), '\n')
   
    
    return 

def mean(r, p):
    return r*(1.0-p)/p

def var(r, p):
    return mean(r, p)/p

def mode(r, p): 
    if r>0: 
        return ((r-1) * (1-p))/ p
    else: 
        return 0
