import pickle
import numpy as np
import pandas as pd

def generate_indice_global(case , num_sample):
        
        num_subset,name_folder,name_dataset,_= case.get_parameters()
        indices_global = np.zeros((num_sample,num_subset))
        
        for i in range(num_subset):
                with open(f'{name_folder}/{name_dataset}indices_data_sampler_{i}_{num_subset}.pkl', 'rb') as file:
                        loaded_indices_list = np.array(pickle.load(file))      

                print(loaded_indices_list)
                indices=[(j in loaded_indices_list) for j in range(num_sample)]
                indices_global[:,i]=indices
        column =np.array([str(f"Was trained on dataset {i} : (Yes/No)")for i in range(num_subset) ])
        printed_list=pd.DataFrame(indices_global, columns=column)
        return indices_global,printed_list