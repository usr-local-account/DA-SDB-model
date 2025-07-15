# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# In[0] max_min_params for spectral_slope
# the default_params is Time_Bimini (for other unlist test)
Time_Bimini = np.array([[-14.74468085, 40.38297872], [-12.42857143, 6.385714286],
                        [-17.20952381, 3.371428571], [-9.461538462, 20.04273504],
                        [-9.986486486, 10.47297297], [-13.12571429, 3.257142857]])
Time_GBR = np.array([[-28.53191489, 55.91489362], [-8.585714286, 3.514285714],
                        [-14.78095238, 0.438095238], [-13.44444444, 19.5982906],
                        [-8.779279279, 9.400900901], [-10.20571429, 1.36]])
Time_Dongsha = np.array([[-20.70212766, 34.42553191],[-11.3, 3.371428571],
                         [-14.23809524, -0.628571429], [-10.54700855, 11.23931624],
                         [-8.572072072, 4.288288288], [-10.46857143, 0.194285714]])
Zone_BI23_SW22 = np.array([[-14.74468085, 40.38297872], [-12.42857143, 6.385714286],
                       [-17.20952381, 3.371428571], [-9.461538462, 20.04273504],
                       [-9.986486486, 10.47297297], [-13.12571429, 3.257142857]])
Zone_DA20_BI23 = np.array([[-20.70213, 40.3882979], [-12.42857, 6.3857143],
                     [-17.20952, 3.3714286], [-10.54701, 20.042735],
                     [-9.986486, 10.472973], [-13.12571, 3.2571429]])
Zone_DA20_SW22 = np.array([[-20.70213, 34.42553],[-11.3, 3.371429],
                       [-14.74286, -0.62857],[-10.54701, 11.23932],
                       [-8.572072, 4.288288],[-10.77143, 0.194286]])
                      
class CustomDataset(Dataset):
    def __init__(self, csv_file: str, input_dim: int = 14, use_slope: bool = True):
        self.input_dim = input_dim
        self.features_data = pd.read_csv(csv_file, skiprows=0).iloc[:, 3:3+self.input_dim-1].values
        self.labels_data = pd.read_csv(csv_file, skiprows=0).iloc[:, -1].values
        self.use_slope = use_slope
        self.features_data = self.features_data.astype(np.float32)
        if np.min(self.features_data) < 500: # for early sentinel-2 version data
            normalized_data = (self.features_data-1)/(4000-1)
        else:  
            self.features_data -= 1000   # for new sentinel-2 version data
            normalized_data = (self.features_data-1)/(4000-1) 
        
        if self.use_slope:
           S4349 = (self.features_data[:, 1] - self.features_data[:, 0]) / (490 - 443)
           S4956 = (self.features_data[:, 2] - self.features_data[:, 1]) / (560 - 490)
           S5665 = (self.features_data[:, 3] - self.features_data[:, 2]) / (665 - 560)
           S4356 = (self.features_data[:, 2] - self.features_data[:, 0]) / (560 - 443)
           S4365 = (self.features_data[:, 3] - self.features_data[:, 0]) / (665 - 443)
           S4965 = (self.features_data[:, 3] - self.features_data[:, 1]) / (665 - 490)
           N_values = Time_Bimini
           N4349 = (S4349 - N_values[0,0]) / (N_values[0,1] - N_values[0,0])
           N4956 = (S4956 - N_values[1,0]) / (N_values[1,1] - N_values[1,0])
           N5665 = (S5665 - N_values[2,0]) / (N_values[2,1] - N_values[2,0])
           N4356 = (S4356 - N_values[3,0]) / (N_values[3,1] - N_values[3,0])
           N4365 = (S4365 - N_values[4,0]) / (N_values[4,1] - N_values[4,0])
           N4965 = (S4965 - N_values[5,0]) / (N_values[5,1] - N_values[5,0])
           normalized_data = np.concatenate((normalized_data, np.stack((N4349, N4956, N5665, N4356, N4365, N4965), axis=1)), axis=1)
        self.normalized_data  = normalized_data
        
    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        features = self.normalized_data[idx, :]  
        labels = self.labels_data[idx]  
        
        return features, labels
