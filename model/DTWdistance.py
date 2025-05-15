import numpy as np
from fastdtw import fastdtw

class DTWDistanceCalculator:
    def __init__(self, data_array):
        self.data = data_array
        self.dtw_distances = []
        
    @staticmethod
    def normalize(series):
        return (series - np.min(series)) / (np.max(series) - np.min(series))
    
    @staticmethod
    def custom_euclidean(u, v):
        return np.sqrt(np.sum((u - v) ** 2))
    
    def calculate_distances(self, normalize_features=True):
            
        main_feature = self.data[0, :]
        
        if normalize_features:
            main_feature = self.normalize(main_feature)
        
        self.dtw_distances = []
        
        for i in range(1, self.data.shape[0]):
            other_feature = self.data[i, :]
            
            if normalize_features:
                other_feature = self.normalize(other_feature)
            
            distance, _ = fastdtw(main_feature, other_feature, dist=self.custom_euclidean)
            self.dtw_distances.append((i, distance))
            
        return self.dtw_distances
    
