import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import mpl_toolkits.mplot3d.art3d as art3d
import os
import pandas as pd
import sys
import read_fils as rf

class Filament:
    def __init__(self, path_to_filament_NDskl):
        self.path_to_filament_NDskl = path_to_filament_NDskl
        self.filament_dict = self.import_fils()


    def import_fils(self):
        '''
        Imports filaments and creates dictionary containing filament coordinates and metrics
        based on a given path to filament skeleton file (.NDskl file).

        Returns:
        filament_dm_dict (dict): Dictionary of filaments containing coordinates and other
        filament metrics obtained from DisPerSE.
        '''
        # Temporarily adjust the path for imports
        sys.path.insert(0, self.path_to_filament_NDskl)

        skeleton_file_dm = self.path_to_filament_NDskl
        filaments_dm = rf.ReadFilament(skeleton_file_dm)
        filament_dm_dict = filaments_dm.filament_dict

        # Optionally remove the path modification after import if no longer needed
        # sys.path.remove(self.path_to_filament_NDskl)

        return filament_dm_dict


    def filament_coordinates(self):
        '''
        Creates a Pandas DataFrame of filament coordinates, with ID's corresponding to each half segment 
        (Node -> Saddle Point).

        Returns:
        df (Pandas.DataFrame): DataFrame of filament px py pz coordinates and Filament ID. 
        '''
        df = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = self.filament_dict['filaments'][i]['px,py,pz']
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz'])
            p['Filament ID'] = i
            df = pd.concat([df, p])
        return df
    
    
    def segment_coordinates(self):
        '''
        Creates a Pandas DataFrame of filament segments and their coordinates and Filament ID's. 
        Each segment consists of two points, corresponding to (px,py,pz) and (px2,py2,pz2). 
        Compared to the Filament_Coordinates method, this will result in one less row per Filament ID (due to the "picket fence" problem).

        Returns: 
        fil_segments (Pandas.DataFrame): DataFrame of filament segments for each Filament ID. 
        '''
        fil_segments = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = np.array(self.filament_dict['filaments'][i]['px,py,pz'])
            p = np.hstack([p[:-1],p[1:]])
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz','px2','py2','pz2'])
            p['Filament ID'] = i
            fil_segments = pd.concat([fil_segments, p])
        return fil_segments
    

    def get_cp(self, cp_id):
        '''
        Slices filament_dm_dict based on a specified critical point ID.
        CRITICAL POINT ID's:
        1: Voids
        2: Walls
        3: Saddle Points
        4: Nodes
        5: Bifurcation Points

        Returns: 
        cp_list (dict): Dictionary of a specified critical point type.
        '''
        critical_points = self.filament_dict['critical_points']
        N = len(critical_points)
        cp_list = [] 
        for i in range(N):
            C = critical_points[i]['cp_idx']
            if C == cp_id:
                cp_list.append(critical_points[i])
        return cp_list 
    
    
    def voids(self):
        return self.get_cp(0)
    
    def walls(self):
        return self.get_cp(1)
    
    def saddles(self):
        return self.get_cp(2)
    
    def nodes(self):
        return self.get_cp(3)
    
    def bifurcation_points(self):
        return self.get_cp(4)
    
    
    def lengths(self, DataFrame=True):
        '''
        Calculates the lengths of each (half) filament based on the sub-segments of each unique Filament ID. 
        Gives option of returning DataFrame with filament segments, IDs and lengths or just the lengths corresponding to each unique Filament ID.

        Parameters:
        DataFrame (bool): Determines whether DataFrame of filament segments, IDs and lengths will be returned. Set to False to return array instead.

        Returns:
        segments (Pandas.DataFrame): DataFrame of Filament segments, IDs and lengths. 
            or
        filament_lengths (array-like): Array of lengths correspinding to each unique Filament ID. 
        '''
        segments = self.segment_coordinates()
        # For ease/speed we convert to Numpy arrays 
        segment_array = np.array(segments)

        # Calculate the distance between segments for each Filament ID; note that no ID is stored 
        L = len(segment_array)
        segment_distances = np.zeros(L)
        for i in range(L):
            segment_distances[i] = np.linalg.norm((segment_array[i][:3]) - (segment_array[i][3:-1]))
            
        # Check how many unique Filament ID's there are
        filament_ids = np.array(segments["Filament ID"])
        unique_ids = np.unique(filament_ids)
                        
        # Next loop over the unique Filament IDs and sum distances between segments for each Filament ID
        M = len(unique_ids) # length of unique ids, from 0 to 7340
        filament_lengths = np.zeros(M)
        for i in range(M): 
            ID_mask = np.where(filament_ids == i)  # masks entire filament length array by filament ID (which should == iterator step)
            filament_lengths[i] = np.sum(segment_distances[ID_mask]) # sums all of the non-masked values from iterator check, saves to i-th value of new array

        if DataFrame == True:
            # Save Filament Lengths for each Filament ID (necessary step given that DataFrame has multiple rows of the same ID's)
            N = len(filament_ids) # larger array 
            P = len(filament_lengths) # smaller array 
            all_filament_lengths = np.zeros(N)
            for i in range(P): 
                ID_mask = np.where(filament_ids == i)
                all_filament_lengths[ID_mask] = filament_lengths[i]
            
            # Add columnn to DataFrame
            segments['Filament Length'] = all_filament_lengths

            return segments
        
        else:
            # Instead of entire DataFrame, return only the Filament Lengths corresponding to each unique Filament ID
            return filament_lengths


        