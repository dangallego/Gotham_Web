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
        self.filament_dict = self.import_fil()
        #self.print_filament_dict

    def import_fil(self):
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
    

    def print_filament_dict(self):
        print(self.filament_dict)




    def specific_critical_point(self, critical_point_type):

        if critical_point_type == 'voids' or 'void':
            cp = 0
        elif critical_point_type == 'walls' or 'wall':
            cp = 1
        elif critical_point_type == 'saddles' or 'saddle' or 'saddle_points':
            cp = 2
        elif critical_point_type == 'peaks' or 'peak':
            cp = 3
        elif critical_point_type == 'bifurcation_points' or 'bi_points' or 'bp':
            cp = 4
        else:
            return "Invalid critical point type. Refer to docstring for available critical point types."

        critical_points = self.filament_dict['critical_points']
        N = len(critical_points)
        cp_list = []
        for i in range(N):
            C = critical_points[i]['cp_idx']
            if C == cp:
                cp_list.append(critical_points[i])
        return cp_list


    def DataFrame(self):
        df = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = self.filament_dict['filaments'][i]['px,py,pz']
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz'])
            p['Filament ID'] = i
            df = pd.concat([df, p])
        return df
    
    
    def filament_segments(self):
        df = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = np.array(self.filament_dict['filaments'][i]['px,py,pz'])
            p = np.hstack([p[:-1],p[1:]])
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz','px2','py2','pz2'])
            p['Filament ID'] = i
            df = pd.concat([df, p])
        return df
    

    def slice(self, value):
        critical_points = self.filament_dict['critical_points']
        N = len(critical_points)
        cp_list = [] 
        for i in range(N):
            C = critical_points[i]['cp_idx']
            if C == value:
                cp_list.append(critical_points[i])
        return cp_list 
    
    
    def voids(self):
        return self.dict_slice(0)
    
    def walls(self):
        return self.dict_slice(1)
    
    def saddles(self):
        return self.dict_slice(2)
    
    def nodes(self):
        return self.dict_slice(3)
    
    def bifurcation_points(self):
        return self.dict_slice(4)



#function to slice a dictionary list by a particular set of keys 
def dict_slice(dict, key, value):
    '''Indexes/slices a subset of a larger dictionary. 
        For use with separating critical points (since the CP id's are given).
        Output is a list of dictionaries.'''
    N = len(dict)
    list = []
    for i in range(N):
        C = dict[i][key]
        if C == value:
            list.append(dict[i])
    return list 



        