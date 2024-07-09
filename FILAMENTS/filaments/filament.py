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
        self.print_filament_dict

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


        return filament_dm_dict

    def print_filament_dict(self):
        print(self.filament_dict)

        
        