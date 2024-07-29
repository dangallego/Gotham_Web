"""
README: 
This is a formal class to insert all gas cube related functions

It is very much a living document and will surely need to be expanded upon

Credit: Lianys (compiled methods into the class and contributed methods)
        Janvi  (contributed many methods that were later adapted for this class)
        Sneha  (condributed methods that were later adapted for this class )
"""

import os
from os.path import join
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import ctypes as c
import struct 
from scipy.ndimage import gaussian_filter

class Cubes:
    """

    This class serves to interact and do numerical conversions with simulated data cubes from New Horizon and New Horison AGN

    """
    def __init__(self,file_path=None):
        
        self.cube, self.sizes = self.read(file)
        

    def read(self,file_path):
        """
        Reads in frotran file cube and returns crutial information about the gas cube
        ------------------------------------------------------------
        Parameters

        file: the file or file path of desired cube
    
        ------------------------------------------------------------
        Returns

        cube: a 3D arrayt of values from the 
        sizes: the length width and depth sizes of your cube
        """
    
        f = FortranFile(file_path,'r') #reading in as fortran file
    
        sizes = f.read_record('i') #getting the sizes of each axis 
    
        cube =f.read_reals(dtype='f4').reshape((sizes[0],sizes[1],sizes[2]),order='F') #for fortran files 
    
        return cube,sizes
    
    def smooth(self,sigma= 5):
        """
        Uses gaussian smoothing to gas cube. The default is a 5 sigma smooth but can be edited.
        ---------------------------------------------------------------------------------------
        Parameters 
        file: Fortran file of gas cube 
        sigma: the amount of smoothing (default set to 5)
        ---------------------------------------------------------------------------------------
        Returns
        smooth_gas(np.array): gas after gaussian smoothing
    
        """
        smooth_gas= gaussian_filter(self.cube, sigma)
        return smooth_gas 
    
    def center_pixle(self):
        """""
        Find the center coordinates of each pixel in the 3D image.
        =============================================================
        Parameters:
        image (numpy.ndarray): 3D array representing the image.

        =============================================================
        Returns:
        numpy.ndarray: Array of shape (N, 3) containing the center coordinates of each pixel.

        """""

        height, width, depth = self.shape
        #so each of the these lines starts with the center of a pixel and increments by 1
        x = np.arange(0.5, width, 1)
        #print(x)
        y = np.arange(0.5, height, 1)
        z = np.arange(0.5, depth, 1)
        #forming 3, 3D coordinate arrays
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        #print(xv)
        #flatten takes an array and puts everything in one dimension
        #column_stack takes these 1-D arrays and stacks them as columns to make a 2D array
        centers = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))
        return centers


