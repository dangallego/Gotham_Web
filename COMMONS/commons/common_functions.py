# common_functions.py
# Developed by: Daniel Gallego, Lianys Feliciano

'''
Python file containing list of functions that are useful across the entire Gotham Web project.  

'''

# Necessary imports for use in functions
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def Euclidean_Coordinates(RA, DEC, Z):
    '''
    Converts spherical coordinates RA, DEC, and Z to cartesian (x, y, z) coordinates.

    Parameters: 
    RA (float): Right Ascension.
    DEC (float): Declination.
    Z (float): Redshift.

    Returns: 
    numpy.ndarray: An array containing the x, y, and z coordinates of the point.
    '''
    H_0 = 67.8 # Hubble constant, km/s/Mpc
    c = 2.99792458e+5 # speed of light,  km/s 

    numerator = ((1+Z)**2)-1
    denominator = ((1+Z)**2)+1
    H = c/H_0
    radius = (numerator/denominator)*H    # gives radius

    for i in Z:
        d = radius
    
    RArad = RA * (np.pi/180)     # converts RA and DEC into radians 
    DECrad = DEC * (np.pi/180)

    cos = np.cos      # slightly easier than typing "np.cos" everytime
    sin = np.sin

    x = d*cos(RArad)*cos(DECrad)
    y = d*sin(RArad)*cos(DECrad)
    z = d*sin(DECrad)
    return x, y, z