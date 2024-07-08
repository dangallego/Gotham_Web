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

def ploting(filament_idx, filament_dict,ax,colorfil="teal"):
    """
    This function will plot fillaments given the fillament index and fillament dictionary
    ======================================================================================
    Paremeter

    filament_idx: The index of each filament
    filament_dict (dictionary): The dictionary of filaments 
    ax: The axis you've created to plot the filaments on
    colorfil: the color you wish to plot (default set ot teal)

    =======================================================================================
    Returns

    A  3D   plot of filaments 

    """
    sample=filament_dict[fillament_idx]["nsamp"] 
    cords=filament_dict[fillament_idx]["px,py,pz"]
    
    #plot the samples in between
    px = []
    py = []
    pz = []

    for i in range(sample):

        px1,py1,pz1 = cords[i][0],cords[i][1],cords[i][2]
        px.append(px1)
        py.append(py1)
        pz.append(pz1)

    
    fil_line = ax.plot3D(px,py,pz,c=colorfil,lw = '2',alpha=0.4)

def plotting_2D(fillament_idx, filament_dict,L,colorfil="black",offset=0.5):
    """
    This function plots fillaments in 2D atop a gas cube centered at 0.5
    ============================================================================
    Parameters

    filament_idx: The index of each filament
    filament_dict (dictionary): The dictionary of filaments 
    ax: The axis you've created to plot the filaments on
    colorfil: the color you wish to plot (default set to black)
    offset: The offset to the cube you are plotting the filament over (default set of 0.5)

    ===========================================================================
    Returns

    A 2D plot of filaments 

    """
    sample=filament_dict[fillament_idx]["nsamp"] 
    cords=filament_dict[fillament_idx]["px,py,pz"]
    
    px= []
    py= []
    
    for i in range(sample):
        
        px1 =L*(cords[i][0]- offset)
        py1 =L*(cords[i][1]- offset)
        
        px.append(px1)
        py.append(py1)
        
    fil = plt.plot(px,py,c=colorfil,lw='2',alpha=0.5) 

def nodes(): 
    """
    This function 
    ==============================================================
    Paremeters:

    Returns: 
    nodes (array): A 1D array of all nodes within a filament dictionary
    """
    
def  saddles (): 
    """

    """

def plot_nodes(): 
    """
    """

def plot_saddles(): 
    """

    """
def length(): 
    """
    
    """
#### ROUTINES BELOW TO IMPORT DATA AUTOMATICALLY AND MAKE PLOTS
    
def import_fill(path_to_filament_NDskl):
    '''
    Imports filaments and creates dictionary containing filament coordinates and metrics
    based on a given path to filament skeleton file. 

    Parameters: 
    path_to_filament_NDskl: Path fo file containing filament skeleton (.NDskl file)

    Returns:
    filament_dm_dict (dict): Dictionary of filaments containing coordinates and other filament metrics obtained from DisPerSE. 
    '''
    # Imports the methods created by Janvi to read filaments 
    import read_fils as rf
    # Redirect to where filament .NDskl skeleton file from DisPerSE is located 
    sys.path.insert(0, path_to_filament_NDskl)
    import filament_functions as fils

    skeleton_file_dm = path_to_filament_NDskl
    filaments_dm = rf.ReadFilament(skeleton_file_dm)
    filament_dm_dict = filaments_dm.filament_dict

    #separate 'filaments' and 'critical_points' dictionaries (each is now a list of dictionaries)
    fils = filament_dm_dict['filaments'] ; crit_points = filament_dm_dict['critical_points']
    #number of filaments
    nfils = filament_dm_dict['nfils'] ; ncrit = filament_dm_dict['ncrit']

    voids = dict_slice(crit_points, 'cp_idx',0)
    walls = dict_slice(crit_points, 'cp_idx',1)
    saddles = dict_slice(crit_points, 'cp_idx',2)   #filament saddles
    peaks = dict_slice(crit_points, 'cp_idx',3)     #peaks -- nodes! 
    bi_points = dict_slice(crit_points, 'cp_idx',4) #bifurcation points

    return filament_dm_dict

        

#function to slice a dictionary list by a particular set of keys 
def dict_slice(dict, key, value):
    '''Indexes/slices a subset of a larger dictionary. 
        For use with separating critical points (since the CP id's are gven).
        Output is a list of dictionaries.'''
    N = len(dict)
    list = []
    for i in range(N):
        C = dict[i][key]
        if C == value:
            list.append(dict[i])
    return list 


#function to find saddle critical points with desired number of nfils attached
def cp_nfils(critical_points, CP_ID, nfils):
    '''Takes dicitonary of critical points and input of desired nfils.
        Returns dicitonary of desired critical_point ID type with 
        the desired number of filaments attached. 
        Mainly for use with saddles to get saddles with nfils  == 2.'''
    cp_type = dict_slice(critical_points, 'cp_idx', CP_ID)
    cp_nfils = [] ; N = len(cp_type) 

    for i in range(N): 
        nfils = nfils
        if cp_type[i]['nfil'] == nfils: 
            cp_nfils.append(cp_type[i])
    return(cp_nfils)


#function that gives the filaments with cp indices corresponding to nfil == 2
def filaments_nfil2(filament_dict, critical_points):
    '''Builds on cp_nfils array by taking the result of finding 
        saddles with nfil == 2 and using the \'filID\'s\' of each
         of those saddles to slice the original filament dictionary.
         Result is a reduced dictionary of what should be proper half filaments 
         (each filament in the dictionary corresponds to saddles with nfils == 2).'''
    
    saddles_nfils = cp_nfils(critical_points, 2, 2) #creates selective saddle dictionary with nfil == 2
    #finds index of half filaments 
    filIDs = np.zeros(len(saddles_nfils)) ; filIDs2 = np.zeros(len(saddles_nfils))
    for i, filID in enumerate(saddles_nfils):
        filIDs[i] = saddles_nfils[i]['destID,filID'][0][1]
        filIDs2[i] = saddles_nfils[i]['destID,filID'][1][1]
    fil_ids = np.append(filIDs, filIDs2) ; fil_ids = np.sort(fil_ids) ; fil_ids = (np.rint(fil_ids)).astype(int)
    fils_nfils2 = [filament_dict[i] for i in fil_ids]
    return fils_nfils2


#function to get the coordinates (if px,py, and pz are separate) for each CP type
def cp_plotter(cp):
    '''If the coordinates for a CP are separated by x,y,z 
        then this returns an array with all the values for 
        that critical point.(Perhaps too specialized - come back 
        and generalize for other use besides coordinates)'''
    N = len(cp)
    coordinates = np.zeros((N,3))
    x = 3
    for i in range(N): 
        coordinates[i,0] = cp[i]['px']
        coordinates[i,1] = cp[i]['py']
        coordinates[i,2] = cp[i]['pz']
    x = coordinates[:,0] ; y = coordinates[:,1] ; z = coordinates[:,2]
    return x, y, z




