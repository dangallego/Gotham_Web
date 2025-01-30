# filament_functions.py
# Developed by: Lianys Feliciano 
# Contributions by: Daniel Gallego - added plotting functions and functions for working with dictionary structure returned from read_fils.py routine. 

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

def plot(filament_idx, filament_dict,ax,colorfil="teal"):
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
    sample=filament_dict[filament_idx]["nsamp"] 
    cords=filament_dict[filament_idx]["px,py,pz"]
    
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

def plot_2D(fillament_idx, filament_dict,L,colorfil="black",offset=0.5):
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
    

def nodes(fil_dict): 
    """
    Creates an array of all the nodes withing a filament dictionary 
    
    ==============================================================
    Paremeters: 
    
    fil_dict: The dictionary of filaments

    Returns: 
    nodes (array): A 1D array of all nodes within a filament dictionary
    
    """
    crit_points = fil_dict['critical_points']
    
    nodes= [] #empty list
    
    for i in range(len(crit_points)): #you must iterate over every critical point
        
        if crit_points[i]['cp_idx']==5: # 5 indicates nodes or peaks 
            
            nodes_temp =[crit_points[i]['px'],crit_points[i]['py'],crit_points[i]['pz']]# creating an array of x,y,z cords
            
            nodes.append(nodes_temp)
            
    nodes=np.array(nodes)
    
    #separates x,y,z arrays 
    nodes_x= nodes[:,0]
    nodes_y= nodes[:,1]
    nodes_z= nodes[:,2]
    
    return nodes     
    
def saddles(fil_dict): 
    """
    Creates an array of all the saddles withing a filament dictionary 
    
    ==============================================================
    Paremeters: 
    
    fil_dict: The dictionary of filaments

    Returns: 
    saddles (array): A 1D array of all nodes within a filament dictionary
    
    """
    crit_points = fil_dict['critical_points']
    
    saddles= [] #empty list
    
    for i in range(len(crit_points)): #you must iterate over every critical point
        
        if crit_points[i]['cp_idx']==2: # 2 identifies saddles 
            
            saddles_temp =[crit_points[i]['px'],crit_points[i]['py'],crit_points[i]['pz']] # creating an array of x,y,z cords
            
            saddles.append(sadles_temp)
            
    saddles=np.array(saddles)
    
    #separates x,y,z arrays
    saddles_x= saddles[:,0]
    saddles_y= saddles[:,1]
    saddles_z= saddles[:,2]
    
    return saddles
    
    
def bifurcation(fil_dict):
    """
    Creates an array of all the bifurcation points withing a filament dictionary 
    
    ==============================================================
    Paremeters: 
    
    fil_dict: The dictionary of filaments

    Returns: 
    bif (array): A 1D array of all nodes within a filament dictionary
    
    """
    crit_points = fil_dict['critical_points']
    
    bif= [] #empty list
    
    for i in range(len(crit_points)): #you must iterate over every critical point
        
        if crit_points[i]['cp_idx']==4: # 4 identifies bifurcation points
            
            bif_temp =[crit_points[i]['px'],crit_points[i]['py'],crit_points[i]['pz']] # creating an array of x,y,z cords
            
            bif.append(bif_temp)
            
    bif=np.array(bif)
    
    #separates x,y,z arrays
    bif_x= bif[:,0]
    bif_y= bif[:,1]
    bif_z= bif[:,2]
    
    return bif
def ploting_crit(filament_dict,ax):
    """
    Plots the all critical points of filaments in 3D 

    ----------------------------------------------
    Parameters: 
    filament_dict: filament dictionary
    ax: axis you's like to plot on
    -------------------------------------------
    Returns: 
    3D plot of critical points
    """
    critical_points= filament_dict['critical_points']
    sample =len(critical_points)
    
    px=[]
    py=[]
    pz=[]
    
    for i in range(sample):
        px1= critical_points[i]['px']
        py1= critical_points[i]['py']
        pz1= critical_points[i]['pz']
        
        px.append(px1)
        py.append(py1)
        pz.append(pz1)
            
    ax.scatter(px,py,pz,alpha=0.3,s=5,c='purple',label='Critical Points')

def plot_nodes(fil_dict,ax): 
    """
    Plots nodes onto a 3D axis
    
    Parameter
    fil_dict: filament dictionary
    ax: axis you wish to plot on
    
    Returns
    scatter plot of nodes onto axis with label
    """
    
    nodes = nodes(fil_dict)
    
    x= nodes[:,0]
    y= nodes[:,1]
    z= nodes[:,2]
    
    return ax.scatter(x,y,z,alpha=0.5,label='nodes')
    
    

def plot_saddles(fil_dict,ax): 
    """
    Plots saddles onto a 3D axis
    
    Parameter
    fil_dict: filament dictionary
    ax: axis you wish to plot on
    
    Returns
    scatter plot ofsaddles onto axis with label
    """
    
    saddles = saddles(fil_dict)
    
    x= saddles[:,0]
    y= saddles[:,1]
    z= saddles[:,2]
    
    return ax.scatter(x,y,z,alpha=0.5,label='saddles')

def plot_bif(fil_dict,ax): 
    """
    Plots saddles onto a 3D axis
    
    Parameter
    fil_dict: filament dictionary
    ax: axis you wish to plot on
    
    Returns
    scatter plot ofsaddles onto axis with label
    """
    
        
    bif = bifurcation(fil_dict)
    
    
    x= bif[:,0]
    y= bif[:,1]
    z= bif[:,2]
    
    return ax.scatter(x,y,z,alpha=0.5,label='bifurcation')
    
    
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
        For use with separating critical points (since the CP id's are given).
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

def get_tot_fils(systems,syst,filament_dict,cube_gas,rescale2phys = True,rescale2codeedge=False,switch_coords=False,cosmic=False):
    xfils = []
    yfils = []
    zfils = []
    
    for filament_idx in range(int(filament_dict['nfils'])):
        nsamp = filament_dict['filaments'][filament_idx]['nsamp']
        
        pxs,pys,pzs = [],[],[]
        for i in range(nsamp):
            positions = filament_dict['filaments'][filament_idx]['px,py,pz']
            px_,py_,pz_ = positions[i][0],positions[i][1],positions[i][2]
            
            if rescale2phys:
                
                #get filaments to code edges 
                zibest,hx,hy,hz,xmin,xmax, ymin, ymax, zmin, zmax = get_gas_coords(cube_gas,systems=systems,syst=syst,verbose=False,code_HAGN=True)
                
                if switch_coords:
                    pxr = rescale_to_1d_code_edges(px_,start=ymin,end=ymax)
                    pyr = rescale_to_1d_code_edges(py_,start=xmin,end=xmax)
                    pzr = rescale_to_1d_code_edges(pz_,start=zmin,end=zmax)
                else:
                    pxr = rescale_to_1d_code_edges(px_,start=xmin,end=xmax)
                    pyr = rescale_to_1d_code_edges(py_,start=ymin,end=ymax)
                    pzr = rescale_to_1d_code_edges(pz_,start=zmin,end=zmax)
                
                
                px_r,py_r,pz_r = rescale_from_code_to_HAGN(pxr,pyr,pzr,aexp=0.82587326)
                pxs.append(px_r)
                pys.append(py_r)
                pzs.append(pz_r)
                
            elif rescale2codeedge:
                #get filaments to code edges 
                zibest,hx,hy,hz,xmin,xmax, ymin, ymax, zmin, zmax = get_gas_coords(cube_gas,systems=systems,syst=syst,verbose=False,code_HAGN=True)
                pxr = rescale_to_1d_code_edges(px_,start=xmin,end=xmax)
                pyr = rescale_to_1d_code_edges(py_,start=ymin,end=ymax)
                pzr = rescale_to_1d_code_edges(pz_,start=zmin,end=zmax)
                
                
                pxs.append(pxr)
                pys.append(pyr)
                pzs.append(pzr)
            
            elif cosmic:

                minfils,maxfils = get_maxmin_fils(filament_dict,cube_gas)


                #first rescale everything
                px_,py_,pz_ = positions[i][0],positions[i][1],positions[i][2]

                pxr,pyr,pzr = rescale_0_to_1(minfils,maxfils,px_,py_,pz_)

                px_nh,py_nh,pz_nh = rescale_from_code_to_NH(pxr,pyr,pzr,aexp=0.82587326)
                pxs.append(px_nh)
                pys.append(py_nh)
                pzs.append(pz_nh)

                
            else:
                pxs.append(px_)
                pys.append(py_)
                pzs.append(pz_)


        pxs = np.asarray(pxs)
        pys = np.asarray(pys)
        pzs = np.asarray(pzs)

    
        xlist = list(pxs)
        ylist = list(pys)
        zlist = list(pzs)
        

        xfils.append(xlist)
        yfils.append(ylist)
        zfils.append(zlist)

    
    return xfils,yfils,zfils 
    
    
            

            
"""            elif cosmic:
                #first rescale everything
                px_,py_,pz_ = positions[i][0],positions[i][1],positions[i][2]
                
                ext_min = 0.4152412325636912
                ext_max = 0.5847587674363087

                pxr,pyr,pzr = rescale_to_code_edges(px_,py_,pz_,start=ext_min,end=ext_max)
    
                length_of_box = ext_max-ext_min
                rsx,rsy,rsz = rescale_0_to_1(ext_min,ext_max,pxr,pyr,pzr)

                px_nh,py_nh,pz_nh = rescale_from_code_to_NH(rsx,rsy,rsz,aexp=0.82587326)
                pxs.append(px_nh)
                pys.append(py_nh)
                pzs.append(pz_nh)

"""

def get_maxmin_fils(filament_dict,cube_gas):
    
    xfils = []
    yfils = []
    zfils = []
    
    for filament_idx in range(int(filament_dict['nfils'])):
        nsamp = filament_dict['filaments'][filament_idx]['nsamp']
        
        pxs,pys,pzs = [],[],[]
        for i in range(nsamp):
            positions = filament_dict['filaments'][filament_idx]['px,py,pz']
            px_,py_,pz_ = positions[i][0],positions[i][1],positions[i][2]
            pxs.append(px_)
            pys.append(py_)
            pzs.append(pz_)

            
            
        pxs = np.asarray(pxs)
        pys = np.asarray(pys)
        pzs = np.asarray(pzs)

    
        xlist = list(pxs)
        ylist = list(pys)
        zlist = list(pzs)
        

        xfils.append(xlist)
        yfils.append(ylist)
        zfils.append(zlist)
            
            
    d_filsx_arr = np.asarray([item for sublist in xfils for item in sublist])
    d_filsy_arr = np.asarray([item for sublist in yfils for item in sublist])

    minfils = min(d_filsx_arr)
    maxfils = max(d_filsx_arr)
    
    return minfils, maxfils


