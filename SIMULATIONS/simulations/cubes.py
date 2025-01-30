# cubes.py
# Developed by: 
# Contributions by: 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Circle, PathPatch
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.stats as ss
import scipy.signal as sig
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle as Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import random
from scipy.io import FortranFile 
from astropy.io import ascii
from astropy.table import Table
from scipy.ndimage import gaussian_filter

from scipy import spatial #this is what we use to implement KDTree

#This package are tools that for Hew Horizona dn New Horizon AGN Simulated Boxes


def read(file):
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
    
    f = FortranFile(file,'r') #reading in as fortran file
    
    sizes = f.read_record('i') #getting the sizes of each axis 
    
    cube =f.read_reals(dtype='f4').reshape((sizes[0],sizes[1],sizes[2]),order='F') #for fortran files 
    
    return cube,sizes


def scale_code_to_NH(x,y,z,axep):
    """
    Transforms code units to NH scale
    ----------------------------------------
    Parameters
    x,y,z: The gas cordinates 
    axep: The expansion rate at the instance the cube was extracted from
    ----------------------------------------
    Returns

    xnh:
    ynh:
    znh:
    
    """
    
    H0 = 0.703000030517578e2
    h = H0/100 
    d1 = 20 #starting size of box [Mpc]

    lbox = d1/h #comoving distance (for physical, you multiply by aexp)

    lbox_h_nh = lbox*axep
    c = lbox_h_nh/2

    
    
    xnh = x * lbox_h_nh
    xnh -= c
    
    ynh = y * lbox_h_nh
    ynh -= c
    
    znh = z * lbox_h_nh
    znh -= c
    
    return xnh,ynh,znh


def find_pixel_centers(image):
    """""
    Find the center coordinates of each pixel in the 3D image.
    =============================================================
    Parameters:
        image (numpy.ndarray): 3D array representing the image.

    =============================================================
    Returns:
        numpy.ndarray: Array of shape (N, 3) containing the center coordinates of each pixel.

    """""

    height, width, depth = image.shape
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

def write(cube,directory,save_name,integer=False):
    """
    Write a manipulated cube into a fortran file
    -----------------------------------------------------
    Parameters
    cube: smoothed, scalled or manipulate cube 
    directory: desired directory to
    save_name: name of your new saved file 
    integers: if you are writing a mask from 0-1 
    -----------------------------------------------------
    Returns
    
    """
    cube_size = np.shape(cube)
    size = np.asarray(cube_size,'i4')
    cubedata = cube.reshape(cube_size[0],cube_size[1],cube_size[2],order='F')
    f = FortranFile(directory+save_name, 'w')
    f.write_record(size[1],size[0],size[2])
    print(size)
    #if you’re writing a mask where your values are 0 or 1
    if integer:
        f.write_record(cubedata,'i8')
    else:
        f.write_record(cubedata)
    f.close()
    print('File written to:', directory+save_name)


def smooth(file,sigma= 5):
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
    rho, sizes =read(file)
    smooth_gas= gaussian_filter(rho, sigma)
    return smooth_gas 


def rescale_0_to_1(xmin,xmax,x,y,z):

    
    rs_x = (x - xmin)/(xmax - xmin)

    
    rs_y = (y - xmin)/(xmax - xmin)

    
    rs_z = (z - xmin)/(xmax - xmin)



    return rs_x,rs_y,rs_z

def rescale_to_1d_code_edges(x, start=0,end=1):
    
    #assuming code is from 0 to 1 
    #assuming code is already between [0,1]
    
    scale = (end - start)
    rs_x = (x) * scale + start

    return rs_x

def rescale_to_code_edges(x,y,z,center = 0.5, start=0,end=1):
    
    #assuming code is from 0 to 1 
    #assuming code is already between [0,1]
    
    scale = (end - start)
    rs_x = (x) * scale + start
    rs_y = (y) * scale + start 
    rs_z = (z) * scale + start 



    return rs_x,rs_y,rs_z

def rescale_0_to_1_1d(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rescale_from_code_to_HAGN(x,y,z,aexp,c = [58.99,58.99,58.99]):
    H0 = 0.703000030517578e2
    h = H0/100 
    d1 = 100 #starting size of box [Mpc]

    lbox = d1/h #comoving distance (for physical, you multiply by aexp)

    #lbox_hagn = lbox*aexp
    lbox_hagn = 117.981895
    #c = lbox_hagn/2
    
    xhagn = x * lbox_hagn
    xhagn -= c[0]
    
    yhagn = y * lbox_hagn
    yhagn -= c[1]
    
    zhagn = z * lbox_hagn
    zhagn -= c[2]
    
    return xhagn,yhagn,zhagn

def rescale_from_HAGN_to_code(x,y,z,aexp,c = [0.5,0.5,0.5]):
    #xnh, ynh, znh =0.18776, 0.42237, 0.27435 #position of nh in hagn


    H0 = 0.703000030517578e2
    h = H0/100 
    d1 = 100 #starting size of box [Mpc]

    lbox = d1/h #comoving distance (for physical, you multiply by aexp)

    #lbox_h_hagn = lbox*aexp

    lbox_h_hagn = 117.981895
    xhagn = x / lbox_h_hagn
    xhagn += c[0]
    
    yhagn = y / lbox_h_hagn
    yhagn += c[1]
    
    zhagn = z / lbox_h_hagn
    zhagn += c[2]
    
    return xhagn,yhagn,zhagn
    

def rescale_from_NH_to_code(x,y,z,aexp,HAGN=False):

    #xnh, ynh, znh =0.18776, 0.42237, 0.27435 #position of nh in hagn

    cxnh, cynh, cznh = 0.5,0.5,0.5

    
  
    H0 = 0.703000030517578e2
    h = H0/100 
    if HAGN:
        d1 = 100
        lbox_h_nh = 117.981895
    else:
        d1 = 20 #starting size of NH box [Mpc]

        lbox = d1/h #comoving distance (for physical, you multiply by aexp)

        lbox_h_nh = lbox*aexp
        #lbox_h_nh = lbox

    
    xnh = x / lbox_h_nh
    xnh += cxnh
    
    ynh = y / lbox_h_nh
    ynh += cynh
    
    znh = z / lbox_h_nh
    znh += cznh
    
    return xnh,ynh,znh

def rescale_rad_from_NH_to_code(r,aexp,HAGN=False):

  
    H0 = 0.703000030517578e2
    h = H0/100 
    if HAGN:
        d1 = 100
        lbox_h_nh = 117.981895
    else:
        d1 = 20 #starting size of box [Mpc]

        lbox = d1/h #comoving distance (for physical, you multiply by aexp)

        lbox_h_nh = lbox*aexp

    
    rcode = r / lbox_h_nh

    return rcode


def rescale_from_code_to_NH(x,y,z,aexp):

    H0 = 0.703000030517578e2
    h = H0/100 
    d1 = 20 #starting size of box [Mpc]

    lbox = d1/h #comoving distance (for physical, you multiply by aexp)

    lbox_h_nh = lbox*aexp
    c = lbox_h_nh/2


    
    xnh = x * lbox_h_nh
    xnh -= c
    
    ynh = y * lbox_h_nh
    ynh -= c
    
    znh = z * lbox_h_nh
    znh -= c
    
    return xnh,ynh,znh


def rescale_from_NH_to_HAGN(x,y,z,aexp=0.82587326):

    xnh, ynh, znh =0.18776, 0.42237, 0.27435 #position of nh in hagn

    

    H0 = 0.703000030517578e2
    h = H0/100 
    d1 = 100 #starting size of box [Mpc]

    lbox = d1/h #comoving distance (for physical, you multiply by aexp)

    lbox_h_hagn = lbox*aexp

    
    xhagn = x / lbox_h_hagn
    xhagn += xnh
    
    yhagn = y / lbox_h_hagn
    yhagn += ynh
    
    zhagn = z / lbox_h_hagn
    zhagn += znh
    
    #rhagn = r/lbox_h_hagn
    
    return xhagn,yhagn,zhagn


def redshift(a):

    z =( 1-a)/a
    
    return z

def find_min_max(x,y,z):
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    zmin = min(z)
    zmax = max(z)
    
    print('-xmi',xmin)
    print('-xma',xmax)
    print('-ymi',ymin)
    print('-yma',ymax)
    print('-zmi',zmin)
    print('-zma',zmax)
    print('length,x', xmax - xmin)
    print('length,y', ymax - ymin)
    print('length,z', zmax - zmin)


def flatten_cube(cube,axis=2):
    #flatten gas
    flat_gas = np.sum(cube,axis)
    
    return flat_gas


def get_gas_coords(cube_gas,syst,systems,code_HAGN=True,verbose=False):
    x,y,z = systems[syst]['MW_px'][0],systems[syst]['MW_py'][0],systems[syst]['MW_pz'][0]
    sx,sy,sz = systems[syst]['sat_pxs'],systems[syst]['sat_pys'],systems[syst]['sat_pzs']
    hx,hy,hz= systems[syst]['halo_px'][0],systems[syst]['halo_py'][0],systems[syst]['halo_pz'][0]
    hrvir = systems[syst]['halo_rvir']
    
    #cut a box of size n rvir around the halo
    #nrvir = 8
    nrvir = systems[syst]['nrvir']
    
    #get everything in code units
    if code_HAGN:
        #halo
        h_pxc,h_pyc,h_pzc = rescale_from_NH_to_code(hx,hy,hz,aexp_snap,HAGN=True)

        #central 
        mwxc, mwyc, mwzc = rescale_from_NH_to_code(x,y,z,aexp_snap,HAGN=True)
        
        #radius
        rcode = rescale_rad_from_NH_to_code(nrvir*hrvir,aexp_snap,HAGN=True)
    else:
        
        #halo
        h_pxc,h_pyc,h_pzc = rescale_from_NH_to_code(hx,hy,hz,aexp_snap,HAGN=False)

        #central 
        mwxc, mwyc, mwzc = rescale_from_NH_to_code(x,y,z,aexp_snap,HAGN=False)
        
        #radius
        rcode = rescale_rad_from_NH_to_code(nrvir*hrvir,aexp_snap,HAGN=False)
    
    #extents    

    ext = rcode

    #sats
    sxs_c,sys_c,szs_c = [],[],[]
    for i in range(len(sx)):
        if code_HAGN:
            sxc, syc, szc = rescale_from_NH_to_code(sx[i],sy[i],sz[i],aexp_snap,HAGN=True)
        else:
            sxc, syc, szc = rescale_from_NH_to_code(sx[i],sy[i],sz[i],aexp_snap,HAGN=False)
        sxs_c.append(sxc)
        sys_c.append(syc)
        szs_c.append(szc)
        
        
    #get the slice of gas where the halo is centered
    n = np.shape(cube_gas)[0]
    slices = np.linspace(0,n,n)
    cslices = slices/n 

    zind = (np.abs(cslices - h_pzc)).argmin()
        
    if code_HAGN:

        l_xext = h_pxc - ext[0]
        r_xext = h_pxc + ext[0]
        l_yext = h_pyc - ext[0]
        r_yext = h_pyc + ext[0]
        l_zext = h_pzc - ext[0]
        r_zext = h_pzc + ext[0]
    else:
        l_xext = h_pxc[0] - ext[0]
        r_xext = h_pxc[0] + ext[0]
        l_yext = h_pyc[0] - ext[0]
        r_yext = h_pyc[0] + ext[0]
        l_zext = h_pzc[0] - ext[0]
        r_zext = h_pzc[0] + ext[0]
        

        
    if verbose:
        if code_HAGN:
            print('Code units in HAGN scale:')
        else:
            print('Code units in NH scale:')
        print(f'System {syst}')
        print('Center of halo:', h_pxc,h_pyc,h_pzc)
        print('Center of halo in NH:', hx,hy,hz)
        print(f'{nrvir} Rvir extent for cube:' )
        #remember fortran ordering is y, x, z so give amr2cube yxz but all calculations in python remain xyz
        print('x:',f'{l_yext:.4f}', f'{r_yext:.4f}')
        print('y:',f'{l_xext:.4f}', f'{r_xext:.4f}')
        print('z:',f'{l_zext:.4f}', f'{r_zext:.4f}')
    return zind, h_pxc,h_pyc,h_pzc, float('%.4f' % l_xext), float('%.4f' % r_xext), float('%.4f' % l_yext), float('%.4f' % r_yext), float('%.4f' % l_zext),float('%.4f' % r_zext)


def get_plot_ext_phys(systems,syst,cube):
    #get extent in HAGN units
    zibest,rhx,rhy,rhz,xmin,xmax, ymin, ymax, zmin, zmax = get_gas_coords(cube,systems=systems,syst=syst,verbose=False,code_HAGN=True)


    xminh, yminh, zminh = rescale_from_code_to_HAGN(xmin,ymin,zmin,aexp=0.82587326)
    xmaxh, ymaxh, zmaxh = rescale_from_code_to_HAGN(xmax,ymax,zmax,aexp=0.82587326)
    
    return xminh, xmaxh, yminh, ymaxh, zminh, zmaxh 

