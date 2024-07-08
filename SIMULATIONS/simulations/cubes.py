import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#This package are tools that for Hew Horizona dn New Horizon AGN Simulated Boxes

def read_cube(file):
    """
    Reads in frotran file cube and returns files as an array. 
    ------------------------------------------------------------
    Parameters

    ------------------------------------------------------------
    Returns

    """
    
    f = fort(file,'r') #reading in as fortran file
    
    sizes = f.read_record('i') #getting the sizes of each axis 
    
    nx,ny,nz= sizes[0],sizes[1],sizes[2] #labeling x,y,z sizes
    
    cube =f.read_reals(dtype='f4').reshape((nx,ny,nz),order='F') #for fortran files 
    
    return cube,sizes,nx,ny,nz


def scale_code_to_NH(cube,axep):
    """
    Transforms code units to NH scale
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

