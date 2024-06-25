import numpy as np 
import pandas as pd 


def ploting_fills(fillament_idx, filament_dict,ax,colorfil="teal"):
    """
    This function will plot fillaments given the fillament index and fillament dictionary
    """
    sample=fils[fillament_idx]["nsamp"] 
    cords=fils[fillament_idx]["px,py,pz"]
    
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

def plotting_fills_2D(fillament_idx, filament_dict,L,colorfil="black",offset=0.5):
    """
    This function plots fillaments in 2D so it can be mapped atop of gas cubes
    """
    sample=fils[fillament_idx]["nsamp"] 
    cords=fils[fillament_idx]["px,py,pz"]
    
    px= []
    py= []
    
    for i in range(sample):
        
        px1 =L*(cords[i][0]- offset)
        py1 =L*(cords[i][1]- offset)
        
        px.append(px1)
        py.append(py1)
        
    fil = plt.plot(px,py,c=colorfil,lw='2',alpha=0.5) 