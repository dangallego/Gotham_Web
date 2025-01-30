# read_fil functions: #intialization file and functions/classes created by Janvi Madhani 
# Developed by: Janvi Madhani 
'''
Extensive module created by Janvi Madhani to read in files from DisPerSE into Python as dictionaries. 
'''
import os
from os.path import join
import re
import numpy as np
import time as time
from scipy.io import FortranFile
import ctypes as c
import struct


class ReadFilament2D:
    def __init__(self,file_path=None):
        
        """
        Make a filaments dictionary out of ASCII NDSKL file
        """
        self.file_path = file_path
        self.filament_dict = None
        self.read_data()
   

        
    def read_data(self):
        t0 = time.time()
        self.filament_dict = {}

        #read the file first and write each line into data
        data = []
        f = open(self.file_path,'r')
        for line in f:
            data.append(line)
        f.close()
        
        def convert_to_list(ascii_chars,type=float):
            #strip whitespace from ends
            ascii_chars = str(ascii_chars)

            char_list = list(ascii_chars.split(" "))
            char_list = ' '.join(char_list).split()
            

            p_list = list(map(type, char_list))
            
            return p_list
        
        header1 = data[0]
        print('header1,',header1)
        
        ndims = data[1]
        print('ndims,', ndims)

        comments = data[2]
        print('Comments,',comments)

        extent = data[3]
        print('Bounding box,', extent)

        #data[4] is str(Critical Points)

        ncrit = int(data[5])
        print('ncrit,', ncrit)
        self.filament_dict['ncrit'] = ncrit

        #store all data for critical points in here
        self.filament_dict['critical_points'] = []

        ##### CPs
        
        add_to_idx = 6 
        for i in range(ncrit):
            cp_dict = {}
            i = 0
            
            i += add_to_idx #make sure you are at the right line in the data list 
            critical_vals = data[i]
            
            c_idx, px, py,  value, pairID, boundary = convert_to_list(critical_vals)
            #next line in data

            cp_dict['cp_idx']  = c_idx
            cp_dict['px'] = px 
            cp_dict['py'] = py
            cp_dict['pair_ID'] = pairID
            cp_dict['boundary'] = boundary 


            i += 1
            nfil = int(data[i])
            cp_dict['nfil'] = nfil
            cp_dict['destID,filID'] = []
            
            for k in range(nfil):

                i += 1
                cp_on_fil = data[i]
                destID, filID = convert_to_list(cp_on_fil,int)
        
                cp_dict['destID,filID'].append([destID,filID])

            #make this to fill out later
            cp_dict['Field Vals'] = []
            #add all info to cp dict
            self.filament_dict['critical_points'].append(cp_dict)
            
            add_to_idx = i + 1


        ##### Filaments

        fil_idx = i + 1
        nfils = int(data[fil_idx+1])
        self.filament_dict['nfils'] = nfils
        print('nfils,', nfils)

        #store all data for filaments in here
        self.filament_dict['filaments'] = []

        fil_add = fil_idx+2
        for i in range(nfils):
            i = 0
            fil_dict = {}
            
            i += fil_add #make sure you are at the right line in the data list 
            fil_info = data[i]
            
            cp1_idx, cp2_idx, nsamp = convert_to_list(fil_info)
            nsamp = int(nsamp)
            
            fil_dict['cp1_idx'] = cp1_idx
            fil_dict['cp2_idx'] = cp2_idx
            fil_dict['nsamp'] = nsamp
            fil_dict['px,py'] = []

            
            for k in range(nsamp):

                i += 1
                positions = data[i]
                px,py= convert_to_list(positions)
                #print('px,py,pz:',px,py,pz)
                fil_dict['px,py'].append([px,py])
            
            #make this to fill out later
            fil_dict['Field Vals'] = []

            #add filament info to dict
            self.filament_dict['filaments'].append(fil_dict)
            fil_add = i + 1

        cp_dat_idx = i + 1


        #Field Data
        print('Reading data fields:')
        nb_cp_dat_fields = int(data[cp_dat_idx+1])
        cp_dat_add = cp_dat_idx+2
        self.filament_dict['nb_CP_fields'] = nb_cp_dat_fields
        self.filament_dict['CP_fields'] = []

        for i in range(nb_cp_dat_fields):
            i = 0
            i += cp_dat_add #make sure you are at the right line in the data list 
            cp_field_info = data[i]
            print('CP field:',cp_field_info)
            self.filament_dict['CP_fields'].append(cp_field_info)
            
            cp_dat_add = i + 1

        cp_field_val_idx = i + 1 

        cp_val_add = cp_field_val_idx
        cp_field_vals = []
        for i in range(ncrit):
            i = 0
            i += cp_val_add #make sure you are at the right line in the data list 
            cp_field_val_info = data[i]
            list_of_cp_vals = convert_to_list(cp_field_val_info)
            cp_field_vals.append(list_of_cp_vals)
            
            cp_val_add = i + 1
            
        fil_dat_idx = i + 1  

        #put the field vals in the right place in the dictionary
        for j in range(ncrit):
            self.filament_dict['critical_points'][j]['Field Vals'] = cp_field_vals[j]
        
        nb_fil_dat_fields = int(data[fil_dat_idx+1])
        self.filament_dict['nb_fil_fields'] = nb_fil_dat_fields
        self.filament_dict['fil_fields'] = []
        fil_dat_add = fil_dat_idx+2
        for i in range(nb_fil_dat_fields):
            i = 0
            i += fil_dat_add #make sure you are at the right line in the data list 
            fil_field_info = data[i]
            print('Filament field:',fil_field_info)
            self.filament_dict['fil_fields'].append(fil_field_info)
            
            fil_dat_add = i + 1

        fil_field_val_idx = i + 1  

        fil_val_add = fil_field_val_idx

        fil_field_vals = []
        for i in range(nfils):
            i = 0
            i += fil_val_add #make sure you are at the right line in the data list 
            fil_field_val_info = data[i]
            list_of_fil_vals = convert_to_list(fil_field_val_info)
            fil_field_vals.append(list_of_fil_vals)
            
            fil_val_add = i + 1
        #put the field vals in the right place in the dictionary
        for j in range(nfils):
            self.filament_dict['filaments'][j]['Field Vals'] = fil_field_vals[j]
    

    
        t1 = time.time()
        print('Reading filaments took {:0.2f} secs.'.format(t1 - t0))
            
        return self.filament_dict
        
        
   
    
class ReadFilament:
    def __init__(self,file_path=None):
        
        """
        Make a filaments dictionary out of ASCII NDSKL file
        """
        self.file_path = file_path
        self.filament_dict = None
        self.read_data()
   

        
    def read_data(self):
        t0 = time.time()
        self.filament_dict = {}

        #read the file first and write each line into data
        data = []
        f = open(self.file_path,'r')
        for line in f:
            data.append(line)
        f.close()
        
        def convert_to_list(ascii_chars,type=float):
            #strip whitespace from ends
            ascii_chars = str(ascii_chars)

            char_list = list(ascii_chars.split(" "))
            char_list = ' '.join(char_list).split()
            

            p_list = list(map(type, char_list))
            
            return p_list
        
        header1 = data[0]
        print('header1,',header1)
        
        ndims = data[1]
        print('ndims,', ndims)

        comments = data[2]
        print('Comments,',comments)

        extent = data[3]
        print('Bounding box,', extent)

        #data[4] is str(Critical Points)

        ncrit = int(data[5])
        print('ncrit,', ncrit)
        self.filament_dict['ncrit'] = ncrit

        #store all data for critical points in here
        self.filament_dict['critical_points'] = []

        ##### CPs
        
        add_to_idx = 6 
        for i in range(ncrit):
            cp_dict = {}
            i = 0
            
            i += add_to_idx #make sure you are at the right line in the data list 
            critical_vals = data[i]
            
            c_idx, px, py, pz, value, pairID, boundary = convert_to_list(critical_vals)
            #next line in data

            cp_dict['cp_idx']  = c_idx
            cp_dict['px'] = px 
            cp_dict['py'] = py
            cp_dict['pz'] = pz
            cp_dict['pair_ID'] = pairID
            cp_dict['boundary'] = boundary 


            i += 1
            nfil = int(data[i])
            cp_dict['nfil'] = nfil
            cp_dict['destID,filID'] = []
            
            for k in range(nfil):

                i += 1
                cp_on_fil = data[i]
                destID, filID = convert_to_list(cp_on_fil,int)
        
                cp_dict['destID,filID'].append([destID,filID])

            #make this to fill out later
            cp_dict['Field Vals'] = []
            #add all info to cp dict
            self.filament_dict['critical_points'].append(cp_dict)
            
            add_to_idx = i + 1


        ##### Filaments

        fil_idx = i + 1
        nfils = int(data[fil_idx+1])
        self.filament_dict['nfils'] = nfils
        print('nfils,', nfils)

        #store all data for filaments in here
        self.filament_dict['filaments'] = []

        fil_add = fil_idx+2
        for i in range(nfils):
            i = 0
            fil_dict = {}
            
            i += fil_add #make sure you are at the right line in the data list 
            fil_info = data[i]
            
            cp1_idx, cp2_idx, nsamp = convert_to_list(fil_info)
            nsamp = int(nsamp)
            
            fil_dict['cp1_idx'] = cp1_idx
            fil_dict['cp2_idx'] = cp2_idx
            fil_dict['nsamp'] = nsamp
            fil_dict['px,py,pz'] = []

            
            for k in range(nsamp):

                i += 1
                positions = data[i]
                px,py,pz = convert_to_list(positions)
                #print('px,py,pz:',px,py,pz)
                fil_dict['px,py,pz'].append([px,py,pz])
            
            #make this to fill out later
            fil_dict['Field Vals'] = []

            #add filament info to dict
            self.filament_dict['filaments'].append(fil_dict)
            fil_add = i + 1

        cp_dat_idx = i + 1


        #Field Data
        print('Reading data fields:')
        nb_cp_dat_fields = int(data[cp_dat_idx+1])
        cp_dat_add = cp_dat_idx+2
        self.filament_dict['nb_CP_fields'] = nb_cp_dat_fields
        self.filament_dict['CP_fields'] = []

        for i in range(nb_cp_dat_fields):
            i = 0
            i += cp_dat_add #make sure you are at the right line in the data list 
            cp_field_info = data[i]
            print('CP field:',cp_field_info)
            self.filament_dict['CP_fields'].append(cp_field_info)
            
            cp_dat_add = i + 1

        cp_field_val_idx = i + 1 

        cp_val_add = cp_field_val_idx
        cp_field_vals = []
        for i in range(ncrit):
            i = 0
            i += cp_val_add #make sure you are at the right line in the data list 
            cp_field_val_info = data[i]
            list_of_cp_vals = convert_to_list(cp_field_val_info)
            cp_field_vals.append(list_of_cp_vals)
            
            cp_val_add = i + 1
            
        fil_dat_idx = i + 1  

        #put the field vals in the right place in the dictionary
        for j in range(ncrit):
            self.filament_dict['critical_points'][j]['Field Vals'] = cp_field_vals[j]
        
        nb_fil_dat_fields = int(data[fil_dat_idx+1])
        self.filament_dict['nb_fil_fields'] = nb_fil_dat_fields
        self.filament_dict['fil_fields'] = []
        fil_dat_add = fil_dat_idx+2
        for i in range(nb_fil_dat_fields):
            i = 0
            i += fil_dat_add #make sure you are at the right line in the data list 
            fil_field_info = data[i]
            print('Filament field:',fil_field_info)
            self.filament_dict['fil_fields'].append(fil_field_info)
            
            fil_dat_add = i + 1

        fil_field_val_idx = i + 1  

        fil_val_add = fil_field_val_idx

        fil_field_vals = []
        for i in range(nfils):
            i = 0
            i += fil_val_add #make sure you are at the right line in the data list 
            fil_field_val_info = data[i]
            list_of_fil_vals = convert_to_list(fil_field_val_info)
            fil_field_vals.append(list_of_fil_vals)
            
            fil_val_add = i + 1
        #put the field vals in the right place in the dictionary
        for j in range(nfils):
            self.filament_dict['filaments'][j]['Field Vals'] = fil_field_vals[j]
    

    
        t1 = time.time()
        print('Reading filaments took {:0.2f} secs.'.format(t1 - t0))
            
        return self.filament_dict
