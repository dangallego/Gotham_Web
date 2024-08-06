import numpy as np 
import pandas as pd 
from scipy import spatial
import pandas as pd
import sys
import read_fils as rf

class Filament: 
    def __init__(self, path_to_filament_NDskl):
        self.path_to_filament_NDskl = path_to_filament_NDskl
        self.filament_dict = self.import_fils()


    def import_fils(self):
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


    def filament_coordinates(self):
        '''
        Creates a Pandas DataFrame of filament coordinates, with ID's corresponding to each half segment 
        (Node -> Saddle Point).

        Returns:
        df (Pandas.DataFrame): DataFrame of filament px py pz coordinates and Filament ID. 
        '''
        df = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = self.filament_dict['filaments'][i]['px,py,pz']
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz'])
            p['Filament ID'] = i
            df = pd.concat([df, p])
        return df
    
    
    def segment_coordinates(self):
        '''
        Creates a Pandas DataFrame of filament segments and their coordinates and Filament ID's. 
        Each segment consists of two points, corresponding to (px,py,pz) and (px2,py2,pz2). 
        Compared to the Filament_Coordinates method, this will result in one less row per Filament ID (due to the "picket fence" problem).

        Returns: 
        fil_segments (Pandas.DataFrame): DataFrame of filament segments for each Filament ID. 
        '''
        fil_segments = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = np.array(self.filament_dict['filaments'][i]['px,py,pz'])
            p = np.hstack([p[:-1],p[1:]])
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz','px2','py2','pz2'])
            p['Filament ID'] = i
            fil_segments = pd.concat([fil_segments, p])
        return fil_segments
    

    def get_cp(self, cp_id):
        '''
        Slices filament_dm_dict based on a specified critical point ID.
        CRITICAL POINT ID's:
        1: Voids
        2: Walls
        3: Saddle Points
        4: Nodes
        5: Bifurcation Points

        Returns: 
        cp_list (dict): Dictionary of a specified critical point type.
        '''
        critical_points = self.filament_dict['critical_points']
        N = len(critical_points)
        cp_list = [] 
        for i in range(N):
            C = critical_points[i]['cp_idx']
            if C == cp_id:
                cp_list.append(critical_points[i])
        return cp_list 
    
    def get_cp_coordinates(self, cp_id):
        '''
        Slices filament_dm_dict based on a specified critical point ID.
        CRITICAL POINT ID's:
        1: Voids
        2: Walls
        3: Saddle Points
        4: Nodes
        5: Bifurcation Points

        Returns: 
        cp_array (array-like): Array of coordinates corresponding to a specified critical point type.
        '''
        critical_points = self.filament_dict['critical_points']
        N = len(critical_points)
        cp_list = [] 
        for i in range(N):
            C = critical_points[i]['cp_idx']
            if C == cp_id:
                cp_list.append([critical_points[i]['px'],critical_points[i]['py'],critical_points[i]['pz']])
        cp_array = np.array(cp_list)
        return cp_array    
    
    
    def voids(self):
        return self.get_cp(0)
    
    def void_coordinates(self):
        return self.get_cp_coordinates(0)
    
    
    def walls(self):
        return self.get_cp(1)
    
    def wall_coordinates(self):
        return self.get_cp_coordinates(1)
    
    
    def saddles(self):
        return self.get_cp(2)
    
    def saddle_coordinates(self):
        return self.get_cp_coordinates(2)
    
    
    def nodes(self):
        return self.get_cp(3)
    
    def node_coordinates(self):
        return self.get_cp_coordinates(3)
    
    
    def bifurcation_points(self):
        return self.get_cp(4)
    
    def bifurcation_coordinates(self):
        return self.get_cp_coordinates(4)
    
    
    def lengths(self, DataFrame=True):
        '''
        Calculates the lengths of each (half) filament based on the sub-segments of each unique Filament ID. 
        Gives option of returning DataFrame with filament segments, IDs and lengths or just the lengths corresponding to each unique Filament ID.

        Parameters:
        DataFrame (bool): Determines whether DataFrame of filament segments, IDs and lengths will be returned. Set to False to return array instead.

        Returns:
        segments (Pandas.DataFrame): DataFrame of Filament segments, IDs and lengths. 
            or
        filament_lengths (array-like): Array of lengths correspinding to each unique Filament ID. 
        '''
        segments = self.segment_coordinates()
        # For ease/speed we convert to Numpy arrays 
        segment_array = np.array(segments)

        # Calculate the distance between segments for each Filament ID; note that no ID is stored 
        L = len(segment_array)
        segment_distances = np.zeros(L)
        for i in range(L):
            segment_distances[i] = np.linalg.norm((segment_array[i][:3]) - (segment_array[i][3:-1]))
            
        # Check how many unique Filament ID's there are
        filament_ids = np.array(segments["Filament ID"])
        unique_ids = np.unique(filament_ids)
                        
        # Next loop over the unique Filament IDs and sum distances between segments for each Filament ID
        M = len(unique_ids) # length of unique ids, from 0 to 7340
        filament_lengths = np.zeros(M)
        for i in range(M): 
            ID_mask = np.where(filament_ids == i)  # masks entire filament length array by filament ID (which should == iterator step)
            filament_lengths[i] = np.sum(segment_distances[ID_mask]) # sums all of the non-masked values from iterator check, saves to i-th value of new array

        if DataFrame == True:
            # Save Filament Lengths for each Filament ID (necessary step given that DataFrame has multiple rows of the same ID's)
            N = len(filament_ids) # larger array 
            P = len(filament_lengths) # smaller array 
            all_filament_lengths = np.zeros(N)
            for i in range(P): 
                ID_mask = np.where(filament_ids == i)
                all_filament_lengths[ID_mask] = filament_lengths[i]
            
            # Add columnn to DataFrame
            segments['Filament Length'] = all_filament_lengths

            return segments
        
        else:
            # Instead of entire DataFrame, return only the Filament Lengths corresponding to each unique Filament ID
            return filament_lengths
        

    def densities(self, type='linear'):
        '''
        CURRENTLY WIP 

        Calculate filament densities based on catalog of galaxies and filaments.

        WIP! COME BACK TO ADD GENERALIZATION TO MAKE THIS WORK WITH DENSITIES FROM SIMULATIONS PERHAPS?
        ATTENTION : ASSUMES DATAFRAME OF GALAXIES (GAMA) ARE PAIRED WITH FILAMENT PROPERTIES AS WELL (currently not implemented) - should this have option to join galaxies to filaments? 
                    Could check if galaxies are joined with some sort of: if lengths_df['CATAID'] == True check
                    If doesn't exist we could have a method to join DataFrames (catalog of galaxies and filaments) to join them together 

        Relies on calculating filament lengths (method above) + potentially galaxy distances + having 5nn density category in GAMA catalog?
        '''
        lengths_df = self.lengths(DataFrame=True)

        if type == 'linear':
            IDs = np.unique(lengths_df['Filament ID']) # unique filament IDs
            N = len(IDs)
            filament_densities = np.zeros(N)
            filament_lengths = np.zeros(N)

            # LINEAR DENSITY: # of galaxies / length of filament
            for i in range(N): 
                idx = lengths_df[lengths_df['Filament ID'] == IDs[i]] # gives DataFrame of all rows with the same filament ID
                filament_lengths[i] = idx['Filament Length'].iloc[0]
                filament_densities[i] = np.sum(len(idx)) # sums/counts all of the instances/galaxies for each "i-th" filament ID
            
            lengths_df['Filament Linear Densities'] = filament_densities / filament_lengths

            return lengths_df
        





    ### WIP DISTANCE ALGORITHM BELOW ### 


    def distance(self, filament_dm_dict, data=None, data_type=0, knn=10, x=None, y=None, z=None):
        '''
        Distance algorithm.
        ===============================================================================================================
        Parameters:
        filament_dm_dict (dictionary) : dictionary of filament values obtained from Janvi's 'read_fils' workflow. 
        data (array_like, *optional) : array or dataframe of points you want to find closest filaments to.
        data_type (int) : (for future implementation) 0 means galaxies, 1 means cube, in which case conversion will have to take place.
        knn (int) : number of k-nearest neighbors that KD Tree should look for for each point. Value is 10 by default.
        x, y, z (array_like, optional): separate arrays of x, y, and z coordinates. Used if `data` is not provided.

        Returns: 
        shortest_distances (array_like) : array of shortest distances between filaments (and segments) from filament_dm_dict and data points (ie. galaxies or pixels).
        '''
        ###### FILAMENT DATAFRAME CREATION ######
        # Convert filament dictionary into separate DataFrame of filament segments 
        filament_segments_df = pd.DataFrame()
        for i in range(self.filament_dict['nfils']):
            p = np.array(self.filament_dict['filaments'][i]['px,py,pz'])
            p = np.hstack([p[:-1],p[1:]]) # Removes final element of first parts and first element of second (to address 'picket fence' problem)
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz','px2','py2','pz2']) # Adds the second pair of points (the 'end' part of the extremity)
            p['Filament ID'] = i
            filament_segments_df = pd.concat([filament_segments_df, p])


        #### DATA INPUT DECISION ( DATAFRAME or X,Y,Z separate coordinates ) ####
        # Check if x, y, z are provided and data is None, then construct the data array
        if data is None and None not in (x, y, z):
            data = np.vstack((x, y, z)).T  # Transpose to get the shape (N, 3)
        elif data is not None:
            # Convert data to array if it is a DataFrame
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
        else:
            raise ValueError("Either 'data' must be provided or all of x, y, z arrays must be provided.")


        #### KD TREE IMPLENTATION ####
        # Need to check all points, but also need index arrays to be same size as filament segment arrays (if we use total filaments for KD Tree the indices go higher than for segments)

        # Indices and distances corresponding to first set of filament segment points
        filament_segment1 = filament_segments_df[['px','py','pz']]  # First 3 columns: px, py, pz  (removes filament_ID)
        distance1, kdt_indices1 = spatial.KDTree(filament_segment1).query(data, k=knn)
        # Indices and distances for second set of filament segment points (together these cover all filament points)
        filament_segment1 = filament_segments_df[['px2','py2','pz2']]    
        distance2, kdt_indices2 = spatial.KDTree(filament_segment1).query(data, k=knn)
        # Concatenate the distances and indices and then sort to find the top 10
        combined_distances = np.hstack((distance1, distance2))
        combined_indices = np.hstack((kdt_indices1, kdt_indices2))
        # Each galaxy now has 20 potential neighbors (10 from each set), we sort and select the top 10
        knn_shortest_distances = np.argpartition(combined_distances, kth=knn, axis=1)[:, :knn]

        N = len(knn_shortest_distances)  # Number of points
        # Array of the indices corresponding to closest indices of filament values
        filament_indices = np.zeros((N, knn), dtype=int)
        for i in range(N): 
            # For each point, select the indices corresponding to the # knn smallest distances
            filament_indices[i] = combined_indices[i, knn_shortest_distances[i]]


        ##### DISTANCE CALCULATION / LOOP #####
        # Convert DataFrame to array for faster calculation 
        filament_segments = np.array(filament_segments_df)

        # Initialize array of the shortest distances for each point
        shortest_distances = np.zeros(len(data))

        # Distance calculation: for each "point" in our data, do the following
        for point in range(len(data)):
            # Initialize distance array, which should be the same size as the number of knn indices we are checking
            distances = np.zeros(len(filament_indices[0]))
            # Initialize array to hold closest filament points by indexing the total filament array by knn indices for that "point" iteration
            closest_filament_points = filament_segments[filament_indices[point]]

            # Nested loop to check distance to each closest filament point (determined by # knn)
            for i in range(knn):
                # Extract points A and B
                A = closest_filament_points[i, :3]
                B = closest_filament_points[i, 3:-1]  # Adjust indices according to your filament_segments structure
                AB = B - A
                G = data[point]  # Data point we are checking (galaxy or pixel's x,y,z coordinates)
                AG = G - A

                # Create unit vector
                u = AB / np.linalg.norm(AB)

                # Calculate distances based on geometric conditions
                if np.dot(AG, u) < 0:
                    distances[i] = np.linalg.norm(G - A)
                elif np.dot(AG, u) > np.linalg.norm(AB):
                    distances[i] = np.linalg.norm(G - B)
                else:
                    distances[i] = np.linalg.norm((np.cross((A-B), (A-G)))) / (np.linalg.norm(G-B))

            # Save the shortest distance corresponding to this "point"
            shortest_distances[point] = np.min(distances)

        # might want to add option to return the segment dataframes? 

        return shortest_distances   







    #### ONE GENERALIZED DISTANCE ALGORITHM TO RULE THEM ALL (soon, hopefully) ####

    def distance_to_filament(self, filament_dm_dict, data=None, data_type=0, knn=10, streams=None, x=None, y=None, z=None):
        '''
        Calculates closest {data} to filament distances for each {data} point. Requires user to select whether their data is a cube or array / DataFrame of galaxies. 
        Currently a WIP.
        =======================================================================================================================================================================
        Parameters:
        filament_dm_dict (dictionary) : dictionary of filament values obtained from Janvi's 'read_fils' workflow. 
        data (array_like, *optional) : array or dataframe of points you want to find closest filaments to.
        data_type (int) : (for future implementation, current WIP) select 0 for galaxies, 1 for cube, in which case conversion will have to take place.
        knn (int) : number of k-nearest neighbors that KD Tree should look for for each point. Value is 10 by default.
        x, y, z (array_like, optional): separate arrays of x, y, and z coordinates. Used if `data` is not provided.

        Returns: 
        shortest_distances (array_like) : array of shortest distances between filaments (and segments) from filament_dm_dict and data points (ie. galaxies or pixels).
        '''

        ###### FILAMENT DATAFRAME CREATION ######
        # Convert filament dictionary into separate DataFrame of filament segments 
        filament_segments_df = pd.DataFrame()
        for i in range(filament_dm_dict['nfils']):
            p = np.array(filament_dm_dict['filaments'][i]['px,py,pz'])
            p = np.hstack([p[:-1],p[1:]]) # Removes final element of first parts and first element of second (to address 'picket fence' problem)
            p = pd.DataFrame(p,columns = ['px', 'py', 'pz','px2','py2','pz2']) # Adds the second pair of points (the 'end' part of the extremity)
            p['Filament ID'] = i
            filament_segments_df = pd.concat([filament_segments_df, p])


        # Option 0 : data is array or dataframe of galaxies
        if data_type== 0: 
                #### DATA INPUT DECISION ( DATAFRAME or X,Y,Z separate coordinates ) ####
            # Check if x, y, z are provided and data is None, then construct the data array
            if data is None and None not in (x, y, z):
                data = np.vstack((x, y, z)).T  # Transpose to get the shape (N, 3)
            elif data is not None:
                # Convert data to array if it is a DataFrame
                if isinstance(data, pd.DataFrame):
                    data = data.to_numpy()
            else:
                raise ValueError("Either 'data' must be provided or all of x, y, z arrays must be provided.")
            
        
        # Option 1: Data is raw cube 
        elif data_type== 1: # one coresponds with a gas cube

                total_pixel= len(data)

                H0 = 0.703000030517578e2
                h = H0/100
                d1 = 100 #starting size of box [Mpc]

                lbox = d1/h #comoving distance (for physical, you multiply by aexp)

                #lbox_hagn = lbox*aexp
                lbox_hagn = 117.981895 # length of box of new horizons agn

                x_pixle=np.linspace(0,1,total_pixel)

                y_pixle=np.linspace(0,1,total_pixel)

                z_pixle=np.linspace(0,1,total_pixel)

                x=np.abs((x_pixle -0.5)*(lbox/total_pixel))

                y=np.abs((y_pixle -0.5)*(lbox/total_pixel))

                z=np.abs((z_pixle- 0.5)*(lbox/total_pixel))

                data = np.vstack((x, y, z)).T


        #### KD TREE IMPLENTATION ####
        # Need to check all points, but also need index arrays to be same size as filament segment arrays (if we use total filaments for KD Tree the indices go higher than for segments)

        # Indices and distances corresponding to first set of filament segment points
        filament_segment1 = filament_segments_df[['px','py','pz']]  # First 3 columns: px, py, pz  (removes filament_ID)
        distance1, kdt_indices1 = spatial.KDTree(filament_segment1).query(data, k=knn)
        # Indices and distances for second set of filament segment points (together these cover all filament points)
        filament_segment1 = filament_segments_df[['px2','py2','pz2']]    
        distance2, kdt_indices2 = spatial.KDTree(filament_segment1).query(data, k=knn)
        # Concatenate the distances and indices and then sort to find the top 10
        combined_distances = np.hstack((distance1, distance2))
        combined_indices = np.hstack((kdt_indices1, kdt_indices2))
        # Each galaxy now has 20 potential neighbors (10 from each set), we sort and select the top 10
        knn_shortest_distances = np.argpartition(combined_distances, kth=knn, axis=1)[:, :knn]

        N = len(knn_shortest_distances)  # Number of points
        # Array of the indices corresponding to closest indices of filament values
        filament_indices = np.zeros((N, knn), dtype=int)
        for i in range(N): 
            # For each point, select the indices corresponding to the # knn smallest distances
            filament_indices[i] = combined_indices[i, knn_shortest_distances[i]]


        ##### DISTANCE CALCULATION / LOOP #####
        # Convert DataFrame to array for faster calculation 
        filament_segments = np.array(filament_segments_df)

        # Initialize array of the shortest distances for each point
        shortest_distances = np.zeros(len(data))

        # Distance calculation: for each "point" in our data, do the following
        for point in range(len(data)):
            # Initialize distance array, which should be the same size as the number of knn indices we are checking
            distances = np.zeros(len(filament_indices[0]))
            # Initialize array to hold closest filament points by indexing the total filament array by knn indices for that "point" iteration
            closest_filament_points = filament_segments[filament_indices[point]]

            # Nested loop to check distance to each closest filament point (determined by # knn)
            for i in range(knn):
                # Extract points A and B
                A = closest_filament_points[i, :3]
                B = closest_filament_points[i, 3:-1]  # Adjust indices according to your filament_segments structure
                AB = B - A
                G = data[point]  # Data point we are checking (galaxy or pixel's x,y,z coordinates)
                AG = G - A

                # Create unit vector
                u = AB / np.linalg.norm(AB)

                # Calculate distances based on geometric conditions
                if np.dot(AG, u) < 0:
                    distances[i] = np.linalg.norm(G - A)
                elif np.dot(AG, u) > np.linalg.norm(AB):
                    distances[i] = np.linalg.norm(G - B)
                else:
                    distances[i] = np.linalg.norm((np.cross((A-B), (A-G)))) / (np.linalg.norm(G-B))

            # Save the shortest distance corresponding to this "point"
            shortest_distances[point] = np.min(distances)    



        # might want to add option to return the segment dataframes? 
        return shortest_distances