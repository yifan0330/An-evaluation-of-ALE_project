import numpy as np
import nibabel as nib
import nimare
from nimare.transforms import mm2vox, vox2mm
from scipy.stats import multivariate_normal
import itertools


class Foci_generator(object):
    """
    generating foci using simulation

    Parameters
    ----------
    population_center: :obj: `numpy array`
        store the coordinates of population centers
    probs: :obj: `numpy array`
        the probability of generating 0/1/2/.../n foci around each population center
    mask_file: :obj: `string`
        the file name of brain mask in the format of nii file
    sigma: :obj: `float`
        the diagonal entry in 3x3 covariance matrix
    grid_width: :obj: `int`
        side length for 3 dimensional grid
    Reference
    ---------
    Samartsidis, Pantelis, et al. "The coordinate-based meta-analysis of neuroimaging data." 
    Statistical science: a review journal of the Institute of Mathematical Statistics 32.4 (2017): 580.
    """
    def __init__(self, population_center, probs, mask_file,sigma=10,grid_width=33):
        self.population_center = population_center
        self.n_population_center =  population_center.shape[0] # number of population centers
        self.probs = probs
        self.n_probs = probs.shape[0] # possible number of foci generated from a population center
        self.sigma = sigma
        self.grid_width = grid_width
        # coordinate system
        xx,yy,zz = 91,109,91
        coordinate = []
        for element in itertools.product(np.arange(1,xx+1), np.arange(1,yy+1), np.arange(1,zz+1)):
            coordinate.append(list(element))
        self.coord = np.array(coordinate) # shape: (902629,3) by default
        
        # index for each voxel
        self.voxel_index = np.arange(self.coord.shape[0]) # shape: (902629,) by default
        
        # convert mask into the form of numpy array
        load_mask = np.array(nib.load(mask_file).dataobj)
        load_mask[load_mask>0] = 1
        self.mask = load_mask
        # affine
        self.affine = nib.load(mask_file).affine


    # normal probabilities if placed on a grid
    def sample_probability(self):
        """
        probability density of normal distribution for a grid (in 3 dimensional space)
        """
        sigma = self.sigma
        grid_width = self.grid_width
        grid = []
        for element in itertools.product(np.arange(1,grid_width+1), np.arange(1,grid_width+1), np.arange(1,grid_width+1)):
            grid.append(list(element))
        grid_coord = np.array(grid)

        # probability density function for mvnorm
        center_point = 0.5*(grid_width+1)
        mean = np.array([center_point,center_point,center_point])
        cov = np.diag(np.array([sigma]*3))
        grid_density = multivariate_normal.pdf(grid_coord, mean, cov)

        grid_density = grid_density.reshape((grid_width,grid_width,grid_width))
        # normalize
        grid_density = grid_density/np.sum(grid_density)
        return grid_density

    def _indicators(self,radius):
        """
        indicator function using 0/1 to represent if the euclidean distance to the
        central point is less than the chosen radius
        Parameters
        ----------
        radius: :obj: `float`
        """
        grid_width = self.grid_width
        tmp = np.arange(start=1,stop=grid_width+1, step=1) #np array: [1, 3, 5, ..., 65]
        tmp_coords = list()
        for element in itertools.product(tmp,tmp,tmp):
            tmp_coords.append(list(element))
        tmp_coords = np.array(tmp_coords) #shape:(35937, 3)

        length = grid_width**3 #35937 by default
        center_coord = np.array([[0.5*(grid_width+1),0.5*(grid_width+1),0.5*(grid_width+1)]]*length) #shape:(35937, 3)
        dist = np.linalg.norm(tmp_coords - center_coord, axis=1).reshape((grid_width,grid_width,grid_width))
            
        # use 0/1 to indicate if the euclidean distrance is less than the specified radius
        dist[dist < radius] = 1
        dist[dist > radius] = 0

        return dist

    
    def map_voxels(self):
        """
        probability map for sampling voxels from population centers
        """
        sigma = self.sigma
        grid_width = self.grid_width

        half_width = 0.5*(grid_width - 1)
        p = self.sample_probability()
        map_list = list()

        for i in range(self.n_population_center):
            center = self.population_center[i] # the i^th population center
            map = np.zeros((91,109,91))
            # assign the sampling probability to the region near each population center
            map[int(center[0]-half_width):int(center[0]+half_width+1), int(center[1]-half_width):int(center[1]+half_width+1), int(center[2]-half_width):int(center[2]+half_width+1)] = p
            map[self.mask==0] = 0
            map[map<=0.000005] = 0
            map = map/np.sum(map) # normalization
            
            map_list.append(map)
        return np.array(map_list)

    
    def map_noise(self):
        """
        probability map for sampling the noise voxels
        """
        map_noise = np.zeros((91,109,91))
        map_noise[self.mask==1] = 1
        map_noise = map_noise/np.sum(map_noise)
        
        return map_noise
        
    
    def areas(self, radius):
        area_indicator = self._indicators(radius) #shape: (33,33,33)
        p = self.sample_probability()
        #voxels_map = self.map_voxels() #shape:(n_population_center,91,109,91)

        areas = list()
        for i in range(self.n_population_center):
            area_tmp = np.zeros((91,109,91))
            center_coord = self.population_center[i]
            area_tmp[center_coord[0]-16:center_coord[0]+17, center_coord[0]-16:center_coord[0]+17, center_coord[0]-16:center_coord[0]+17] = area_indicator*p
            #area = area_tmp*voxels_map[i]
            areas.append(area_tmp)
        areas = np.array(areas)
        
        return areas
    


    def sample_drawer(self, n_total, n_valid):
        """
        sampling foci around population centers in the valid studies
        and/or sampling foci uniformly in the noise studies

        Parameters
        ----------
        n_total: the total number of studies
        n_valid: the number of valid studies

        Note
        ----
        The last column in the numpy array indicates the study's index that the foci are sampled
        """
        sigma = self.sigma
        grid_width = self.grid_width


        foci_coords_array = np.empty((0,4))
        
        if n_valid == 0: #foci all come from noise studies
            for i in range(n_total):
                n_cent = np.zeros((self.n_population_center))
                while np.sum(n_cent) == 0:
                    n_cent = np.random.choice(self.n_probs, size=self.n_population_center,replace=True, p=self.probs)
                # sampling noise voxels uniformly from the brain map
                noise_index = np.random.choice(self.voxel_index, size=np.sum(n_cent),replace=False,p=self.map_noise().reshape(902629))
                # get the corresponding coordinates in 3 dimensional space
                noise_coord = np.array([list(self.coord[i]) for i in noise_index]) #shape: (np.sum(n_cent), 3)
                
                # a column vector indicating which study that the focus is sampled from
                i_column_vector = np.array([i]*np.sum(n_cent)) 
                i_column_vector = i_column_vector.reshape((np.sum(n_cent),1))
                
                noise_coord = np.append(noise_coord, i_column_vector, axis=1)
                # concatenate all the sampled voxels by rows
                foci_coords_array = np.concatenate((foci_coords_array, noise_coord), axis=0)
        
        elif n_total == n_valid: #foci all come from consistent studies
            for i in range(n_total):
                n_cent = np.zeros((self.n_population_center))
                while np.sum(n_cent) == 0:
                    n_cent = np.random.choice(self.n_probs, size=self.n_population_center,replace=True, p=self.probs)
                
                foci_coords_tmp = np.empty((0,3)) # empty array to store the simulated foci coordinates for each study
                
                for j_index in range(len(n_cent)):
                    j = n_cent[j_index] # number of foci sampled from each population center
                    
                    if j > 0: 
                        p = self.map_voxels()[j_index]
                        foci_index = np.random.choice(self.voxel_index, size=j,replace=False,p=p.reshape(902629))
                        foci_coordinate = np.array([list(self.coord[i]) for i in foci_index])
                        
                        foci_coords_tmp = np.append(foci_coords_tmp, foci_coordinate, axis=0)
                    
                i_column_vector = np.array([i]*np.sum(n_cent)) 
                i_column_vector = i_column_vector.reshape((np.sum(n_cent),1))
                
                foci_coords_tmp = np.append(foci_coords_tmp, i_column_vector, axis=1)
                
                foci_coords_array = np.concatenate((foci_coords_array, foci_coords_tmp), axis=0)
                    
        else: # foci come from both consisent studies and noise studies
            for i in range(n_valid): # foci which come from consisent studies
                n_cent = np.zeros((self.n_population_center))
                while np.sum(n_cent) == 0:
                    n_cent = np.random.choice(self.n_probs, size=self.n_population_center,replace=True, p=self.probs)
                
                foci_coords_tmp = np.empty((0,3)) # empty array to store the simulated foci coordinates for each study
                
                for j_index in range(len(n_cent)):
                    j = n_cent[j_index] # number of foci sampled from each population center
                    
                    if j > 0: 
                        p = self.map_voxels()[j_index]
                        foci_index = np.random.choice(self.voxel_index, size=j,replace=False,p=p.reshape(902629))
                        foci_coordinate = np.array([list(self.coord[i]) for i in foci_index])
                        
                        foci_coords_tmp = np.append(foci_coords_tmp, foci_coordinate, axis=0)
                    
                i_column_vector = np.array([i]*np.sum(n_cent)) 
                i_column_vector = i_column_vector.reshape((np.sum(n_cent),1))
                
                foci_coords_tmp = np.append(foci_coords_tmp, i_column_vector, axis=1)
                
                foci_coords_array = np.concatenate((foci_coords_array, foci_coords_tmp), axis=0)
            
            
            for i in range(n_valid, n_total): # foci which come from noise studies
                n_cent = np.zeros((self.n_population_center))
                while np.sum(n_cent) == 0:
                    n_cent = np.random.choice(self.n_probs, size=self.n_population_center,replace=True, p=self.probs)
                # sampling noise voxels uniformly from the brain map
                noise_index = np.random.choice(self.voxel_index, size=np.sum(n_cent),replace=False,p=self.map_noise().reshape(902629))
                # get the corresponding coordinates in 3 dimensional space
                noise_coord = np.array([list(self.coord[i]) for i in noise_index]) #shape: (np.sum(n_cent), 3)
                
                # a column vector indicating which study that the focus is sampled from
                i_column_vector = np.array([i]*np.sum(n_cent)) 
                i_column_vector = i_column_vector.reshape((np.sum(n_cent),1))
                
                noise_coord = np.append(noise_coord, i_column_vector, axis=1)
                # concatenate all the sampled voxels by rows
                foci_coords_array = np.concatenate((foci_coords_array, noise_coord), axis=0)
            
        return  foci_coords_array
    
    

    
