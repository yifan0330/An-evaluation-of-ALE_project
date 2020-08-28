import numpy as np
import nimare
from ale import ALESubtraction
from nimare.results import MetaResult
from nimare.correct import FWECorrector
from nimare.transforms import mm2vox, vox2mm
from simulation import Foci_generator
from permutation import foci_dataset
import nibabel as nib

population_center = np.array([[63,67,50], [63,35,50], [31,73,31], [58,74,31],
                            [28,71,50], [28,38,50], [57,44,66], [34,44,66]])
n_population_center = 8                         
probs = np.array([0.35,0.50,0.10,0.05]) # probabilities of generating 0/1/2/3 foci
mask_file = 'mask.nii'

n_total = 40
n_valid = 22
sigma = 10 # results in (Eickhoff et al, 2009)
grid_width = 33

mask = nib.load(mask_file) #type: nibabel.nifti1.Nifti1Image
mask_array = np.array(mask.dataobj)
mask_array[mask_array>0] = 1 #only keep the value 0/1


# two sample comparison     
n_total1,  n_total2 = 100, 100
n_valid1, n_valid2 = 25, 10

foci_generator = Foci_generator(population_center, probs, mask_file, sigma, grid_width)
def two_sample_pipeline(n_total1, n_total2, n_valid1, n_valid2):
    # draw the coordinates of foci
    foci_coords1 = foci_generator.sample_drawer(n_total1, n_valid1)
    foci_coords2 = foci_generator.sample_drawer(n_total2, n_valid2)
    # convert numpy array into the form of `nimare.dataset`
    dataset1 = foci_dataset(foci_coords1)
    dataset2= foci_dataset(foci_coords2)

    areas = foci_generator.areas(radius=17)

    ale1 = nimare.meta.cbma.ALE(sample_size=25)
    ale2 = nimare.meta.cbma.ALE(sample_size=25)
    ale1.fit(dataset1)
    ale2.fit(dataset2) 

    ale_subtraction = ALESubtraction(n_iters=1000)
    result = ale_subtraction.fit(ale1, ale2)
    #print('ale subtraction finished')

    #corrector = (result, voxel_thresh=0.001, n_iters=10, n_cores=- 1)
    corrector = FWECorrector(method='bonferroni',n_iters=1000, voxel_thresh=0.001, n_cores=-1)
    corrected_result = corrector.transform(result)
    corrected_p_value = corrected_result.get_map(name='p', return_type='array')

    # replace the nonzero voxels in brain mask with its corrected p value
    corrected_p_mask = mask_array.copy()
    np.place(corrected_p_mask, corrected_p_mask==1, corrected_p_value)

    center_p = list()
    general_activation = list()
    for j in range(n_population_center):
        center = population_center[j]
        p = corrected_p_mask[center[0],center[1],center[2]]
        center_p.append(p)
        activation_arround_foci = ((corrected_p_mask[areas[j]>0]<0.05)&(corrected_p_mask[areas[j]>0]>0)).sum()
        general_activation.append(activation_arround_foci)
    center_p = np.array(center_p).reshape((1,8))
    general_activation = np.array(general_activation).reshape((1,8))
    fp = ((corrected_p_mask<0.05)&(corrected_p_mask>0)).sum() - sum(general_activation)
    # 'P-value at each population center', 'number of voxels with P-value less than 0.05 around each population center', 'false positive'
    return center_p, general_activation,fp


"""
n_total1_list = [20+5*i for i in range(13)]
n_total2 = 80
n_valid1 = 20
n_valid2 = 20

for i in range(13):
    n_total1 = n_total1_list[i]

    file_name = 'n_total' + str(n_total1) + '&' + str(n_total2) + 'n_valid' + str(n_valid1) + '&' + str(n_valid2)
    center_p_array, general_activation_array = np.empty((0,8)), np.empty((0,8))
    fp_list = list()
    # simulate 20 times for each parameter setting
    for i in range(5):
        simulation_result = two_sample_pipeline(n_total1, n_total2, n_valid1, n_valid2)
        center_p_array = np.concatenate((center_p_array, simulation_result[0]), axis=0)
        general_activation_array = np.concatenate((general_activation_array, simulation_result[1]), axis=0)
        fp_list.append(simulation_result[2])
    fp_array = np.array(fp_list)
    # save the results in npy file
    np.save(file='two_sample_results/total studies 20-80/' + file_name+'_p-value', arr=center_p_array)
    np.save(file='two_sample_results/total studies 20-80/' + file_name+'_general_activation', arr=general_activation_array)
    np.save(file='two_sample_results/total studies 20-80/' + file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')

print('first part finished')
"""

n_total1 = 100
n_total2 = 100
n_valid1_list = [20+5*i for i in range(17)]
n_valid2 = 20

for i in range(17):
    n_valid1 = n_valid1_list[i]

    file_name = 'n_total' + str(n_total1) + '&' + str(n_total2) + 'n_valid' + str(n_valid1) + '&' + str(n_valid2)
    center_p_array, general_activation_array = np.empty((0,8)), np.empty((0,8))
    fp_list = list()
    # simulate 20 times for each parameter setting
    for i in range(5):
        simulation_result = two_sample_pipeline(n_total1, n_total2, n_valid1, n_valid2)
        center_p_array = np.concatenate((center_p_array, simulation_result[0]), axis=0)
        general_activation_array = np.concatenate((general_activation_array, simulation_result[1]), axis=0)
        fp_list.append(simulation_result[2])
    fp_array = np.array(fp_list)
    # save the results in npy file
    np.save(file='two_sample_results/valid studies 20-100/' + file_name+'_p-value', arr=center_p_array)
    np.save(file='two_sample_results/valid studies 20-100/' + file_name+'_general_activation', arr=general_activation_array)
    np.save(file='two_sample_results/valid studies 20-100/' + file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')

print('100 finished')

n_total1 = 80
n_total2 = 80
n_valid1_list = [20+5*i for i in range(13)]
n_valid2 = 20

for i in range(17):
    n_valid1 = n_valid1_list[i]

    file_name = 'n_total' + str(n_total1) + '&' + str(n_total2) + 'n_valid' + str(n_valid1) + '&' + str(n_valid2)
    center_p_array, general_activation_array = np.empty((0,8)), np.empty((0,8))
    fp_list = list()
    # simulate 20 times for each parameter setting
    for i in range(5):
        simulation_result = two_sample_pipeline(n_total1, n_total2, n_valid1, n_valid2)
        center_p_array = np.concatenate((center_p_array, simulation_result[0]), axis=0)
        general_activation_array = np.concatenate((general_activation_array, simulation_result[1]), axis=0)
        fp_list.append(simulation_result[2])
    fp_array = np.array(fp_list)
    # save the results in npy file
    np.save(file='two_sample_results/valid studies 20-80/' + file_name+'_p-value', arr=center_p_array)
    np.save(file='two_sample_results/valid studies 20-80/' + file_name+'_general_activation', arr=general_activation_array)
    np.save(file='two_sample_results/valid studies 20-80/' + file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')