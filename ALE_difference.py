import numpy as np
import nimare
from ale import ALESubtraction
from nimare.results import MetaResult
from nimare.correct import FWECorrector
from nimare.transforms import mm2vox, vox2mm
from simulation import Foci_generator
from permutation import foci_dataset
from null_distribution import compute_null, ale_to_p
import nibabel as nib

population_center = np.array([[63,67,50], [63,35,50], [31,73,31], [58,74,31],
                            [28,71,50], [28,38,50], [57,44,66], [34,44,66]])
n_population_center = 8                         
probs = np.array([0.35,0.50,0.10,0.05]) # probabilities of generating 0/1/2/3 foci
mask_file = 'mask.nii'


sigma = 10 # results in (Eickhoff et al, 2009)
grid_width = 33
foci_generator = Foci_generator(population_center, probs, mask_file, sigma, grid_width)

mask = nib.load(mask_file) #type: nibabel.nifti1.Nifti1Image
mask_array = np.array(mask.dataobj)
mask_array[mask_array>0] = 1 #only keep the value 0/1


# two sample comparison     
#n_total1,  n_total2 = 100, 100
#n_valid1, n_valid2 = 20, 100



def ALE_difference_pipeline(n_total1, n_total2, n_valid1, n_valid2):
    # draw the coordinates of foci
    foci_coords1 = foci_generator.sample_drawer(n_total1, n_valid1)
    foci_coords2 = foci_generator.sample_drawer(n_total2, n_valid2)
    # convert numpy array into the form of `nimare.dataset`
    dataset1 = foci_dataset(foci_coords1)
    dataset2= foci_dataset(foci_coords2)

    areas = foci_generator.areas(radius=17)

    ale1 = nimare.meta.cbma.ALE(sample_size=25, n_iters=10)
    ale2 = nimare.meta.cbma.ALE(sample_size=25, n_iters=10)
    ale1.fit(dataset1)
    ale2.fit(dataset2)

    masker = ale1.dataset.masker

    ma_maps1 = ale1.kernel_transformer.transform(ale1.inputs_["coordinates"], masker=masker, return_type="image")
    ma_maps2 = ale2.kernel_transformer.transform(ale2.inputs_["coordinates"], masker=masker, return_type="image")
    
    null_distributions_ = compute_null(ma_maps1, ma_maps2)

    n_grp1 = len(ma_maps1)
    ma_maps = ma_maps1 + ma_maps2
    id_idx = np.arange(len(ma_maps))
    

    # Get MA values for both samples.
    ma_arr = masker.transform(ma_maps)
    n_voxels = ma_arr.shape[1] #352328
    
    # Get ALE values for first group.
    grp1_ma_arr = ma_arr[:n_grp1, :]
    grp1_ale_values = np.ones(n_voxels)
    for i_exp in range(grp1_ma_arr.shape[0]):
        grp1_ale_values *= 1.0 - grp1_ma_arr[i_exp, :]
    grp1_ale_values = 1 - grp1_ale_values #shape: (352328,)

    # Get ALE values for second group.
    grp2_ma_arr = ma_arr[n_grp1:, :]
    grp2_ale_values = np.ones(n_voxels)
    for i_exp in range(grp2_ma_arr.shape[0]):
        grp2_ale_values *= 1.0 - grp2_ma_arr[i_exp, :]
    grp2_ale_values = 1 - grp2_ale_values

    p_arr = np.ones(n_voxels)

    diff_ale_values = grp1_ale_values - grp2_ale_values #shape: (352328,)

    diff_p_values = ale_to_p(null_distributions_, diff_ale_values)[0]
    #diff_z_values = ale_to_p(null_distributions_, diff_ale_values)[1]

    p_mask = mask_array.copy()
    np.place(p_mask, p_mask==1, diff_p_values)

    center_p = list()
    general_activation = list()
    for j in range(n_population_center):
        center = population_center[j]
        p = p_mask[center[0],center[1],center[2]]
        center_p.append(p)
        activation_arround_foci = ((p_mask[areas[j]>0]<0.05)&(p_mask[areas[j]>0]>0)).sum()
        general_activation.append(activation_arround_foci)
    center_p = np.array(center_p).reshape((1,8))
    general_activation = np.array(general_activation).reshape((1,8))
    fp = ((p_mask<0.05)&(p_mask>0)).sum() - sum(general_activation)

    return center_p, general_activation, fp



n_total1_list = [20+5*i for i in range(21)]
n_total2 = 120
n_valid1 = 20
n_valid2 = 20

for i in range(2,21):
    n_total1 = n_total1_list[i]

    file_name = 'n_total' + str(n_total1) + '&' + str(n_total2) + 'n_valid' + str(n_valid1) + '&' + str(n_valid2)
    center_p_array, general_activation_array = np.empty((0,8)), np.empty((0,8))
    fp_list = list()
    # simulate 20 times for each parameter setting
    for i in range(10):
        simulation_result = ALE_difference_pipeline(n_total1, n_total2, n_valid1, n_valid2)
        center_p_array = np.concatenate((center_p_array, simulation_result[0]), axis=0)
        general_activation_array = np.concatenate((general_activation_array, simulation_result[1]), axis=0)
        fp_list.append(simulation_result[2])
    fp_array = np.array(fp_list)
    # save the results in npy file
    np.save(file='two_sample_results/total studies 20-120(ALE difference)/' + file_name+'_p-value', arr=center_p_array)
    np.save(file='two_sample_results/total studies 20-120(ALE difference)/' + file_name+'_general_activation', arr=general_activation_array)
    np.save(file='two_sample_results/total studies 20-120(ALE difference)/' + file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')

print('first part finished')

n_total1 = 120
n_total2 = 120
n_valid1_list = [20+5*i for i in range(21)]
n_valid2 = 20

for i in range(21):
    n_valid1 = n_valid1_list[i]

    file_name = 'n_total' + str(n_total1) + '&' + str(n_total2) + 'n_valid' + str(n_valid1) + '&' + str(n_valid2)
    center_p_array, general_activation_array = np.empty((0,8)), np.empty((0,8))
    fp_list = list()
    # simulate 20 times for each parameter setting
    for i in range(10):
        simulation_result = ALE_difference_pipeline(n_total1, n_total2, n_valid1, n_valid2)
        center_p_array = np.concatenate((center_p_array, simulation_result[0]), axis=0)
        general_activation_array = np.concatenate((general_activation_array, simulation_result[1]), axis=0)
        fp_list.append(simulation_result[2])
    fp_array = np.array(fp_list)
    # save the results in npy file
    np.save(file='two_sample_results/valid studies 20-120(ALE difference)/' + file_name+'_p-value', arr=center_p_array)
    np.save(file='two_sample_results/valid studies 20-120(ALE difference)/' + file_name+'_general_activation', arr=general_activation_array)
    np.save(file='two_sample_results/valid studies 20-120(ALE difference)/' + file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')
