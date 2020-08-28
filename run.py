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
n_total1,  n_total2 = 25, 100
n_valid1, n_valid2 = 10, 10

foci_generator = Foci_generator(population_center, probs, mask_file, sigma, grid_width)


# draw the coordinates of foci
foci_coords1 = foci_generator.sample_drawer(n_total1, n_valid1)
foci_coords2 = foci_generator.sample_drawer(n_total2, n_valid2)
# convert numpy array into the form of `nimare.dataset`
dataset1 = foci_dataset(foci_coords1)
dataset2= foci_dataset(foci_coords2)

areas = foci_generator.areas(radius=17)

meta1 = nimare.meta.cbma.ALE(sample_size=25)
meta2 = nimare.meta.cbma.ALE(sample_size=25)
meta1.fit(dataset1)
meta2.fit(dataset2) 

masker = meta1.dataset.masker
#transform: Generate ALE modeled activation images for each Contrast in dataset
ma_maps1 = meta1.kernel_transformer.transform(
            meta1.inputs_['coordinates'],
            masker=masker,
            return_type='image'
        )
ma_maps2 = meta2.kernel_transformer.transform(
            meta2.inputs_['coordinates'],
            masker=masker,
            return_type='image'
        )

n_grp1 = len(ma_maps1) #25
ma_maps = ma_maps1 + ma_maps2
id_idx = np.arange(len(ma_maps)) #from 0 to 124

# Get MA values for both samples.
ma_arr = masker.transform(ma_maps) #shape: (125,352328)
n_voxels = ma_arr.shape[1] #352328

# Get ALE values for first group.
grp1_ma_arr = ma_arr[:n_grp1, :] #shape: (25, 352328)
grp1_ale_values = np.ones(n_voxels) # 352328
for i_exp in range(grp1_ma_arr.shape[0]):
    grp1_ale_values *= (1. - grp1_ma_arr[i_exp, :])
grp1_ale_values = 1 - grp1_ale_values #shape: (352328,)

# Get ALE values for second group.
grp2_ma_arr = ma_arr[n_grp1:, :]
grp2_ale_values = np.ones(n_voxels)
for i_exp in range(grp2_ma_arr.shape[0]):
    grp2_ale_values *= (1. - grp2_ma_arr[i_exp, :])
grp2_ale_values = 1 - grp2_ale_values #shape: (352328,)

p_arr = np.ones(n_voxels)

diff_ale_values = grp1_ale_values - grp2_ale_values

n_iters=10000
iter_diff_values = np.zeros((n_iters, n_voxels)) #shape: (10000, 352328)

for i_iter in range(n_iters):
    np.random.shuffle(id_idx)
    
    iter_grp1_ale_values = np.ones(n_voxels)
    for j_exp in id_idx[:n_grp1]:
        iter_grp1_ale_values *= (1. - ma_arr[j_exp, :])
    iter_grp1_ale_values = 1 - iter_grp1_ale_values

    iter_grp2_ale_values = np.ones(n_voxels)
    for j_exp in id_idx[n_grp1:]:
        iter_grp2_ale_values *= (1. - ma_arr[j_exp, :])
    iter_grp2_ale_values = 1 - iter_grp2_ale_values

    iter_diff_values[i_iter, :] = iter_grp1_ale_values - iter_grp2_ale_values

for voxel in range(n_voxels):
    p_arr[voxel] = null_to_p(diff_ale_values[voxel],
                            iter_diff_values[:, voxel],
                            tail='two')
#diff_signs = np.sign(diff_ale_values - np.median(iter_diff_values, axis=0))
#z_arr = p_to_z(p_arr, tail='two') * diff_signs

images = {
    'p': p_arr
    }




#ale_subtraction = ALESubtraction(n_iters=1000)
#result = ale_subtraction.fit(ale1, ale2)



"""
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

"""
n_total1_list = [120]*11
n_total2_list = [120]*11
n_valid1_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120]
n_valid2_list = [10]*11

for i in range(1):
    n_total1 = n_total1_list[i]
    n_total2 = n_total2_list[i]
    n_valid1 = n_valid1_list[i]
    n_valid2 = n_valid2_list[i]

    file_name = 'n_total120' + 'n_valid' + str(n_valid1)+'&' + str(n_valid2)

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
    np.save(file=file_name+'_p-value', arr=center_p_array)
    np.save(file=file_name+'_general_activation', arr=general_activation_array)
    np.save(file=file_name+'_false_positive', arr=fp_array)
    print('---------------------------------------------------')


#a = np.load('two_sample_results/n_total110&120n_valid10_p-value.npy')
#print(a)


#print('P-value at each population center', center_p)
#print('number of voxels with P-value less than 0.05 around each population center', general_activation)
#fp = ((corrected_p_mask<0.05)&(corrected_p_mask>0)).sum() - sum(general_activation)
#print('false positive',fp)
"""
