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
sigma = 10
grid_width = 33

mask = nib.load(mask_file) #type: nibabel.nifti1.Nifti1Image
mask_array = np.array(mask.dataobj)
mask_array[mask_array>0] = 1 #only keep the value 0/1



# one sample setting
n_total = 40
n_valid = 22
sigma = 10
grid_width=33

foci_generator = Foci_generator(population_center, probs, mask_file, sigma, grid_width)
foci = foci_generator.sample_drawer(n_total, n_valid)
foci = foci_dataset(foci)

areas = foci_generator.areas(radius=17)


ale = nimare.meta.cbma.ALE(sample_size=25)
result = ale.fit(foci)


corrector = FWECorrector(method='montecarlo',n_iters=10000, n_cores=-1)
corrected_result = corrector.transform(result)
corrected_p_value = corrected_result.get_map(name='p', return_type='array') #shape:(352328,) with the given brain mask


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

print('P-value at each population center', center_p)
print('number of voxels with P-value less than 0.05 around each population center', general_activation)
fp = ((corrected_p_mask<0.05)&(corrected_p_mask>0)).sum() - sum(general_activation)
print('false positive',fp)


