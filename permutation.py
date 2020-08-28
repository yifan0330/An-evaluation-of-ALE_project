import numpy as np
import nimare
from nimare.transforms import mm2vox, vox2mm
import nibabel as nib
from simulation import Foci_generator
from random import randrange

def foci_dataset(foci_coords, mask_file='mask.nii'):
    """
    summarize foci coordinates information 
    and convert into the format of nimare dataset
    """
    mask = nib.load(mask_file) #type: nibabel.nifti1.Nifti1Image
    mask_array = np.array(mask.dataobj)
    mask_array[mask_array>0] = 1 #only keep the value 0/1
    adjusted_mask = nib.Nifti1Image(mask_array, mask.affine) # convert back to nifti1.Nifti1Image
    
    data = dict()

    for study_index in np.unique(foci_coords[:,3]):
        study_id = 'study' + str(int(study_index))
            
        study_foci_indices = foci_coords[foci_coords[:,3] == study_index]
        # convert the indices of foci to mni coordinates
        study_foci = np.array([vox2mm(ijk, affine=mask.affine) for ijk in study_foci_indices[:,:-1]])

        n_study_foci = study_foci.shape[0]

        # x,y,z coordinate of foci
        study_foci_x = list(study_foci[:,0])
        study_foci_y = list(study_foci[:,1])
        study_foci_z = list(study_foci[:,2])
        study_foci_coords = {'space':'MNI', 'x':study_foci_x, 'y':study_foci_y, 'z':study_foci_z}
            
        # set the contrast_id to be some random value
        #  but the same for all foci arising from a given map.
        contrast = randrange(n_study_foci)
        sample_size = 25 # ALE adapts the kernel size according to the sample size
        meta_data = {'sample_sizes':[sample_size]}
            
        data[study_id] = {'contrasts':{contrast:{'coords':study_foci_coords, 'metadata':meta_data}}}

    dataset = nimare.dataset.Dataset(source=data,mask=adjusted_mask)
        
    return dataset



#def two_group_permutation(data1, data2):
    #"""
    #Randomly shuffle group assignments for experiments across two groups, 
    #regardless of valid studies or noise study
    #"""
    # the number of studies in each dataset
    #n_data1_study = np.unique(data1[:,3]).shape[0]
    #n_data2_study = np.unique(data2[:,3]).shape[0]
    #n_total_study = n_data1_study + n_data2_study
    
    # the study index in dataset2 should follow dataset1
    #data2[:,3] = data2[:,3] + n_data1_study
    #total_study = np.concatenate((data1, data2), axis=0)
    
    #full_index = [i for i in range(n_total_study)]
    #chosen_index1 = list(np.random.choice(a=n_total_study, size=n_data1_study,replace=False))
    #chosen_index2 = list(set(full_index) - set(chosen_index1))
    
    #data_group1 = np.empty((0,4))
    #for study_index in chosen_index1:
        #study_foci = total_study[total_study[:,3] == study_index]
        #data_group1 = np.concatenate((data_group1, study_foci), axis=0)

    #data_group2 = np.empty((0,4))
    #for study_index in chosen_index2:
        #study_foci = total_study[total_study[:,3] == study_index]
        #data_group2 = np.concatenate((data_group2, study_foci), axis=0)

    #return data_group1, data_group2

#def ale_diff(dataset1, dataset2):
    #"""
    #Subtraction of ALE maps for given datasets
    #"""
    #ale1 = nimare.meta.cbma.ALE()
    #ale2 = nimare.meta.cbma.ALE()
    
    #ale1.fit(dataset1)
    #ale2.fit(dataset2)
    
    #ale_subtraction = nimare.meta.cbma.ALESubtraction()
    #ale_subtraction_fit = ale_subtraction.fit(ale1, ale2)
    
    #ale_diff = ale_subtraction_fit.get_map(name='z_desc-group1MinusGroup2', return_type='map')
    #return ale_diff

#def ALE_subtraction_result(foci_coords1, foci_coords2, N=50):
    
    #ALE_subtraction_array = None # np.empty((0,228453))
    #for i in range(N):
        #new_foci_coords1, new_foci_coords2 = two_group_permutation(foci_coords1, foci_coords2)
        #new_dataset1 = foci_dataset(new_foci_coords1)
        #new_dataset2 = foci_dataset(new_foci_coords2)
        #new_ale_diff = ale_diff(new_dataset1, new_dataset2).reshape(1,228453)
        #if ALE_subtraction_array is None:
        #    ALE_subtraction_array = new_ale_diff
        #else:
        #    ALE_subtraction_array = np.concatenate((ALE_subtraction_array,new_ale_diff), axis=0)
    # obtain np.array of shape (228453, N) through transpose
    #ALE_subtraction_array = ALE_subtraction_array.T
    
    # save the results in npy file
    #outfile = 'ALE_subtraction_after_permutation.npy'
    #np.save(outfile, ALE_subtraction_array)
    #return ALE_subtraction_array
