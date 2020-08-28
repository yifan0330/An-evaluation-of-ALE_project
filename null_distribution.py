import numpy as np
import nimare
from ale import ALESubtraction
from nimare.results import MetaResult
from nimare.correct import FWECorrector
from nimare.transforms import mm2vox, vox2mm
from nimare.utils import round2
from nimare.transforms import p_to_z
from simulation import Foci_generator
from permutation import foci_dataset
import nibabel as nib
import multiprocessing as mp

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
n_total1,  n_total2 = 20, 100
n_valid1, n_valid2 = 20, 20

foci_coords1 = foci_generator.sample_drawer(n_total1, n_valid1)
foci_coords2 = foci_generator.sample_drawer(n_total2, n_valid2)
# convert numpy array into the form of `nimare.dataset`
dataset1 = foci_dataset(foci_coords1)
dataset2= foci_dataset(foci_coords2)

#areas = foci_generator.areas(radius=17)

ale1 = nimare.meta.cbma.ALE(sample_size=25, n_iters=10)
ale2 = nimare.meta.cbma.ALE(sample_size=25, n_iters=10)
ale1.fit(dataset1)
ale2.fit(dataset2)


masker = ale1.dataset.masker
        
ma_maps1 = ale1.kernel_transformer.transform(ale1.inputs_["coordinates"], masker=masker, return_type="image")
ma_maps2 = ale2.kernel_transformer.transform(ale2.inputs_["coordinates"], masker=masker, return_type="image")

def compute_null(ma_maps1, ma_maps2):
    ma_values1 = masker.transform(ma_maps1) #shape: (n_total1, 352328)
    ma_values2 = masker.transform(ma_maps2) #shape: (n_total2, 352328)

    # Determine histogram bins for ALE-value null distribution
    max_poss_ale1, min_poss_ale1 = 1.0, 1.0
    for i in range(ma_values1.shape[0]):
        max_poss_ale1 *= 1 - np.max(ma_values1[i, :])
        min_poss_ale1 *= 1 - np.min(ma_values1[i, :])
    max_poss_ale1 = 1 - max_poss_ale1
    min_poss_ale1 = 1 - min_poss_ale1


    max_poss_ale2, min_poss_ale2 = 1.0, 1.0
    for i in range(ma_values2.shape[0]):
        max_poss_ale2 *= 1 - np.max(ma_values2[i, :])
        min_poss_ale2 *= 1 - np.min(ma_values2[i, :])
    max_poss_ale2 = 1 - max_poss_ale2
    min_poss_ale2 = 1 - min_poss_ale2

    max_poss_ale_diff = max_poss_ale1 - min_poss_ale2
    min_poss_ale_diff = min_poss_ale1 - max_poss_ale2

    null_distributions_ = {}
    null_distributions_["histogram_bins"] = np.round(np.arange(min_poss_ale_diff-0.001, max_poss_ale1 + 0.001, 0.0001), 4) #length: 7542


    ## null ALE for the first group of experiments
    ma_hists1 = np.zeros((ma_values1.shape[0], null_distributions_["histogram_bins"].shape[0])) #shape: (20, 7542)
    for i in range(ma_values1.shape[0]):
        # Remember that histogram uses bin edges (not centers), so it
        # returns a 1xhist_bins-1 array
        n_zeros1 = len(np.where(ma_values1[i, :] == 0)[0])
        reduced_ma_values1 = ma_values1[i, ma_values1[i, :] > 0]
        
        ma_hists1[i, 5902] = n_zeros1
        selector1 = [x for x in range(ma_hists1.shape[1]) if x != 5902]
        ma_hists1[i, selector1] = np.histogram(
                    a=reduced_ma_values1, bins=null_distributions_["histogram_bins"], density=False
                )[0]
    # Inverse of step size in histBins (0.0001) = 10000
    step = 1 / np.mean(np.diff(null_distributions_["histogram_bins"])) #10000

    # Null distribution to convert ALE to p-values.
    ale_hist1 = ma_hists1[0, :]
    for i_exp in range(1, ma_hists1.shape[0]):
        temp_hist1 = np.copy(ale_hist1)
        ma_hist1 = np.copy(ma_hists1[i_exp, :])

        # Find histogram bins with nonzero values for each histogram.
        ale_idx1 = np.where(temp_hist1 > 0)[0]
        exp_idx1 = np.where(ma_hist1 > 0)[0]

        # Normalize histograms.
        temp_hist1 /= np.sum(temp_hist1)
        ma_hist1 /= np.sum(ma_hist1)

        # Perform weighted convolution of histograms.
        ale_hist1 = np.zeros(null_distributions_["histogram_bins"].shape[0]) #shape: (7542,)

        for j_idx in exp_idx1:
        # Compute probabilities of observing each ALE value in histBins
        # by randomly combining maps represented by maHist and aleHist.
        # Add observed probabilities to corresponding bins in ALE
        # histogram.
            probabilities = ma_hist1[j_idx] * temp_hist1[ale_idx1]
            ale_scores1 = 1 - (1 - null_distributions_["histogram_bins"][j_idx]) * (
                1 - null_distributions_["histogram_bins"][ale_idx1])
            score_idx1 = np.floor(ale_scores1 * step).astype(int)
            np.add.at(ale_hist1, score_idx1, probabilities)


    ## null ALE for the first group of experiments
    ma_hists2 = np.zeros((ma_values2.shape[0], null_distributions_["histogram_bins"].shape[0])) #shape: (100, 7542)
    for i in range(ma_values2.shape[0]):
        # Remember that histogram uses bin edges (not centers), so it
        # returns a 1xhist_bins-1 array
        n_zeros2 = len(np.where(ma_values2[i, :] == 0)[0])
        reduced_ma_values2 = ma_values2[i, ma_values2[i, :] > 0]
        
        ma_hists2[i, 5902] = n_zeros2
        selector2 = [x for x in range(ma_hists2.shape[1]) if x != 5902]
        ma_hists2[i, selector2] = np.histogram(
                    a=reduced_ma_values2, bins=null_distributions_["histogram_bins"], density=False
                )[0]
    # Inverse of step size in histBins (0.0001) = 10000
    step = 1 / np.mean(np.diff(null_distributions_["histogram_bins"])) #10000

    # Null distribution to convert ALE to p-values.
    ale_hist2 = ma_hists2[0, :]
    for i_exp in range(1, ma_hists2.shape[0]):
        temp_hist2 = np.copy(ale_hist2)
        ma_hist2 = np.copy(ma_hists2[i_exp, :])

        # Find histogram bins with nonzero values for each histogram.
        ale_idx2 = np.where(temp_hist2 > 0)[0]
        exp_idx2 = np.where(ma_hist2 > 0)[0]

        # Normalize histograms.
        temp_hist2 /= np.sum(temp_hist2)
        ma_hist2 /= np.sum(ma_hist2)

        # Perform weighted convolution of histograms.
        ale_hist2 = np.zeros(null_distributions_["histogram_bins"].shape[0]) #shape: (7542,)

        for j_idx in exp_idx2:
        # Compute probabilities of observing each ALE value in histBins
        # by randomly combining maps represented by maHist and aleHist.
        # Add observed probabilities to corresponding bins in ALE
        # histogram.
            probabilities = ma_hist2[j_idx] * temp_hist2[ale_idx2]
            ale_scores2 = 1 - (1 - null_distributions_["histogram_bins"][j_idx]) * (
                1 - null_distributions_["histogram_bins"][ale_idx2])
            score_idx2 = np.floor(ale_scores2 * step).astype(int)
            np.add.at(ale_hist2, score_idx2, probabilities)

    ale_hist_diff = ale_hist1 - ale_hist2
    null_distribution = ale_hist_diff / np.sum(ale_hist_diff)
    null_distribution = np.cumsum(null_distribution[::-1])[::-1]
    null_distribution /= np.max(null_distribution)
    null_distributions_["histogram_weights"] = null_distribution

    #print(null_distribution)
    return null_distributions_



def ale_to_p(null_distributions_, ale_values):
    """
    Compute p- and z-values.
    """
    step = 1 / np.mean(np.diff(null_distributions_["histogram_bins"]))

    # Determine p- and z-values from ALE values and null distribution.
    p_values = np.ones(ale_values.shape)

    idx = np.where(ale_values > 0)[0]
    ale_bins = round2(ale_values[idx] * step)
    p_values[idx] = null_distributions_["histogram_weights"][ale_bins]
    z_values = p_to_z(p_values, tail="one")
    return p_values, z_values


