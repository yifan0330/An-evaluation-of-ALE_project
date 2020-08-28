import threading 
import os
import numpy as np
import nimare
from simulation import Foci_generator
from permutation import foci_dataset, two_group_permutation, ale_diff, ALE_subtraction_result


population_center = np.array([[63,67,50], [63,35,50], [31,73,31], [58,74,31],
                            [28,71,50], [28,38,50], [57,44,66], [34,44,66]])
probs = np.array([0.35,0.50,0.10,0.05]) # probabilities of generating 0/1/2/3 foci
mask_file = 'mask.nii'

# two sample comparison     
n_total1,  n_total2 = 25, 100
n_valid1, n_valid2 = 10, 10

sigma = 10 # results in (Eickhoff et al, 2009)
grid_width = 33

simulation = Foci_generator(population_center, probs, mask_file)
# draw the coordinates of foci
foci_coords1 = simulation.sample_drawer(n_total1, n_valid1, sigma, grid_width)
foci_coords2 = simulation.sample_drawer(n_total2, n_valid2, sigma, grid_width)
# convert numpy array into the form of `nimare.dataset`
dataset1 = foci_dataset(foci_coords1)
dataset2= foci_dataset(foci_coords2)
original_ale_diff = ale_diff(dataset1, dataset2)

# save the results in npy file
outfile = 'ALE_subtraction_original.npy'
np.save(outfile, original_ale_diff)

print('----------------------------------')

def task1():
    result = ALE_subtraction_result(foci_coords1, foci_coords2, N=20)
    # save the results in npy file
    outfile = 'ALE_subtraction_after_permutation1.npy'
    np.save(outfile, result)

def task2():
    result = ALE_subtraction_result(foci_coords1, foci_coords2, N=20)
    # save the results in npy file
    outfile = 'ALE_subtraction_after_permutation2.npy'
    np.save(outfile, result)

def task3():
    result = ALE_subtraction_result(foci_coords1, foci_coords2, N=20)
    # save the results in npy file
    outfile = 'ALE_subtraction_after_permutation3.npy'
    np.save(outfile, result)

def task4():
    result = ALE_subtraction_result(foci_coords1, foci_coords2, N=20)
    # save the results in npy file
    outfile = 'ALE_subtraction_after_permutation4.npy'
    np.save(outfile, result)

def task5():
    result = ALE_subtraction_result(foci_coords1, foci_coords2, N=20)
    # save the results in npy file
    outfile = 'ALE_subtraction_after_permutation5.npy'
    np.save(outfile, result)

# creating threads 
t1 = threading.Thread(target=task1, name='t1') 
t2 = threading.Thread(target=task2, name='t2')   
t3 = threading.Thread(target=task3, name='t3') 
t4 = threading.Thread(target=task4, name='t4')
t5 = threading.Thread(target=task5, name='t5')

# starting threads 
t1.start() 
t2.start() 
t3.start() 
t4.start() 
t5.start() 

# wait until all threads finish 
t1.join() 
t2.join() 
t3.join() 
t4.join() 
t5.join() 
