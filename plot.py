import numpy as np
from matplotlib import pyplot as plt

def center_detect(data):
    with np.errstate(invalid='ignore'):
        n_experiment = data.shape[0]
        p_value_info = np.sum(data<=0.05, axis=1)
        mean_n_center = np.mean(p_value_info)
        at_least_one = (p_value_info>=1).sum()/n_experiment
        all_detected = (p_value_info==8).sum()/n_experiment
    
    return mean_n_center, at_least_one, all_detected

# one sample comparison
"""
# load the data from npy file
valid0 = np.load('one_sample_results/total studies 120/n_total120n_valid0_p-value.npy')
valid5 = np.load('one_sample_results/total studies 120/n_total120n_valid5_p-value.npy')
valid10 = np.load('one_sample_results/total studies 120/n_total120n_valid10_p-value.npy')
valid15 = np.load('one_sample_results/total studies 120/n_total120n_valid15_p-value.npy')
valid20 = np.load('one_sample_results/total studies 120/n_total120n_valid20_p-value.npy')
valid25 = np.load('one_sample_results/total studies 120/n_total120n_valid25_p-value.npy')
valid30 = np.load('one_sample_results/total studies 120/n_total120n_valid30_p-value.npy')
valid35 = np.load('one_sample_results/total studies 120/n_total120n_valid35_p-value.npy')
valid40 = np.load('one_sample_results/total studies 120/n_total120n_valid40_p-value.npy')
valid45 = np.load('one_sample_results/total studies 120/n_total120n_valid45_p-value.npy')
valid50 = np.load('one_sample_results/total studies 120/n_total120n_valid50_p-value.npy')
valid55 = np.load('one_sample_results/total studies 120/n_total120n_valid55_p-value.npy')
valid60 = np.load('one_sample_results/total studies 120/n_total120n_valid60_p-value.npy')
valid65 = np.load('one_sample_results/total studies 120/n_total120n_valid65_p-value.npy')
valid70 = np.load('one_sample_results/total studies 120/n_total120n_valid70_p-value.npy')
valid75 = np.load('one_sample_results/total studies 120/n_total120n_valid75_p-value.npy')
valid80 = np.load('one_sample_results/total studies 120/n_total120n_valid80_p-value.npy')
valid85 = np.load('one_sample_results/total studies 120/n_total120n_valid85_p-value.npy')
valid90 = np.load('one_sample_results/total studies 120/n_total120n_valid90_p-value.npy')
valid95 = np.load('one_sample_results/total studies 120/n_total120n_valid95_p-value.npy')
valid100 = np.load('one_sample_results/total studies 120/n_total120n_valid100_p-value.npy')
valid105 = np.load('one_sample_results/total studies 120/n_total120n_valid105_p-value.npy')
valid110 = np.load('one_sample_results/total studies 120/n_total120n_valid110_p-value.npy')
valid115 = np.load('one_sample_results/total studies 120/n_total120n_valid115_p-value.npy')
valid120 = np.load('one_sample_results/total studies 120/n_total120n_valid120_p-value.npy')

valid0_2 = np.load('one_sample_results/total studies 100/n_total100n_valid0_p-value.npy')
valid5_2 = np.load('one_sample_results/total studies 100/n_total100n_valid5_p-value.npy')
valid10_2 = np.load('one_sample_results/total studies 100/n_total100n_valid10_p-value.npy')
valid15_2 = np.load('one_sample_results/total studies 100/n_total100n_valid15_p-value.npy')
valid20_2 = np.load('one_sample_results/total studies 100/n_total100n_valid20_p-value.npy')
valid25_2 = np.load('one_sample_results/total studies 100/n_total100n_valid25_p-value.npy')
valid30_2 = np.load('one_sample_results/total studies 100/n_total100n_valid30_p-value.npy')
valid35_2 = np.load('one_sample_results/total studies 100/n_total100n_valid35_p-value.npy')
valid40_2 = np.load('one_sample_results/total studies 100/n_total100n_valid40_p-value.npy')
valid45_2 = np.load('one_sample_results/total studies 100/n_total100n_valid45_p-value.npy')
valid50_2 = np.load('one_sample_results/total studies 100/n_total100n_valid50_p-value.npy')
valid55_2 = np.load('one_sample_results/total studies 100/n_total100n_valid55_p-value.npy')
valid60_2 = np.load('one_sample_results/total studies 100/n_total100n_valid60_p-value.npy')
valid65_2 = np.load('one_sample_results/total studies 100/n_total100n_valid65_p-value.npy')
valid70_2 = np.load('one_sample_results/total studies 100/n_total100n_valid70_p-value.npy')
valid75_2 = np.load('one_sample_results/total studies 100/n_total100n_valid75_p-value.npy')
valid80_2 = np.load('one_sample_results/total studies 100/n_total100n_valid80_p-value.npy')
valid85_2 = np.load('one_sample_results/total studies 100/n_total100n_valid85_p-value.npy')
valid90_2 = np.load('one_sample_results/total studies 100/n_total100n_valid90_p-value.npy')
valid95_2 = np.load('one_sample_results/total studies 100/n_total100n_valid95_p-value.npy')
valid100_2 = np.load('one_sample_results/total studies 100/n_total100n_valid100_p-value.npy')

valid0_3 = np.load('one_sample_results/total studies 80/n_total80n_valid0_p-value.npy')
valid5_3 = np.load('one_sample_results/total studies 80/n_total80n_valid5_p-value.npy')
valid10_3 = np.load('one_sample_results/total studies 80/n_total80n_valid10_p-value.npy')
valid15_3 = np.load('one_sample_results/total studies 80/n_total80n_valid15_p-value.npy')
valid20_3 = np.load('one_sample_results/total studies 80/n_total80n_valid20_p-value.npy')
valid25_3 = np.load('one_sample_results/total studies 80/n_total80n_valid25_p-value.npy')
valid30_3 = np.load('one_sample_results/total studies 80/n_total80n_valid30_p-value.npy')
valid35_3 = np.load('one_sample_results/total studies 80/n_total80n_valid35_p-value.npy')
valid40_3 = np.load('one_sample_results/total studies 80/n_total80n_valid40_p-value.npy')
valid45_3 = np.load('one_sample_results/total studies 80/n_total80n_valid45_p-value.npy')
valid50_3 = np.load('one_sample_results/total studies 80/n_total80n_valid50_p-value.npy')
valid55_3 = np.load('one_sample_results/total studies 80/n_total80n_valid55_p-value.npy')
valid60_3 = np.load('one_sample_results/total studies 80/n_total80n_valid60_p-value.npy')
valid65_3 = np.load('one_sample_results/total studies 80/n_total80n_valid65_p-value.npy')
valid70_3 = np.load('one_sample_results/total studies 80/n_total80n_valid70_p-value.npy')
valid75_3 = np.load('one_sample_results/total studies 80/n_total80n_valid75_p-value.npy')
valid80_3 = np.load('one_sample_results/total studies 80/n_total80n_valid80_p-value.npy')

valid0_4 = np.load('one_sample_results/total studies 60/n_total60n_valid0_p-value.npy')
valid5_4 = np.load('one_sample_results/total studies 60/n_total60n_valid5_p-value.npy')
valid10_4 = np.load('one_sample_results/total studies 60/n_total60n_valid10_p-value.npy')
valid15_4 = np.load('one_sample_results/total studies 60/n_total60n_valid15_p-value.npy')
valid20_4 = np.load('one_sample_results/total studies 60/n_total60n_valid20_p-value.npy')
valid25_4 = np.load('one_sample_results/total studies 60/n_total60n_valid25_p-value.npy')
valid30_4 = np.load('one_sample_results/total studies 60/n_total60n_valid30_p-value.npy')
valid35_4 = np.load('one_sample_results/total studies 60/n_total60n_valid35_p-value.npy')
valid40_4 = np.load('one_sample_results/total studies 60/n_total60n_valid40_p-value.npy')
valid45_4 = np.load('one_sample_results/total studies 60/n_total60n_valid45_p-value.npy')
valid50_4 = np.load('one_sample_results/total studies 60/n_total60n_valid50_p-value.npy')
valid55_4 = np.load('one_sample_results/total studies 60/n_total60n_valid55_p-value.npy')
valid60_4 = np.load('one_sample_results/total studies 60/n_total60n_valid60_p-value.npy')


valid_list = [valid0, valid5, valid10, valid15, valid20, valid25, valid30, valid35, valid40, valid45,
            valid50, valid55, valid60, valid65, valid70, valid75, valid80, valid85, valid90, valid95,
            valid100, valid105, valid110, valid115, valid120]
mean_n_center, at_least_one, all_detected = list(), list(), list()

valid_list_2 = [valid0_2, valid5_2, valid10_2, valid15_2, valid20_2, valid25_2, valid30_2, valid35_2, valid40_2, valid45_2,
            valid50_2, valid55_2, valid60_2, valid65_2, valid70_2, valid75_2, valid80_2, valid85_2, valid90_2, valid95_2, valid100_2]
mean_n_center2, at_least_one2, all_detected2 = list(), list(), list()

valid_list_3 = [valid0_3, valid5_3, valid10_3, valid15_3, valid20_3, valid25_3, valid30_3, valid35_3, valid40_3, valid45_3,
            valid50_3, valid55_3, valid60_3, valid65_3, valid70_3, valid75_3, valid80_3]
mean_n_center3, at_least_one3, all_detected3 = list(), list(), list()

valid_list_4 = [valid0_4, valid5_4, valid10_4, valid15_4, valid20_4, valid25_4, valid30_4, valid35_4, valid40_4, valid45_4,
            valid50_4, valid55_4, valid60_4]
mean_n_center4, at_least_one4, all_detected4 = list(), list(), list()

for element in valid_list:
    data_info = center_detect(element)
    mean_n_center.append(data_info[0])
    at_least_one.append(data_info[1])
    all_detected.append(data_info[2])

for element in valid_list_2:
    data_info = center_detect(element)
    mean_n_center2.append(data_info[0])
    at_least_one2.append(data_info[1])
    all_detected2.append(data_info[2])

for element in valid_list_3:
    data_info = center_detect(element)
    mean_n_center3.append(data_info[0])
    at_least_one3.append(data_info[1])
    all_detected3.append(data_info[2])

for element in valid_list_4:
    data_info = center_detect(element)
    mean_n_center4.append(data_info[0])
    at_least_one4.append(data_info[1])
    all_detected4.append(data_info[2])


t = np.arange(0, 125, 5)
t2 = np.arange(0, 105, 5)
t3 = np.arange(0, 85, 5)
t4 = np.arange(0, 65, 5)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)    # The big subplot
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(t, at_least_one, 'brown', label="total studies: 120")
ax1.plot(t2, at_least_one2, 'darkblue',label="total studies: 100")
ax1.plot(t3, at_least_one3, 'gold',label="total studies: 80")
ax1.plot(t4, at_least_one4, 'pink',label="total studies: 60")
ax1.set_xlabel('the total number of valid studies')
ax1.set_ylabel('Probability at least one center detected')
ax1.legend(loc="lower right")

ax2.plot(t, all_detected, 'brown', label="total studies: 120")
ax2.plot(t2, all_detected2, 'darkblue', label="total studies: 100")
ax2.plot(t3, all_detected3, 'gold',label="total studies: 80")
ax2.plot(t4, all_detected4, 'pink',label="total studies: 60")
ax2.set_xlabel('the total number of valid studies')
ax2.set_ylabel('Probability all 8 centers detected')
ax2.legend(loc="lower right")

ax3.plot(t, mean_n_center, 'brown', label="total studies: 120")
ax3.plot(t2, mean_n_center2, 'darkblue', label="total studies: 100")
ax3.plot(t3, mean_n_center3, 'gold', label="total studies: 80")
ax3.plot(t4, mean_n_center4, 'pink', label="total studies: 60")
ax3.set_xlabel('the total number of valid studies')
ax3.set_ylabel('Mean number of centers detected')
ax3.legend(loc="lower right")

plt.savefig('one_sample_results/one_sample_test.png')

"""


"""
# load the data from npy file
total20_120 = np.load('two_sample_results/total studies 20-120/n_total20&120n_valid20&20_p-value.npy')
total25_120 = np.load('two_sample_results/total studies 20-120/n_total25&120n_valid20&20_p-value.npy')
total30_120 = np.load('two_sample_results/total studies 20-120/n_total30&120n_valid20&20_p-value.npy')
total35_120 = np.load('two_sample_results/total studies 20-120/n_total35&120n_valid20&20_p-value.npy')
total40_120 = np.load('two_sample_results/total studies 20-120/n_total40&120n_valid20&20_p-value.npy')
total45_120 = np.load('two_sample_results/total studies 20-120/n_total45&120n_valid20&20_p-value.npy')
total50_120 = np.load('two_sample_results/total studies 20-120/n_total50&120n_valid20&20_p-value.npy')
total55_120 = np.load('two_sample_results/total studies 20-120/n_total55&120n_valid20&20_p-value.npy')
total60_120 = np.load('two_sample_results/total studies 20-120/n_total60&120n_valid20&20_p-value.npy')
total65_120 = np.load('two_sample_results/total studies 20-120/n_total65&120n_valid20&20_p-value.npy')
total70_120 = np.load('two_sample_results/total studies 20-120/n_total70&120n_valid20&20_p-value.npy')
total75_120 = np.load('two_sample_results/total studies 20-120/n_total75&120n_valid20&20_p-value.npy')
total80_120 = np.load('two_sample_results/total studies 20-120/n_total80&120n_valid20&20_p-value.npy')
total85_120 = np.load('two_sample_results/total studies 20-120/n_total85&120n_valid20&20_p-value.npy')
total90_120 = np.load('two_sample_results/total studies 20-120/n_total90&120n_valid20&20_p-value.npy')
total95_120 = np.load('two_sample_results/total studies 20-120/n_total95&120n_valid20&20_p-value.npy')
total100_120 = np.load('two_sample_results/total studies 20-120/n_total100&120n_valid20&20_p-value.npy')
total105_120 = np.load('two_sample_results/total studies 20-120/n_total105&120n_valid20&20_p-value.npy')
total110_120 = np.load('two_sample_results/total studies 20-120/n_total110&120n_valid20&20_p-value.npy')
total115_120 = np.load('two_sample_results/total studies 20-120/n_total115&120n_valid20&20_p-value.npy')
total120_120 = np.load('two_sample_results/total studies 20-120/n_total120&120n_valid20&20_p-value.npy')

total_list = [total20_120, total25_120, total30_120, total35_120, total40_120, total45_120, total50_120,
            total55_120, total60_120, total65_120, total70_120, total75_120, total80_120, total85_120,
            total90_120, total95_120, total100_120, total105_120, total110_120, total115_120, total120_120]


mean_n_center, at_least_one, all_detected = list(), list(), list()

for element in total_list:
    data_info = center_detect(element)
    mean_n_center.append(data_info[0])
    at_least_one.append(data_info[1])
    all_detected.append(data_info[2])

total20_100 = np.load('two_sample_results/total studies 20-100/n_total20&100n_valid20&20_p-value.npy')
total25_100 = np.load('two_sample_results/total studies 20-100/n_total25&100n_valid20&20_p-value.npy')
total30_100 = np.load('two_sample_results/total studies 20-100/n_total30&100n_valid20&20_p-value.npy')
total35_100 = np.load('two_sample_results/total studies 20-100/n_total35&100n_valid20&20_p-value.npy')
total40_100 = np.load('two_sample_results/total studies 20-100/n_total40&100n_valid20&20_p-value.npy')
total45_100 = np.load('two_sample_results/total studies 20-100/n_total45&100n_valid20&20_p-value.npy')
total50_100 = np.load('two_sample_results/total studies 20-100/n_total50&100n_valid20&20_p-value.npy')
total55_100 = np.load('two_sample_results/total studies 20-100/n_total55&100n_valid20&20_p-value.npy')
total60_100 = np.load('two_sample_results/total studies 20-100/n_total60&100n_valid20&20_p-value.npy')
total65_100 = np.load('two_sample_results/total studies 20-100/n_total65&100n_valid20&20_p-value.npy')
total70_100 = np.load('two_sample_results/total studies 20-100/n_total70&100n_valid20&20_p-value.npy')
total75_100 = np.load('two_sample_results/total studies 20-100/n_total75&100n_valid20&20_p-value.npy')
total80_100 = np.load('two_sample_results/total studies 20-100/n_total80&100n_valid20&20_p-value.npy')
total85_100 = np.load('two_sample_results/total studies 20-100/n_total85&100n_valid20&20_p-value.npy')
total90_100 = np.load('two_sample_results/total studies 20-100/n_total90&100n_valid20&20_p-value.npy')
total95_100 = np.load('two_sample_results/total studies 20-100/n_total95&100n_valid20&20_p-value.npy')
total100_100 = np.load('two_sample_results/total studies 20-100/n_total100&100n_valid20&20_p-value.npy')

total_list_100 = [total20_100, total25_100, total30_100, total35_100, total40_100, total45_100, total50_100,
            total55_100, total60_100, total65_100, total70_100, total75_100, total80_100, total85_100,
            total90_100, total95_100, total100_100]


mean_n_center100, at_least_one100, all_detected100 = list(), list(), list()

for element in total_list_100:
    data_info = center_detect(element)
    mean_n_center100.append(data_info[0])
    at_least_one100.append(data_info[1])
    all_detected100.append(data_info[2])

total20_80 = np.load('two_sample_results/total studies 20-80/n_total20&80n_valid20&20_p-value.npy')
total25_80 = np.load('two_sample_results/total studies 20-80/n_total25&80n_valid20&20_p-value.npy')
total30_80 = np.load('two_sample_results/total studies 20-80/n_total30&80n_valid20&20_p-value.npy')
total35_80 = np.load('two_sample_results/total studies 20-80/n_total35&80n_valid20&20_p-value.npy')
total40_80 = np.load('two_sample_results/total studies 20-80/n_total40&80n_valid20&20_p-value.npy')
total45_80 = np.load('two_sample_results/total studies 20-80/n_total45&80n_valid20&20_p-value.npy')
total50_80 = np.load('two_sample_results/total studies 20-80/n_total50&80n_valid20&20_p-value.npy')
total55_80 = np.load('two_sample_results/total studies 20-80/n_total55&80n_valid20&20_p-value.npy')
total60_80 = np.load('two_sample_results/total studies 20-80/n_total60&80n_valid20&20_p-value.npy')
total65_80 = np.load('two_sample_results/total studies 20-80/n_total65&80n_valid20&20_p-value.npy')
total70_80 = np.load('two_sample_results/total studies 20-80/n_total70&80n_valid20&20_p-value.npy')
total75_80 = np.load('two_sample_results/total studies 20-80/n_total75&80n_valid20&20_p-value.npy')
total80_80 = np.load('two_sample_results/total studies 20-80/n_total80&80n_valid20&20_p-value.npy')


total_list_80 = [total20_80, total25_80, total30_80, total35_80, total40_80, total45_80, total50_80,
            total55_80, total60_80, total65_80, total70_80, total75_80, total80_80]
            


mean_n_center80, at_least_one80, all_detected80 = list(), list(), list()

for element in total_list_80:
    data_info = center_detect(element)
    mean_n_center80.append(data_info[0])
    at_least_one80.append(data_info[1])
    all_detected80.append(data_info[2])
# create plots  


t120 = np.arange(20, 125, 5)
t100 = np.arange(20, 105, 5)
t80 = np.arange(20, 85, 5)

plt.plot(t120, mean_n_center, 'blue', label='total studies: 120')
plt.plot(t100, mean_n_center100, 'black', label='total studies: 100')
plt.plot(t80, mean_n_center80, 'red', label='total studies: 80')
plt.title('(n_total2=80/100/120, n_valid1 = n_valid2 = 20)')
plt.xlabel('n_total1 (total number of studies in the first group)')
plt.ylabel('Mean number of centers with p-value<0.05')
plt.legend(loc="upper right")


plt.savefig('two_sample_results/two_sample_test.png')
"""



# load the data from npy file
valid20_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid20&20_p-value.npy')
valid25_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid25&20_p-value.npy')
valid30_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid30&20_p-value.npy')
valid35_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid35&20_p-value.npy')
valid40_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid40&20_p-value.npy')
valid45_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid45&20_p-value.npy')
valid50_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid50&20_p-value.npy')
valid55_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid55&20_p-value.npy')
valid60_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid60&20_p-value.npy')
valid65_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid65&20_p-value.npy')
valid70_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid70&20_p-value.npy')
valid75_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid75&20_p-value.npy')
valid80_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid80&20_p-value.npy')
valid85_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid85&20_p-value.npy')
valid90_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid90&20_p-value.npy')
valid95_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid95&20_p-value.npy')
valid100_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid100&20_p-value.npy')
valid105_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid105&20_p-value.npy')
valid110_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid110&20_p-value.npy')
valid115_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid115&20_p-value.npy')
valid120_20 = np.load('two_sample_results/valid studies 20-120/n_total120&120n_valid120&20_p-value.npy')



valid_list = [valid20_20, valid25_20, valid30_20, valid35_20, valid40_20, valid45_20, valid50_20,
            valid55_20, valid60_20, valid65_20, valid70_20, valid75_20, valid80_20, valid85_20,
            valid90_20, valid95_20, valid100_20, valid105_20, valid110_20, valid115_20, valid120_20]
                

mean_n_center, at_least_one, all_detected = list(), list(), list()

for element in valid_list:
    data_info = center_detect(element)
    mean_n_center.append(data_info[0])
    at_least_one.append(data_info[1])
    all_detected.append(data_info[2])

valid20_100= np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid20&20_p-value.npy')
valid25_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid25&20_p-value.npy')
valid30_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid30&20_p-value.npy')
valid35_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid35&20_p-value.npy')
valid40_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid40&20_p-value.npy')
valid45_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid45&20_p-value.npy')
valid50_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid50&20_p-value.npy')
valid55_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid55&20_p-value.npy')
valid60_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid60&20_p-value.npy')
valid65_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid65&20_p-value.npy')
valid70_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid70&20_p-value.npy')
valid75_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid75&20_p-value.npy')
valid80_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid80&20_p-value.npy')
valid85_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid85&20_p-value.npy')
valid90_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid90&20_p-value.npy')
valid95_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid95&20_p-value.npy')
valid100_100 = np.load('two_sample_results/valid studies 20-100/n_total100&100n_valid100&20_p-value.npy')

valid_list_100 = [valid20_100, valid25_100, valid30_100, valid35_100, valid40_100, valid45_100, valid50_100,
            valid55_100, valid60_100, valid65_100, valid70_100, valid75_100, valid80_100, valid85_100,
            valid90_100, valid95_100, valid100_100]
                

mean_n_center100, at_least_one100, all_detected100 = list(), list(), list()

for element in valid_list_100:
    data_info = center_detect(element)
    mean_n_center100.append(data_info[0])
    at_least_one100.append(data_info[1])
    all_detected100.append(data_info[2])

valid20_80= np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid20&20_p-value.npy')
valid25_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid25&20_p-value.npy')
valid30_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid30&20_p-value.npy')
valid35_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid35&20_p-value.npy')
valid40_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid40&20_p-value.npy')
valid45_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid45&20_p-value.npy')
valid50_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid50&20_p-value.npy')
valid55_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid55&20_p-value.npy')
valid60_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid60&20_p-value.npy')
valid65_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid65&20_p-value.npy')
valid70_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid70&20_p-value.npy')
valid75_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid75&20_p-value.npy')
valid80_80 = np.load('two_sample_results/valid studies 20-80/n_total80&80n_valid80&20_p-value.npy')


valid_list_80 = [valid20_80, valid25_80, valid30_80, valid35_80, valid40_80, valid45_80, valid50_80,
            valid55_80, valid60_80, valid65_80, valid70_80, valid75_80, valid80_80]
                

mean_n_center80, at_least_one80, all_detected80 = list(), list(), list()

for element in valid_list_80:
    data_info = center_detect(element)
    mean_n_center80.append(data_info[0])
    at_least_one80.append(data_info[1])
    all_detected80.append(data_info[2])

# create plots
t120 = np.arange(20, 125, 5)
t100 = np.arange(20, 105, 5)
t80 = np.arange(20, 85, 5)

plt.plot(t120, mean_n_center, 'blue', label='total studies: 120')
plt.plot(t100, mean_n_center100, 'black', label='total studies: 100')
plt.plot(t80, mean_n_center80, 'red', label='total studies: 80')

plt.title('(n_valid2=20, n_total1 = n_total2 = 80/100/120)')
plt.xlabel('n_valid1 (number of valid studies in the first group)')
plt.ylabel('Mean number of centers with p-value<0.05')
plt.legend(loc="lower right")



plt.savefig('two_sample_results/two_sample_test2.png')


