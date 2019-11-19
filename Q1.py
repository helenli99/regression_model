#
# Seng 474
# A1 LSH Finding similar items
#

# Apprx Time: 100k- 15s

import numpy as np

def readin():
    file_name = './data_100k_300.tsv'

    with open(file_name) as file:
        ND = [int(next(file).rstrip('\n')) for i in range(2)]

    data1 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3, usecols=[0]) #stroe the label to data1
    data2 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3) #store the features to data2
    #print(data2)

    data2 = np.delete(data2, 0, 1)
    data2 = np.insert(data2, ND[1], 1, axis=1) #store 1 to every end of features
    #print(data2)

    W = computeW(ND, data1, data2) #compute the W
    #print(W)
    Jw = computeLoss(ND[0], data1, data2, W) # calculate the loss value
    print(Jw)

    f = open('100k_Q1_output.tsv', 'w+') #output a file
    for i in range(1, ND[1]+1):
        f.write('w'+str(i) + '\t')
    f.write('w0\n')

    for i in W:
        f.write(str(i) + '\t')

#This function use to compute the W
def computeW(ND, data1, data2):
    firstPart = np.matmul(data2.transpose(), data2)
    secondPart = np.matmul(data2.transpose(), data1)
    invserFirst = np.linalg.inv(firstPart)
    W = np.matmul(invserFirst, secondPart)
    return W

#This function use to calculate the loss value
def computeLoss(N, data1, data2, W):
    left = np.subtract(np.matmul(data2, W), data1)
    result = (1/(2*N)) * np.matmul(left.transpose(), left)
    return result

if __name__ == '__main__':
    readin()
