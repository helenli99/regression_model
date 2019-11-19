#
# Seng 474
# A1 LSH Finding similar items
#

# Apprx Time: 10k- less than 10s

import numpy as np

def readin():
    #file_name = './pa2-sample.tsv'
    file_name = './data_10k_100.tsv'

    with open(file_name) as file:
        ND = [int(next(file).rstrip('\n')) for i in range(2)]

    data1 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3, usecols=[0]) #store the label to data1
    data2 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3) #store the features to data2

    data2 = np.delete(data2, 0, 1)
    data2 = np.insert(data2, ND[1], 1, axis=1)

    sample = np.random.random_sample(ND[1]+1) #random the sample W

    W = gradientDescent(sample, data2, data1, ND[0])
    #print(W)
    Jw = computeLoss(ND[0], data1, data2, W)
    print(Jw)

    f = open('10k_Q2_output.tsv', 'w+')
    for i in range(1, ND[1]+1):
        f.write('w'+str(i) + '\t')
    f.write('w0\n')

    for i in W:
        f.write(str(i) + '\t')

#compute the batch gradientDescent
def gradientDescent(W, X, Y, N):
    for i in range(200):
        first = np.matmul(np.matmul(X.transpose(), X), W)
        sencond = np.matmul(X.transpose(), Y)
        third = (0.000001/N) * np.subtract(first, sencond)
        W = np.subtract(W, third)
    #print(W)
    return W

def computeW(ND, data1, data2):
    firstPart = np.matmul(data2.transpose(), data2)
    secondPart = np.matmul(data2.transpose(), data1)
    invserFirst = np.linalg.inv(firstPart)
    W = np.matmul(invserFirst, secondPart)
    return W

def computeLoss(N, data1, data2, W):
    left = np.subtract(np.matmul(data2, W), data1)
    result = (1/(2*N)) * np.matmul(left.transpose(), left)
    return result

if __name__ == '__main__':
    readin()
