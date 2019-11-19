#
# Seng 474
# A1 LSH Finding similar items
#

# Apprx Time: 10k- less than 10s, 100k- 2mins-50s

import numpy as np

def readin():
    #file_name = './data_10k_100.tsv'
    file_name = './data_100k_300.tsv'

    with open(file_name) as file:
        ND = [int(next(file).rstrip('\n')) for i in range(2)]

    data1 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3, usecols=[0]) #store the label to data1
    data2 = np.loadtxt(file_name, delimiter='\t', dtype=float, skiprows=3) #store the features to data2
    #print(data2)

    data2 = np.delete(data2, 0, 1)
    data2 = np.insert(data2, ND[1], 1, axis=1)

    sample = np.random.random_sample(ND[1]+1)

    n,m,T = 0,0,0
    output_filename = ''
    if file_name == './data_10k_100.tsv':
        n = 0.000001
        m = 1
        T = 20
        output_filename = '10k_Q3_output.tsv'
    else:
        n = 0.0000001
        m = 1
        T = 12
        output_filename = '100k_Q3_output.tsv'


    W = gradientDescent(sample, data1, data2, ND[0], n, m, T)

    Jw = computeLoss(ND[0], data1, data2, W)
    print(Jw)

    f = open(output_filename, 'w+')
    for i in range(1, ND[1]+1):
        f.write('w'+str(i) + '\t')
    f.write('w0\n')

    for i in W:
        f.write(str(i) + '\t')

#compute the stochastic gradientDescent
def gradientDescent(W, data1, data2, N, n, m, T):
    for i in range(T):
        for j in range(int(N)//m):
            yp = data1[j]
            Yp = np.matmul(W.transpose(), data2[j])
            subY = np.subtract(yp, Yp)
            for k in range(0, len(W)):
                Wj = W[k]
                nm = n/m
                result = nm * (subY * data2[j][k])
                W[k] = Wj + result
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
