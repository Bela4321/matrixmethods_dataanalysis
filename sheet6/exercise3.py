import numpy as np
from matplotlib import pyplot as plt


def display_digit(vector, fname):
    #split to array
    matrix_lists=[vector[i*16:(i+1)*16] for i in range(0,16)]
    matrix=np.array(matrix_lists)
    plt.imsave(fname,matrix)
    
training={}
for i in range(0,10):
    # load from csv
    with open(f"sheet6/USPS Digits/usps-testing-{i}.csv","r") as file:
        vectors=[]
        for line in file:
            vec_str= line.split(",")
            vec=[float(item) for item in vec_str]
            vectors.append(vec)
        training[i]=vectors
        
centroids={}
for i in range(0,10):
    centroid=np.mean(training[i],axis=0)
    centroids[i]=centroid
    display_digit(centroid,f"centroid_{i}.png")
    
    
def centroid_classification(vector):
    min_dist=float("inf")
    min_i=-1
    for i in range(0,10):
        dist=np.linalg.norm(vector-centroids[i],2)
        if dist<min_dist:
            min_dist=dist
            min_i=i
    return min_i

right=0
wrong=0
for i in range(0,10):
    for vector in training[i]:
        if centroid_classification(vector)==i:
            right+=1
        else:
            wrong+=1
print("accuracy with euclidean distance:")
print(right/(right+wrong))

def centroid_cosine_classification(vector):
    max_cosine=-1
    max_i=-1
    for i in range(0,10):
        cosine=np.dot(vector,centroids[i])/(np.linalg.norm(vector,2)*np.linalg.norm(centroids[i],2))
        if cosine>max_cosine:
            max_cosine=cosine
            max_i=i
    return max_i

right_cosine=0
wrong_cosine=0
for i in range(0,10):
    for vector in training[i]:
        if centroid_cosine_classification(vector)==i:
            right_cosine+=1
        else:
            wrong_cosine+=1
print("accuracy with cosine similarity:")
print(right_cosine/(right_cosine+wrong_cosine))


# svd basis classification

def get_residual(vector,basis):
    residual=vector
    for b in np.transpose(basis):
        residual=residual-np.dot(b,vector)*b
    return np.linalg.norm(residual,2)

def svd_classification(vector,bases):
    min_residual=float("inf")
    min_i=-1
    for i in range(0,10):
        residual=get_residual(vector,bases[i])
        if residual<min_residual:
            min_residual=residual
            min_i=i
    return min_i

for k in [1,2,4,6,8,10]:
    print(f"accuracy with k={k}")
    bases={}
    for i in range(0,10):
        # svd
        U,s,Vt=np.linalg.svd(training[i])
        basis=np.transpose(Vt)[:,:k]
        bases[i]=basis
        
    right=0
    wrong=0
    for i in range(0,10):
        for vector in training[i]:
            if svd_classification(vector,bases)==i:
                right+=1
            else:
                wrong+=1
    print("accuracy with svd classification:")
    print(right/(right+wrong))
    