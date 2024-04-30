from numpy import array
from numpy.linalg import eigvals


matrixa = array([[1,2,3],[4,5,6],[7,8,9]])
matrixb = array([[0,1],[-1,0]])

print(eigvals(matrixa))