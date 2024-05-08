from math import sqrt
import numpy as np

a1=np.array([1,3,5])
a2=np.array([2,4,6])

def getRotationmatrix(vector, placeToZero, placeToZeroWith):
    scaleFactor= 1/sqrt(vector[placeToZero]**2+vector[placeToZeroWith]**2)
    rows = []
    for i in range(len(vector)):
        currentRow=[]
        if (i==placeToZeroWith):
            for j in range(len(vector)):
                if (j==placeToZeroWith):
                    currentRow.append(scaleFactor*vector[placeToZeroWith])
                elif (j==placeToZero):
                    currentRow.append(scaleFactor*vector[placeToZero])
                else:
                    currentRow.append(0)
        elif (i==placeToZero):
            for j in range(len(vector)):
                if (j==placeToZeroWith):
                    currentRow.append(-1*scaleFactor*vector[placeToZero])
                elif (j==placeToZero):
                    currentRow.append(scaleFactor*vector[placeToZeroWith])
                else:
                    currentRow.append(0)
        else:
            for j in range(len(vector)):
                if (i==j):
                    currentRow.append(1)
                else:
                    currentRow.append(0)
        rows.append(currentRow)
    return np.array(rows)

G1=getRotationmatrix(a1,2,1)
a1_two=G1@a1
G2=getRotationmatrix(a1_two,1,0)
a1_three=G2@a1_two

a2_two=G1@a2
a2_three=G2@a2_two
print(a2_three)
G3=getRotationmatrix(a2_three,2,1)
a2_four=G3@a2_three
print(a2_four)

A=np.array([[1,2],[3,4],[5,6]])
B=G3@G2@G1@A
print(B)