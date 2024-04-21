import numpy as np
import numpy.linalg as la

matrixA = np.array([[-3,0,1],[2,-5,2],[1,1,-2]])
matrixB = np.array([[-3,0.1,1],[2.1,-5,2],[1.1,1,-2]])

norms = [1, 2, np.inf, "fro"]

for norm in norms:
    normA = la.norm(matrixA, ord=norm)
    normB = la.norm(matrixB, ord=norm)
    diff = round(abs(normA - normB), 6)
    print(f"{norm}-Norm :")
    print(f"Difference between ||A|| and ||B||: {diff}")

matrixDiff= matrixA - matrixB
for norm in norms:
    normDiff = round(la.norm(matrixDiff, ord=norm),6)
    print(f"{norm}-Norm of A - B: {normDiff}")