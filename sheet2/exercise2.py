import numpy as np
from numpy.linalg import norm, inv, cond, solve


matrixA=np.array([[5,2],[2,1]])
vecb=np.array([1,0])
delA=np.array([[0,0.01],[0,0]])
delb=np.array([0,0])

# a)
condition_number_A = cond(matrixA)
print(f"Condition number of A: {condition_number_A}")

# b)

perturbed_matrixA = matrixA + delA
inv_perturbed_matrixA = inv(perturbed_matrixA)
inv_perturbed_matrixA_2norm = norm(inv_perturbed_matrixA, 2)
print(f"2-norm of the inverse of the perturbed matrix A: {inv_perturbed_matrixA_2norm}")

inv_matrixA = inv(matrixA)
inv_matrixA_2norm = norm(inv_matrixA, 2)
r_matrixA = norm(delA, 2)*inv_matrixA_2norm
boundry = inv_matrixA_2norm/(1-r_matrixA)
print(f"Boundry for the norm of inverse of perturbation of A: {boundry}")

# c)

x = solve(matrixA, vecb)
pertubed_vecb = vecb + delb
y = solve(perturbed_matrixA, pertubed_vecb)
relative_error = norm(y-x, 2)/norm(x, 2)
print(f"Relative error in pertubed system: {relative_error}")

error_boundry_factor1 = condition_number_A/(1-r_matrixA)
error_boundry_factor2_summand1 = norm(delA, 2)/norm(matrixA, 2)
error_boundry_factor2_summand2 = norm(delb, 2)/norm(vecb, 2)
error_boundry = error_boundry_factor1*(error_boundry_factor2_summand1+error_boundry_factor2_summand2)
print(f"Error boundry for pertubed system: {error_boundry}")
if relative_error < error_boundry:
    print(f"Relative error is smaller than error boundry: {relative_error} < {error_boundry}")
else:
    print(f"Relative error is not smaller than error boundry: {relative_error} < {error_boundry}")
