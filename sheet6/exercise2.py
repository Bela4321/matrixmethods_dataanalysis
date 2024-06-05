import numpy as np



#create svd approximation
def svd_approx(A,k):
    U,S,Vt=np.linalg.svd(A)
    for i in range(k,S.shape[0]):
        S[i]=0
    S_1=np.diag(S)
    return U@S_1@Vt

for k in range(10):
    print(f"K={k}")
    for _ in range(100):
        # random matrix
        A=np.random.rand(15,15)
        A_1=svd_approx(A,3)
        A_cond=np.linalg.cond(A)
        A_1cond=np.linalg.cond(A_1)
        print(f"{A_cond}        {A_1cond}")