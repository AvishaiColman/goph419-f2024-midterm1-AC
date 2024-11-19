import numpy as np




def lu_factor(a, overwrite_a=False, full_output=False):
    """Factor a non-singular square matrix a
    into p, q, l, and u matrices such that p*a*q = l*u.
    Uses the Gaussian elimination algorithm with complete pivoting.

    Parameters
    ----------
    a : array_like, shape=(M, M)
        The coefficient matrix, must be full rank, det(a) != 0.
    overwrite_a : bool, default=False
        Flag for whether to overwrite a with the lu matrix.
    full_output : bool, default=False
        Flag for returning full l, u, p, and q arrays.

    Returns
    -------
    lu : numpy.ndarray, shape=(M, M) or tuple[numpy.array]
        The l and u matrices in compact storage,
        with l in the lower triangle (below main diagonal)
        and u in in the upper triangle (at and above main diagonal).
    pq : numpy.ndarray, shape=(2, M) or tuple[numpy.array]
        The p and q matrices in compact storage,
        with the first row containing row pivot vector p
        and the second row containing column pivot vector q.

    Notes
    -----
    Assume that matrix a has full rank.
    To separate the l and u matrices,
    create an identity matrix and copy the values below
    the main diagonal in lu to create l,
    and create a matrix of zeros and copy the values at
    and above the main diagonal in lu to create u.
    To create the p and q matrices,
    create an identity matrix and rearrange rows
    in the order given by pq[0, :] to create p
    and create an identity matrix and rearrange columns
    in the order given by pq[1, :] to create q.
    If full_output is set,
    the above steps will be done for you,
    and lu and pq will be tuples containing the full arrays.
    """
    # make a copy or rename the input array
    lu = a if overwrite_a else np.array(a, dtype="float64")
    # check for valid input
    shape = lu.shape
    M = shape[0]
    if len(shape) != 2:
        raise ValueError(f"a has dimension {len(shape)}, should be 2.")
    if M != shape[1]:
        raise ValueError(f"a has shape {shape}, should be square.")
    # initialize pivot array
    pq = np.vstack([np.arange(M), np.arange(M)])
    # forward elimination algorithm
    for k, _ in enumerate(lu):
        kp1 = k + 1
        # perform row and column pivoting
        row, col = np.argwhere(np.abs(lu[k:, k:]) == np.max(np.abs(lu[k:, k:])))[0, :]
        if row:
            swap = k + row
            lu[(k, swap), :] = lu[(swap, k), :]
            pq[0, k], pq[0, swap] = pq[0, swap], pq[0, k]
        if col:
            swap = k + col
            lu[:, (k, swap)] = lu[:, (swap, k)]
            pq[1, k], pq[1, swap] = pq[1, swap], pq[1, k]
        # eliminate below the pivot
        lu[kp1:, k] /= lu[k, k]
        lu[kp1:, kp1:] -= lu[kp1:, k:kp1] @ lu[k:kp1, kp1:]
    # tidy up output
    if full_output:
        i, j = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
        l_mat = np.eye(M, dtype="float64")
        l_mat[i > j] = lu[i > j]
        u_mat = np.zeros_like(lu)
        u_mat[i <= j] = lu[i <= j]
        lu = (l_mat, u_mat)
        p_mat = np.zeros((M, M), dtype="int")
        q_mat = np.zeros((M, M), dtype="int")
        p_mat[pq[0:1, :].T == j] = 1
        q_mat[pq[1:2, :] == i] = 1
        pq = (p_mat, q_mat)
    return lu, pq


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    '''
    Question 1
    '''
    L = 3.828e26    # units of W
    a = 0.306
    D = 1.496e11    # units of m
    sigma = 5.670374419e-8  # This is the Stefan Boltzmann constant with units of W m^-2 K^-4
    delta_L = 0.004e26
    delta_a = 0.001
    delta_D = 0.025e11

    # a)
    # on document 

    # b)
    # (i)
    rel_L_plus = (L - (L + delta_L)) / L
    rel_L_minus = (L - (L - delta_L)) / L
    print(f'The relative error in L is {rel_L_plus} for +{delta_L}, and {rel_L_minus} for -{delta_L}')

    # (ii)
    rel_a_plus = (a - (a + delta_a)) / a
    rel_a_minus = (a - (a - delta_a)) / a
    print(f'The relative error in L is {rel_a_plus} for +{delta_a}, and {rel_a_minus} for -{delta_a}')

    # (iii)
    rel_D_plus = (D - (D + delta_D)) / D
    rel_D_minus = (D - (D - delta_D)) / D
    print(f'The relative error in L is {rel_D_plus} for +{delta_D}, and {rel_D_minus} for -{delta_D}')

    # (iv)
    numerator = (L * (1 - a))
    denominator = 16 * np.pi * sigma * (D ** 2)
    T = (numerator / denominator) ** (0.25)
    print(f'The true value of temperature is {T} K')

    # (v)
    top_dTdL = (1 - a) / denominator
    bottom_dTdL = 4 * (((L * (1 - a)) / denominator) ** 0.75)
    dTdL = np.abs(top_dTdL / bottom_dTdL)
    
    top_dTda = (-L) / denominator
    bottom_dTda = 4 * (((L * (1 - a)) / denominator) ** 0.75)
    dTda = np.abs(top_dTda / bottom_dTda)
    
    D_denom = 16 * np.pi * sigma * (D ** 3)
    top_dTdD = ((-2 * L * (1 - a)) / D_denom)
    bottom_dTdD = 4 * (((L * (1 - a)) / denominator) ** 0.75)
    dTdD = np.abs(top_dTdD / bottom_dTdD)
    
    delta_T = dTdL * delta_L + dTda * delta_a + dTdD * delta_D
    print(f'The total error in T is plus or minus {delta_T}.')

    cont_total = dTdL + dTda + dTdD
    cont_L = (dTdL / cont_total) * 100
    cont_a = (dTda / cont_total) * 100
    cont_D = (dTdD / cont_total) * 100

    print(f'The error in L contributes to {cont_L} % of the total error')
    print(f'The error in a contributes to {cont_a} % of the total error')
    print(f'The error in D contributes to {cont_D} % of the total error')

    # (vi)
    rel_T_plus = (T - (T + delta_T)) / T
    rel_T_minus = (T - (T - delta_T)) / T
    print(f'The relative error in T is {rel_T_plus} for +{delta_T}, and {rel_T_minus} for -{delta_T}')

    # c)

    J = np.array([dTdL, dTda, dTdD])
    print(f'The Jacobian matrix is {J}')

    #---------------------------------------------------------------------------------------------------------------------
    '''
    Question 2
    '''
    # a)
    # on paper and in excel

    # b)
    # on paper and in excel

    # c)
    # in excel

    # d)
    K = np.array(
        [[4, -1, 0, 0], 
         [-1, 2, -1, 0], 
         [0, -1, 4, -3], 
         [0, 0, -3, 3]]
         )
    
    F = np.array([1, 2, 1, 5])

    u = np.array([3, 11, 17, 56/3])
    F_np = np.linalg.matmul(K, u)
    print(f'The F vector is confirmed to be {F_np} using our calculated u vector')
    u_np = np.linalg.solve(K, F)
    print(f'The u vector is confirmed to be solved as u = {u_np}')

    #-------------------------------------------------------------------------------------------------------------------------
    '''
    Question 3
    '''
    # a)
    # on document

    # b)

    print(lu_factor(K)) # this is the same matrix that was derived in the excel sheet

    L_matrix = np.array(
        [[1, 0, 0, 0], 
         [0, 1, 0, 0],
         [-1/4, -1/4, 1, 0],
         [0, -3/4, -1/2, 1]]
    )

    U_matrix = np.array(
        [[4, 0, -1, 0],
         [0, 4, -1, -3],
         [0, 0, 3/2, -3/4],
         [0, 0, 0, 3/8]]
    )

    P_matrix = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )

    Q_matrix = np.array(
         [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )

    left_1 = np.linalg.matmul(P_matrix, K)
    left_2 = np.linalg.matmul(left_1, Q_matrix)
    right = np.linalg.matmul(L_matrix, U_matrix)
    print(f'[P][K][Q] = {left_2} \n which is the same as [L][U] = {right}')


    # c) 
    # in excel

    






if __name__ == '__main__':
    main()
   