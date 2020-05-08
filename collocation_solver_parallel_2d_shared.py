import sys
import time
import argparse
from math import *

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float64

import bvp_problem
import collocation_coefficients
import collocation_node
import gauss_coefficients
import matrix_factorization_cuda
import matrix_operation_cuda
from BVPDAEReadWriteData import bvpdae_write_data
from bvp_problem import _abvp_f, _abvp_g, _abvp_r, _abvp_Df, _abvp_Dg, _abvp_Dr
import pathlib
import solve_babd_system

TPB_N = 32  # threads per block in time dimension
N_shared = TPB_N + 1
TPB_m = 16  # threads per block in collocation dimension
TPB = 32  # threads per block for 1d kernel
m_collocation = 0
global_size_y = 0
global_size_z = 0
global_size_p = 0
global_y_shared_size = 0
residual_type = 1
scale_by_time = True
scale_by_initial = False
residual2 = False


def collocation_solver_parallel(m=4):
    print("m = {}".format(m))
    # construct the bvp-dae problem
    # obtain the initial input
    bvp_dae = bvp_problem.BvpDae()
    global global_size_y, global_size_z, global_size_p, m_collocation, TPB_m, global_y_shared_size
    size_y = bvp_dae.size_y
    global_size_y = size_y
    size_z = bvp_dae.size_z
    global_size_z = size_z
    size_p = bvp_dae.size_p
    global_size_p = size_p
    size_inequality = bvp_dae.size_inequality
    size_sv_inequality = bvp_dae.size_sv_inequality
    output_file = bvp_dae.output_file
    example_name = output_file.split('.')[0]

    t_span0 = bvp_dae.T0
    N = t_span0.shape[0]
    y0 = bvp_dae.Y0
    z0 = bvp_dae.Z0
    p0 = bvp_dae.P0
    # parameters for numerical solvers
    tol = bvp_dae.tolerance
    max_iter = bvp_dae.maximum_newton_iterations
    max_mesh = bvp_dae.maximum_mesh_refinements
    max_nodes = bvp_dae.maximum_nodes
    min_nodes = 3
    max_linesearch = 20
    alpha = 0.1  # continuation parameter
    if size_inequality > 0 or size_sv_inequality > 0:
        alpha_m = 1e-6
    else:
        alpha_m = 0.1
    beta = 0.9  # scale factor
    # specify collocation coefficients
    m_collocation = m
    # minimum number of power of 2 as the TPB in m direction
    pos = ceil(log(m, 2))
    TPB_m = max(int(pow(2, pos)), 2)  # at least two threads in y direction
    global_y_shared_size = TPB_N * m
    rk = collocation_coefficients.lobatto(m)
    M = 8  # number of blocks used to solve the BABD system in parallel
    success_flag = 1
    # benchmark data
    initial_input_time = 0
    initial_input_count = 0
    residual_time = 0
    residual_count = 0
    jacobian_time = 0
    jacobian_count = 0
    reduce_jacobian_time = 0
    reduce_jacobian_count = 0
    recover_babd_time = 0
    recover_babd_count = 0
    segment_residual_time = 0
    segment_residual_count = 0

    para = np.copy(p0)
    t_span = np.copy(t_span0)
    max_residual = 1 / tol

    solver_start_time = time.time()
    start_time_initial_input = time.time()
    # form the initial input of the solver
    y, y_dot, z_tilde = form_initial_input_parallel(size_y, size_z, size_p, m, N, y0, z0, para, rk)
    initial_input_time += (time.time() - start_time_initial_input)
    initial_input_count += 1

    for alphacal in range(max_iter):
        print("Continuation iteration: {}, solving alpha = {}".format(alphacal, alpha))
        mesh_it = 0
        iter_time = 0
        for iter_time in range(max_iter):
            start_time_residual = time.time()
            # compute the residual
            f_q, y_tilde, f_a, f_b, r_bc = compute_f_q_parallel(
                size_y, size_z, size_p, m, N, rk, t_span, y, y_dot, z_tilde, para, alpha)
            residual_time += (time.time() - start_time_residual)
            residual_count += 1
            norm_f_q = np.linalg.norm(f_q, np.inf)
            print("\tnorm: {0:.8f}".format(norm_f_q))
            if norm_f_q < tol:
                print('\talpha = {}, solution is found. Number of nodes: {}'.format(alpha, N))
                break
            start_time_jacobian = time.time()
            # compute each necessary element in the Jacobian matrix
            J, V, D, W, B_0, B_n, V_n = construct_jacobian_parallel(
                size_y, size_z, size_p, m, N, rk, t_span, y, y_tilde, z_tilde, para, alpha)
            jacobian_time += (time.time() - start_time_jacobian)
            jacobian_count += 1
            start_time_reduce_jacobian = time.time()
            # compute each necessary element in the reduced BABD system
            A, C, H, b = reduce_jacobian_parallel(size_y, size_z, size_p, m, N, W, D, J, V, f_a, f_b)
            reduce_jacobian_time += (time.time() - start_time_reduce_jacobian)
            reduce_jacobian_count += 1
            # solve the BABD system
            # perform the partition factorization on the Jacobian matrix with qr decomposition
            index, R, E, J_reduced, G, d, A_tilde, C_tilde, H_tilde, b_tilde = \
                solve_babd_system.partition_factorization_parallel(size_y, size_p, M, N, A, C, H, b)
            # construct the partitioned Jacobian system
            sol = solve_babd_system.construct_babd_mshoot(
                size_y, 0, size_p, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, V_n, -r_bc)
            # perform the qr decomposition to transfer the system
            solve_babd_system.qr_decomposition(size_y, size_p, M + 1, sol)
            # perform the backward substitution to obtain the solution to the linear system of Newton's method
            solve_babd_system.backward_substitution(M + 1, sol)
            # obtain the solution from the reduced BABD system
            delta_s_r, delta_para = solve_babd_system.recover_babd_solution(M, size_y, 0, size_p, sol)
            # get the solution to the BABD system
            delta_y = solve_babd_system.partition_backward_substitution_parallel(
                size_y, size_p, M, N, index, delta_s_r, delta_para, R, G, E, J_reduced, d)
            start_time_recover_babd = time.time()
            # recover delta_k from the reduced BABD system
            delta_k, delta_y_dot, delta_z_tilde = recover_delta_k_parallel(
                size_y, size_z, m, N, delta_y, delta_para, f_a, J, V, W)
            recover_babd_time += (time.time() - start_time_recover_babd)
            recover_babd_count += 1
            # line search
            alpha0 = 1
            for i in range(max_linesearch):
                y_new = y + alpha0 * delta_y
                y_dot_new = y_dot + alpha0 * delta_y_dot
                z_tilde_new = z_tilde + alpha0 * delta_z_tilde
                para_new = para + alpha0 * delta_para
                start_time_residual = time.time()
                f_q_new, _, _, _, _ = compute_f_q_parallel(
                    size_y, size_z, size_p, m, N, rk, t_span, y_new, y_dot_new, z_tilde_new, para_new, alpha)
                residual_time += (time.time() - start_time_residual)
                residual_count += 1
                norm_f_q_new = np.linalg.norm(f_q_new, np.inf)
                if norm_f_q_new < norm_f_q:
                    y = y_new
                    y_dot = y_dot_new
                    z_tilde = z_tilde_new
                    para = para_new
                    # y, y_dot, z_tilde, p0 = vec_to_mat(size_y, size_z, size_p, m, N, q0)
                    break
                alpha0 /= 2
        # check whether the iteration exceeds the maximum number
        if alphacal >= (max_iter - 1) or iter_time >= (max_iter - 1) \
                or N > max_nodes or mesh_it > max_mesh or N < min_nodes:
            print("\talpha = {}, reach the maximum iteration numbers and the problem does not converge!".format(alpha))
            success_flag = 0
            break
        start_time_segment_residual = time.time()
        if not residual2:
            residual, max_residual = \
                compute_segment_residual_parallel(
                    size_y, size_z, size_p, m, N, t_span, y, y_dot, z_tilde, para, alpha, tol)
        else:
            residual, max_residual = \
                compute_segment_residual_parallel2(
                    size_y, size_z, size_p, m, N, t_span, y, y_dot, z_tilde, para, alpha, tol)
        segment_residual_time += (time.time() - start_time_segment_residual)
        segment_residual_count += 1
        print('\tresidual error = {}, number of nodes = {}.'.format(max_residual, N))
        if max_residual > 1:
            z = recover_solution_parallel(size_z, m, N, z_tilde)
            N, t_span, y, z = remesh(size_y, size_z, N, t_span, y, z, residual)
            mesh_it += 1
            start_time_initial_input = time.time()
            y, y_dot, z_tilde = form_initial_input_parallel(size_y, size_z, size_p, m, N, y, z, para, rk)
            initial_input_time += (time.time() - start_time_initial_input)
            initial_input_count += 1
            print("\tRemeshed the problem. Number of nodes = {}".format(N))
            if mesh_sanity_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
                print("\talpha = {}, number of nodes is beyond the limit!".format(
                    alpha))
                success_flag = 0
                break
        else:
            print("\talpha = {}, solution is found with residual error = {}. Number of nodes = {}".format(
                alpha, max_residual, N))
            if alpha <= alpha_m:
                print("Final solution is found, alpha = {}. Number of nodes: {}".format(alpha, N))
                break
            alpha *= beta
    total_time = time.time() - solver_start_time
    print("Maximum residual: {}".format(max_residual))
    print("Elapsed time: {}".format(total_time))
    # recover the final solution
    z = recover_solution_parallel(size_z, m, N, z_tilde)
    # write benchmark result
    benchmark_dir = "./benchmark_performance/"
    # create the directory
    pathlib.Path(benchmark_dir).mkdir(0o755, parents=True, exist_ok=True)
    benchmark_file = benchmark_dir + example_name + "_parallel_benchmark_M_{}.data".format(M)
    write_benchmark_result(benchmark_file,
                           initial_input_time, initial_input_count,
                           residual_time, residual_count,
                           jacobian_time, jacobian_count,
                           reduce_jacobian_time, reduce_jacobian_count,
                           recover_babd_time, recover_babd_count,
                           segment_residual_time, segment_residual_count,
                           total_time)
    # if alpha <= alpha_m and success_flag:
    if alpha <= alpha_m:
        # write solution to the output file
        error = bvpdae_write_data(output_file, N, size_y, size_z, size_p, t_span, y, z, para)
        if error != 0:
            print('Write file failed.')
    # record the solved example
    with open("test_results_m_{}_residual2_{}_residualType_{}_scaleTime_{}_scaleInitial_{}.txt".format(
            m, residual2, residual_type, scale_by_time, scale_by_initial), 'a') as f:
        if alpha <= alpha_m and success_flag:
            f.write("{} solved successfully. alpha = {}. Elapsed time: {}(s).\n".format(
                example_name, alpha, total_time))
            print("Problem solved successfully.")
        else:
            f.write("{} solved unsuccessfully. alpha = {}. Elapsed time: {}(s).\n".format(
                example_name, alpha, total_time))
            print("Problem solved unsuccessfully.")
    # plot the result
    plot_result(size_y, size_z, t_span, y, z)
    return 


# start implementations for forming initial input


'''
       Form the initial input for the collocation algorithm. The inputs for the solver
       are  usually just ODE variables, DAE variables, and parameter variables. However, 
       the inputs to the collocation solver should be ODE variables at each time node, the 
       derivatives of the ODE variables and the value of DAE variables at each collocation 
       point.
       Input:
            size_y: number of ODE variables of the BVP-DAE problem
            size_z: number of DAE variables of the BVP-DAE problem
            m: number of collocation points used in the algorithm
            N: number of time nodes of the system
            y0: values of the ODE variables in matrix form
                dimension: N x size_y, where each row corresponds the values at each time node
            z0: values of the DAE variables in matrix form
                dimension: N x size_z, where each row corresponds the values at each time node
            p0: values of the parameter variables in vector form
                dimension: N x size_z, where each row corresponds the values at each time node
            rk: collocation coefficients which is runge-kutta coefficients usually
       Output:
            y: values of the ODE variables in matrix form
               dimension: N x size_y, where each row corresponds the values at each time node
            y_dot: values of the derivatives of ODE variables in row dominant matrix form
                dimension: (N - 1) * m x size_y, where each row corresponds the values at each time node. 
                The index for the jth collocation point from the ith time node is i * m + j.
            z_tilde: values of the DAE variables in row dominant matrix form
               dimension: (N - 1) * m x size_z, where each row corresponds the values at each time node.
               The index for the jth collocation point from the ith time node is i * m + j.
'''


def form_initial_input_parallel(size_y, size_z, size_p, m, N, y0, z0, p0, rk):
    # lobatto coefficients
    c = rk.c
    # warp dimension for CUDA kernel
    grid_dims_2d = ((N - 1) + TPB_N - 1) // TPB_N, (m + TPB_m - 1) // TPB_m
    block_dims_2d = TPB_N, TPB_m
    # transfer memory from CPU to GPU
    d_c = cuda.to_device(c)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_p0 = cuda.to_device(p0)
    # create holder for temporary variables
    d_y_temp = cuda.device_array(((N - 1) * m, size_y), dtype=np.float64)
    # create holder for output variables
    d_y_dot = cuda.device_array(((N - 1) * m, size_y), dtype=np.float64)
    d_z_tilde = cuda.device_array(((N - 1) * m, size_z), dtype=np.float64)
    form_initial_input_kernel[grid_dims_2d, block_dims_2d](
        d_c, size_y, size_z, size_p, m, N, d_y0, d_z0, d_p0, d_y_temp, d_y_dot, d_z_tilde)
    # transfer the memory back to CPU
    y_dot = d_y_dot.copy_to_host()
    z_tilde = d_z_tilde.copy_to_host()
    return y0, y_dot, z_tilde


'''
    Kernel function for forming initial input.
'''


@cuda.jit
def form_initial_input_kernel(d_c, size_y, size_z, size_p, m, N, d_y0, d_z0, d_p0, d_y_temp, d_y_dot, d_z_tilde):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    shared_d_c = cuda.shared.array(shape=(m_collocation,), dtype=float64)
    shared_d_y0 = cuda.shared.array(shape=(N_shared, global_size_y), dtype=float64)
    shared_d_z0 = cuda.shared.array(shape=(N_shared, global_size_z), dtype=float64)
    shared_d_p0 = cuda.shared.array(shape=(global_size_p, ), dtype=float64)

    # cuda thread index
    i, j = cuda.grid(2)
    tx = cuda.threadIdx.x  # thread index in x direction
    ty = cuda.threadIdx.y  # thread index in y direction

    if i < (N - 1) and j < m:
        # continue if i and j is inside of valid mesh boundary
        shared_d_c[ty] = d_c[j]
        if ty == 0:
            # only need 1 dimensional memory load here
            # let the first thread in y direction do the work
            for k in range(size_y):
                shared_d_y0[tx + 1, k] = d_y0[i + 1, k]
            for k in range(size_z):
                shared_d_z0[tx + 1, k] = d_z0[i + 1, k]
            for k in range(size_p):
                shared_d_p0[k] = d_p0[k]
            if tx == 0:
                # load the additional column in shared memory using the first thread in x direction
                for k in range(size_y):
                    shared_d_y0[0, k] = d_y0[i, k]
                for k in range(size_z):
                    shared_d_z0[0, k] = d_z0[i, k]
        cuda.syncthreads()

        for k in range(size_y):
            d_y_temp[i * m + j, k] = (1 - shared_d_c[ty]) * shared_d_y0[tx, k] + shared_d_c[ty] * shared_d_y0[tx + 1, k]
        for k in range(size_z):
            d_z_tilde[i * m + j, k] = (1 - shared_d_c[ty]) * shared_d_z0[tx, k] + shared_d_c[ty] * shared_d_z0[tx + 1, k]
        _abvp_f(d_y_temp[i * m + j, 0: size_y], d_z_tilde[i * m + j, 0: size_z], shared_d_p0[0: size_p],
                d_y_dot[i * m + j, 0: size_y])
    return

# finish implementations for forming initial input


# start implementation for computing the residual of the system


'''
    Compute the residual of the BVP-DAE using collocation method.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        N: number of time nodes of the system
        rk: collocation coefficients which is runge-kutta coefficients usually
        t_span: time span of the problem
        y: values of the ODE variables in matrix form
           dimension: N x size_y, where each row corresponds the values at each time node
        y_dot: values of the derivatives of ODE variables in row dominant matrix form
            dimension: (N - 1) * m x size_y, where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is i * m + j.
        z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (N - 1) * m x size_z, where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
        p: values of the parameter variables in vector form
           dimension: size_p
        alpha: continuation parameter of the Newton method
    Output:
        f_q_residual : residual of the system in vector form
            dimension: N * size_y + (N - 1) * m * (size_y + size_z) + size_p
        y_tilde: values of ODE variables y at each collocation point in row dominant matrix for
            dimension: m * (N - 1), size_y, where each row corresponds the values at each collocation point
        f_a : matrix of the residual f_a for each time node in row dominant matrix form
            dimension: (N - 1) x m * (size_y + size_z), where each row corresponds the values at each time node
        f_b : matrix of the residual f_b for each time node in row dominant matrix form
            dimension: (N - 1) x size_y, where each row corresponds the values at each time node
        r_bc : boundary conditions of the system in vector form
            dimension: size_y + size_p
'''


def compute_f_q_parallel(size_y, size_z, size_p, m, N, rk, t_span, y, y_dot, z_tilde, p, alpha):
    # calculate the y values on collocation points
    # combine those two kernels into one maybe?
    # grid dimension of the warp of CUDA
    grid_dims_1d = ((N - 1) + TPB - 1) // TPB
    block_dims_1d = TPB
    grid_dims_2d = ((N - 1) + TPB_N - 1) // TPB_N, (m + TPB_m - 1) // TPB_m
    block_dims_2d = TPB_N, TPB_m
    # coefficient A and B from lobatto coefficients
    a = rk.A
    b = rk.b
    # transfer memory from CPU to GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    # container to hold temporary variables
    d_sum_j = cuda.device_array((m * (N - 1), size_y), dtype=np.float64)
    # container to hold output variables y_tilde
    d_y_tilde = cuda.device_array((m * (N - 1), size_y), dtype=np.float64)
    # calculate the y variables at collocation points with the kernel function
    collocation_update_kernel[grid_dims_2d, block_dims_2d](
        size_y, m, N, d_a, d_t_span, d_y, d_y_dot, d_sum_j, d_y_tilde)
    # load the memory back from GPU to CPU
    y_tilde = d_y_tilde.copy_to_host()
    # transfer memory from CPU to GPU
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # container to hold temporary variables
    # sum_i = np.zeros((N - 1, size_y), dtype=np.float64)
    # d_sum_i = cuda.to_device(sum_i)
    d_sum_i = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # container to hold derivatives, initially zeros
    # did't initialize with zeros, no need maybe?
    d_r_h = cuda.device_array((m * (N - 1), size_y), dtype=np.float64)
    d_r_g = cuda.device_array((m * (N - 1), size_z), dtype=np.float64)
    # container to hold residuals, be careful about the dimensions
    d_f_a = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_f_b = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # calculate the f_a and f_b at each time node with the kernel function
    compute_f_q_kernel1[grid_dims_2d, block_dims_2d](
        size_y, size_z, m, N, d_y_dot, d_y_tilde, d_z_tilde, d_p, alpha, d_f_a, d_r_h, d_r_g)
    compute_f_q_kernel2[grid_dims_1d, block_dims_1d](d_b, size_y, m, N, d_t_span, d_y, d_y_dot, d_sum_i, d_f_b)
    # load the memory back from GPU to CPU
    f_a = d_f_a.copy_to_host()
    f_b = d_f_b.copy_to_host()
    # calculate the boundary conditions
    r_bc = np.zeros((size_y + size_p), dtype=np.float64)
    # this boundary function is currently on CPU
    _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r_bc)
    # form the residual
    # performed on CPU currently
    f_q_residual = np.zeros((N * size_y + (N - 1) * m * (size_y + size_z) + size_p), dtype=np.float64)
    # form the residual at the first N - 1 time nodes
    # can be parallelled ?, only return the max norm of the residual
    for i in range(N - 1):
        start_index_f_a = i * (size_y + m * (size_y + size_z))
        for j in range((m * (size_y + size_z))):
            f_q_residual[start_index_f_a + j] = f_a[i, j]
        start_index_f_b = start_index_f_a + m * (size_y + size_z)
        for j in range(size_y):
            f_q_residual[start_index_f_b + j] = f_b[i, j]
    # form the residual of the boundary conditions
    start_index_r = (N - 1) * (size_y + m * (size_y + size_z))
    for j in range(size_y + size_p):
        f_q_residual[start_index_r + j] = r_bc[j]
    return f_q_residual, y_tilde, f_a, f_b, r_bc


'''
    Kernel function to compute each part of the residual of the system.
    d_f_a: N - 1 x m * (size_y + size_z)
    d_f_b: N - 1 x size_y
    d_r_h: (N - 1) * m x size_y
    d_r_g: (N - 1) * m x size_z
'''


@cuda.jit
def compute_f_q_kernel1(size_y, size_z, m, N, d_y_dot, d_y_tilde, d_z_tilde, d_p, alpha, d_f_a, d_r_h, d_r_g):
    i, j = cuda.grid(2)
    if i < (N - 1) and j < m:
        _abvp_f(d_y_tilde[i * m + j, 0: size_y], d_z_tilde[i * m + j, 0: size_z], d_p, d_r_h[i * m + j, 0: size_y])
        _abvp_g(d_y_tilde[i * m + j, 0: size_y], d_z_tilde[i * m + j, 0: size_z], d_p, alpha,
                d_r_g[i * m + j, 0: size_z])
        # calculate the residual $h - y_dot$ on each collocation point
        for k in range(size_y):
            d_r_h[i * m + j, k] -= d_y_dot[i * m + j, k]
        # copy the result to f_a of the collocation point
        # to the corresponding position
        # to do
        start_index_y = j * (size_y + size_z)
        start_index_z = start_index_y + size_y
        # copy the residual of h and g to the corresponding positions
        for k in range(size_y):
            d_f_a[i, start_index_y + k] = d_r_h[i * m + j, k]
        for k in range(size_z):
            d_f_a[i, start_index_z + k] = d_r_g[i * m + j, k]


@cuda.jit
def compute_f_q_kernel2(d_b, size_y, m, N, d_t_span, d_y, d_y_dot, d_sum_i, d_f_b):
    i = cuda.grid(1)
    if i < (N - 1):
        # initialize d_sum_i as zeros
        for k in range(size_y):
            d_sum_i[i, k] = 0
        for j in range(m):
            for k in range(size_y):
                d_sum_i[i, k] += d_b[j] * d_y_dot[i * m + j, k]
        delta_t_i = d_t_span[i + 1] - d_t_span[i]
        for k in range(size_y):
            d_f_b[i, k] = d_y[i + 1, k] - d_y[i, k] - delta_t_i * d_sum_i[i, k]


'''
    Kernel method for computing the values of y variables on each collocation point.
'''


@cuda.jit
def collocation_update_kernel(size_y, m, N, d_a, d_t_span, d_y, d_y_dot, d_sum_j, d_y_tilde):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    shared_d_a = cuda.shared.array(shape=(m_collocation, m_collocation), dtype=float64)
    shared_d_y = cuda.shared.array(shape=(TPB_N, global_size_y), dtype=float64)
    shared_d_y_dot = cuda.shared.array(shape=(global_y_shared_size, global_size_y), dtype=float64)

    # cuda thread index
    i, j = cuda.grid(2)
    tx = cuda.threadIdx.x  # thread index in x direction
    ty = cuda.threadIdx.y  # thread index in y direction

    if i < (N - 1) and j < m:
        if j == 0:
            for k in range(size_y):
                # load d_y to shared memory using the first thread in y direction
                shared_d_y[tx, k] = d_y[i, k]
        for k in range(m):
            # load coefficients a to shared memory first
            shared_d_a[ty, k] = d_a[j, k]
        for k in range(size_y):
            # load d_y_dot to shared memory
            # each thread loads the corresponding row
            shared_d_y_dot[tx * m + ty, k] = d_y_dot[i * m + j, k]
        cuda.syncthreads()

        delta_t_i = d_t_span[i + 1] - d_t_span[i]
        # loop j for each collocation point t_ij
        # zero the initial value
        for l in range(size_y):
            d_sum_j[i * m + j, l] = 0
        # loop k to perform the integral on all collocation points
        for k in range(m):
            # loop l to loop over all the y variables
            for l in range(size_y):
                d_sum_j[i * m + j, l] += shared_d_a[ty, k] * shared_d_y_dot[tx * m + k, l]
        # loop l to loop over all the y variables to update the result
        for l in range(size_y):
            d_y_tilde[i * m + j, l] = shared_d_y[tx, l] + delta_t_i * d_sum_j[i * m + j, l]


# finish the implementation of computing the residual


# start the implementation of constructing the Jacobian matrix


'''
    Compute each small matrix elements in the Jacobian of the system.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        N: number of time nodes of the system
        rk: collocation coefficients which is runge-kutta coefficients usually
        t_span: time span of the problem
        y: values of the ODE variables in matrix form
           dimension: N x size_y, where each row corresponds the values at each time node
        y_tilde: values of ODE variables y at each collocation point in row dominant matrix form
            dimension: m * (N - 1), size_y, where each row corresponds the values at each collocation point.
            The index for the jth collocation point from the ith time node is i * m + j.
        z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (N - 1) * m x size_z, where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
        p: values of the parameter variables in vector form
           dimension: size_p
        alpha: continuation parameter of the Newton method
    Output:
        J: the J matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_y
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
        V: the V matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_p
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
        D: the D matrix element in the Jacobian matrix in row dominant matrix form
           dimension: (N - 1) * size_y x m * (size_y + size_z)
           each size_y x m * (size_y + size_z) corresponds to a matrix block at a time node
        W: the D matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x m * (size_y + size_z)
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
        B_0: derivatives of boundary conditions w.r.t. ODE variables at initial time
             dimension: size_y + size_p x size_y
        B_n: derivatives of boundary conditions w.r.t. ODE variables at final time
             dimension: size_y + size_p x size_y
        V_n: derivatives of boundary conditions w.r.t. parameter varaibels
             dimension: size_y + size_p x size_p
'''


def construct_jacobian_parallel(size_y, size_z, size_p, m, N, rk, t_span, y, y_tilde, z_tilde, p, alpha):
    # grid dimension of the warp of CUDA
    grid_dims = ((N - 1) + TPB - 1) // TPB
    grid_dims_2d = ((N - 1) + TPB_N - 1) // TPB_N, (m + TPB_m - 1) // TPB_m
    block_dims_2d = TPB_N, TPB_m
    # lobatto coefficients
    b = rk.b
    a = rk.A
    # transfer memory from CPU to GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_t_span = cuda.to_device(t_span)
    d_p = cuda.to_device(p)
    d_y_tilde = cuda.to_device(y_tilde)
    d_z_tilde = cuda.to_device(z_tilde)
    # container to hold the derivatives
    '''
        large row dominated matrix 
        start_row_index : i * m * size_y + j * size_y
        end_row_index : start_index + size_y
        d_d_h[start_row_index : end_row_index, :] can access the derivatives of the ODE
        equations at the jth collocation node of the ith time span
        d_d_g[start_row_index : end_row_index, :] can access the derivatives of the DAE
        equations at the jth collocation node of the ith time span
    '''
    # no zero initialize here, initialize it in the kernel
    d_d_h = cuda.device_array((size_y * m * (N - 1), (size_y + size_z + size_p)), dtype=np.float64)
    d_d_g = cuda.device_array((size_z * m * (N - 1), (size_y + size_z + size_p)), dtype=np.float64)
    '''
        large row dominant matrix 
        start_index : i * m * (size_y + size_z + size_p) + j * (size_y + size_z)
        end_index : start_index + (size_y + size_z + size_p)
        d_J[start_index : end_index, 0 : size_y] can access the elements associated with
        the jth collocation node of the ith time span
        d_V[start_index : end_index, 0 : size_p] can access the elements associated with
        the jth collocation node of the ith time span
    '''
    # holder for the output variables
    d_J = cuda.device_array((m * (size_y + size_z) * (N - 1), size_y), dtype=np.float64)
    d_V = cuda.device_array((m * (size_y + size_z) * (N - 1), size_p), dtype=np.float64)
    d_W = cuda.device_array((m * (size_y + size_z) * (N - 1), m * (size_y + size_z)), dtype=np.float64)
    # no zero initialization, initialize it in the kernel
    d_D = cuda.device_array(((N - 1) * size_y, m * (size_y + size_z)), dtype=np.float64)
    # construct the J, V, W, and D matrix on CUDA kernel
    construct_jacobian_kernel[grid_dims_2d, block_dims_2d](
        d_a, d_b, size_y, size_z, size_p, m, N, d_t_span, d_y_tilde, d_z_tilde, d_p, alpha, d_d_h, d_d_g,
        d_J, d_V, d_D, d_W)
    # compute the derivative of the boundary conditions
    d_r = np.zeros(((size_y + size_p), (size_y + size_y + size_p)), dtype=np.float64)
    y_i = y[0, :]  # y values at the initial time
    y_f = y[N - 1, :]  # y values at the final time
    _abvp_Dr(y_i, y_f, p, d_r)
    B_0 = d_r[0: size_y + size_p, 0: size_y]  # B_1 in the paper
    B_n = d_r[0: size_y + size_p, size_y: size_y + size_y]  # B_N in the paper
    V_n = d_r[0: size_y + size_p, size_y + size_y: size_y + size_y + size_p]  # V_N in the paper
    return d_J.copy_to_host(), d_V.copy_to_host(), d_D.copy_to_host(), d_W.copy_to_host(), B_0, B_n, V_n


'''
    Kernel function for computing each element J, V, D, W in the Jacobian matrix
    d_d_h : size_y x m * (N - 1) * (size_y + size_z + size_p)
    d_d_g : size_z x m * (N - 1) * (size_y + size_z + size_p)
    d_J : m * (size_y + size_z) * (N - 1) x size_y
    d_V : m * (size_y + size_z) * (N - 1) x size_p
    d_D : (N - 1) * size_y x m * (size_y + size_z)
    d_W : m * (size_y + size_z) * (N - 1) x m * (size_y + size_z)
'''


@cuda.jit()
def construct_jacobian_kernel(d_a, d_b, size_y, size_z, size_p, m, N, d_t_span, d_y_tilda, d_z_tilda, d_p, alpha,
                              d_d_h, d_d_g, d_J, d_V, d_D, d_W):
    i, j = cuda.grid(2)
    if i < (N - 1) and j < m:
        delta_t_i = d_t_span[i + 1] - d_t_span[i]
        # for j in range(m):
        # the block index for each derivative of d_h
        start_row_index_d_h = i * m * size_y + j * size_y
        end_row_index_d_h = start_row_index_d_h + size_y
        # zero initialize the derivative matrix
        for k in range(start_row_index_d_h, end_row_index_d_h):
            for l in range(0, size_y + size_z + size_p):
                d_d_h[k, l] = 0
        # compute the derivatives
        _abvp_Df(d_y_tilda[i * m + j, 0: size_y], d_z_tilda[i * m + j, 0: size_z], d_p,
                 d_d_h[start_row_index_d_h: end_row_index_d_h, 0: size_y + size_z + size_p])
        # the block index for each derivative of d_g
        start_row_index_d_g = i * m * size_z + j * size_z
        end_row_index_d_g = start_row_index_d_g + size_z
        # zero initialize the derivative matrix
        for k in range(start_row_index_d_g, end_row_index_d_g):
            for l in range(0, size_y + size_z + size_p):
                d_d_g[k, l] = 0
        # compute the derivatives
        _abvp_Dg(d_y_tilda[i * m + j, 0: size_y], d_z_tilda[i * m + j, 0: size_z], d_p, alpha,
                 d_d_g[start_row_index_d_g: end_row_index_d_g, 0: size_y + size_z + size_p])
        '''
            indexing for each derivatives
            h_y = d_d_h[start_row_index_d_h: end_row_index_d_h, 0: size_y])
            h_z = d_d_h[start_row_index_d_h: end_row_index_d_h, size_y: size_y + size_z])
            h_p = d_d_h[start_row_index_d_h: end_row_index_d_h, size_y + size_z: size_y + size_z + size_p])
            g_y = d_d_g[start_row_index_d_g: end_row_index_d_g, 0: size_y]
            g_z = d_d_g[start_row_index_d_g: end_row_index_d_g, size_y: size_y + size_z]
            g_p = d_d_g[start_row_index_d_g: end_row_index_d_g, size_y + size_z: size_y + size_z + size_p]
        '''
        # construct the J and V matrix
        start_index_JV_h = i * m * (size_y + size_z) + j * (size_y + size_z)
        start_index_JV_g = start_index_JV_h + size_y
        for k in range(size_y):
            for l in range(size_y):
                d_J[start_index_JV_h + k, l] = d_d_h[start_row_index_d_h + k, l]
            for l in range(size_p):
                d_V[start_index_JV_h + k, l] = d_d_h[start_row_index_d_h + k, size_y + size_z + l]
        for k in range(size_z):
            for l in range(size_y):
                d_J[start_index_JV_g + k, l] = d_d_g[start_row_index_d_g + k, l]
            for l in range(size_p):
                d_V[start_index_JV_g + k, l] = d_d_g[start_row_index_d_g + k, size_y + size_z + l]
        # construct the D matrix
        start_row_index_D = i * size_y
        start_col_index_D = j * (size_y + size_z)
        for k in range(size_y):
            for l in range(size_y + size_z):
                if k == l:
                    d_D[start_row_index_D + k, start_col_index_D + l] = delta_t_i * d_b[j]
                else:
                    d_D[start_row_index_D + k, start_col_index_D + l] = 0.0
        # construct the W matrix
        # j associates the corresponding row block
        # start_row_index_W = i * m * (size_y + size_z) + j * (size_y + size_z)
        # loop through the m column blocks
        # each column block is size (size_y + size_z) x (size_y + size_z)
        for k in range(m):
            # start row index for the top block in W matrix
            start_row_index_W_top = i * m * (size_y + size_z) + j * (size_y + size_z)
            # start row index for the bottom block in W matrix
            start_row_index_W_bot = start_row_index_W_top + size_y
            # start column index for the left block in W matrix
            start_col_index_W_left = k * (size_y + size_z)
            # start column index for the right block in W matrix
            start_col_index_W_right = start_col_index_W_left + size_y
            # for the diagonal block
            if k == j:
                # top left block: -I + delta_t_i * a[j, k] * h_y
                for ii in range(size_y):
                    for jj in range(size_y):
                        # diagonal element
                        if ii == jj:
                            d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                                -1 + delta_t_i * d_a[j, k] * d_d_h[start_row_index_d_h + ii, jj]
                        else:
                            d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                                delta_t_i * d_a[j, j] * d_d_h[start_row_index_d_h + ii, jj]
                # top right block: h_z
                for ii in range(size_y):
                    for jj in range(size_z):
                        d_W[start_row_index_W_top + ii, start_col_index_W_right + jj] = \
                            d_d_h[start_row_index_d_h + ii, size_y + jj]
                # bottom left block: delta_t_i * a[j, k] * g_y
                for ii in range(size_z):
                    for jj in range(size_y):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a[j, k] * d_d_g[start_row_index_d_g + ii, jj]
                # bottom right block: g_z
                for ii in range(size_z):
                    for jj in range(size_z):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_right + jj] = \
                            d_d_g[start_row_index_d_g + ii, size_y + jj]
            else:
                # top left block: delta_t_i * a[j, k] * h_y
                for ii in range(size_y):
                    for jj in range(size_y):
                        d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a[j, k] * d_d_h[start_row_index_d_h + ii, jj]
                # top right block: 0s
                for ii in range(size_y):
                    for jj in range(size_z):
                        d_W[start_row_index_W_top + ii, start_col_index_W_right + jj] = 0
                # bottom left block: delta_t_i * a[j, k] * g_y
                for ii in range(size_z):
                    for jj in range(size_y):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a[j, k] * d_d_g[start_row_index_d_g + ii, jj]
                # bottom right block: 0s
                for ii in range(size_z):
                    for jj in range(size_z):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_right + jj] = 0


'''
    Construct the reduced BABD system with a self-implemented LU factorization solver.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        N: number of time nodes of the system
        W: the D matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x m * (size_y + size_z)
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
        D: the D matrix element in the Jacobian matrix in row dominant matrix form
           dimension: (N - 1) * size_y x m * (size_y + size_z)
           each size_y x m * (size_y + size_z) corresponds to a matrix block at a time node
        J: the J matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_y
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
        V: the V matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_p
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
        f_a : matrix of the residual f_a for each time node in row dominant matrix form
            dimension: (N - 1) x m * (size_y + size_z), where each row corresponds the values at each time node
        f_b : matrix of the residual f_b for each time node in row dominant matrix form
            dimension: (N - 1) x size_y, where each row corresponds the values at each time node
    Output:
        A : the A matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        C : the C matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        H : the H matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_p
            each size_y x size_p corresponds to a matrix block at a time node
        b : the b vector element of the residual in the reduced BABD system in a row dominant matrix form
            dimension: (N - 1) x size_y
            each row vector with size size_y corresponds to a vector block at a time node
'''


def reduce_jacobian_parallel(size_y, size_z, size_p, m, N, W, D, J, V, f_a, f_b):
    # compute the grid dimensions of CUDA warp
    grid_dims = ((N - 1) + TPB - 1) // TPB
    # transfer memory from CPU to GPU
    d_W = cuda.to_device(W)
    d_D = cuda.to_device(D)
    d_J = cuda.to_device(J)
    d_V = cuda.to_device(V)
    d_f_a = cuda.to_device(f_a)
    d_f_b = cuda.to_device(f_b)
    # holder for output variables
    d_A = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_C = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_H = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_b = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_L = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_U = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_cpy = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_c_J = cuda.device_array(((N - 1) * m * (size_y + size_z), size_y), dtype=np.float64)
    d_y_J = cuda.device_array(((N - 1) * m * (size_y + size_z), size_y), dtype=np.float64)
    d_x_J = cuda.device_array(((N - 1) * m * (size_y + size_z), size_y), dtype=np.float64)
    d_D_W_J = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_c_V = cuda.device_array(((N - 1) * m * (size_y + size_z), size_p), dtype=np.float64)
    d_y_V = cuda.device_array(((N - 1) * m * (size_y + size_z), size_p), dtype=np.float64)
    d_x_V = cuda.device_array(((N - 1) * m * (size_y + size_z), size_p), dtype=np.float64)
    d_D_W_V = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_c_f_a = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_y_f_a = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_x_f_a = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_D_W_f_a = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    # perform the jacobian reduction in parallel with the GPU kernel
    reduce_jacobian_parallel_kernel[grid_dims, TPB](eps, m, N, size_y, size_z, size_p, d_W, d_D, d_J, d_V, d_f_a, d_f_b,
                                                    d_P, d_L, d_U, d_cpy, d_c_J, d_y_J, d_x_J, d_D_W_J,
                                                    d_c_V, d_y_V, d_x_V, d_D_W_V, d_c_f_a, d_y_f_a, d_x_f_a, d_D_W_f_a,
                                                    d_A, d_C, d_H, d_b)
    return d_A.copy_to_host(), d_C.copy_to_host(), d_H.copy_to_host(), d_b.copy_to_host()


'''
    Kernel function for computing each element A, C, H, b in the reduced Jacobian matrix.
'''


@cuda.jit()
def reduce_jacobian_parallel_kernel(eps, m, N, size_y, size_z, size_p, d_W, d_D, d_J, d_V, d_f_a, d_f_b,
                                    d_P, d_L, d_U, d_cpy, d_c_J, d_y_J, d_x_J, d_D_W_J,
                                    d_c_V, d_y_V, d_x_V, d_D_W_V, d_c_f_a, d_y_f_a, d_x_f_a, d_D_W_f_a,
                                    d_A, d_C, d_H, d_b):
    i = cuda.grid(1)
    if i < (N - 1):
        # start row index of the W element
        start_row_index_W = i * m * (size_y + size_z)
        # end row index of the W element
        end_row_index_W = start_row_index_W + m * (size_y + size_z)
        # start row index of the D element
        start_row_index_D = i * size_y
        # end row index of the D element
        end_row_index_D = start_row_index_D + size_y
        # start row index of the J element
        start_row_index_J = i * m * (size_y + size_z)
        # end row index of the J element
        end_row_index_J = start_row_index_J + m * (size_y + size_z)
        # start row index of the V element
        start_row_index_V = i * m * (size_y + size_z)
        # end row index of the V element
        end_row_index_V = start_row_index_V + m * (size_y + size_z)
        # start row index for A, C, and H
        start_row_index_ACH = i * size_y
        # end row index for A, C, and H
        end_row_index_ACH = start_row_index_ACH + size_y
        # perform LU decomposition of matrix W and save the results
        # P * W = L * U
        matrix_factorization_cuda.lu(d_W[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                     d_cpy[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                     d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                     d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                     d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)], eps)
        # A = -I + D * W^(-1) * J
        # compute W^(-1) * J = X first
        # compute P * J first, the result of the product is saved in d_c_J
        matrix_operation_cuda.mat_mul(d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                      d_J[start_row_index_J: end_row_index_J, 0: size_y],
                                      d_c_J[start_row_index_J: end_row_index_J, 0: size_y])
        # first, forward solve the linear system L * (U * X) = (P * J), and the result is saved in d_y_J
        matrix_factorization_cuda.forward_solve_mat(d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                                    d_c_J[start_row_index_J: end_row_index_J, 0: size_y],
                                                    d_y_J[start_row_index_J: end_row_index_J, 0: size_y], eps)
        # first, backward solve the linear system U * X = Y, and the result is saved in d_x_J
        # X = W^(-1) * J
        matrix_factorization_cuda.backward_solve_mat(d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                                     d_y_J[start_row_index_J: end_row_index_J, 0: size_y],
                                                     d_x_J[start_row_index_J: end_row_index_J, 0: size_y], eps)
        # perform D * X
        matrix_operation_cuda.mat_mul(d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
                                      d_x_J[start_row_index_J: end_row_index_J, 0: size_y],
                                      d_D_W_J[start_row_index_D: end_row_index_D, 0: size_y])
        # final step, A = -I + D * X
        # nested for loops, row-wise first, column-wise next
        for j in range(size_y):
            for k in range(size_y):
                if j == k:
                    d_A[start_row_index_ACH + j, k] = -1 + d_D_W_J[start_row_index_D + j, k]
                else:
                    d_A[start_row_index_ACH + j, k] = d_D_W_J[start_row_index_D + j, k]
        # H = D * W^(-1) * V
        # compute W^(-1) * V = X first
        # compute P * V first, the result of the product is saved in d_c_V
        matrix_operation_cuda.mat_mul(d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                      d_V[start_row_index_V: end_row_index_V, 0: size_p],
                                      d_c_V[start_row_index_V: end_row_index_V, 0: size_p])
        # first, forward solve the linear system L * (U * X) = (P * V), and the result is saved in d_y_V
        matrix_factorization_cuda.forward_solve_mat(
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_c_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_y_V[start_row_index_V: end_row_index_V, 0: size_p], eps)
        # first, backward solve the linear system U * X = Y, and the result is saved in d_x_V
        # X = W^(-1) * V
        matrix_factorization_cuda.backward_solve_mat(
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_y_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_x_V[start_row_index_V: end_row_index_V, 0: size_p], eps)
        # final step, perform D * X, then we get the results H
        matrix_operation_cuda.mat_mul(d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
                                      d_x_V[start_row_index_V: end_row_index_V, 0: size_p],
                                      d_H[start_row_index_ACH: end_row_index_ACH, 0: size_p])
        # C = I
        # nested for loops, row-wise first, column-wise next
        for j in range(size_y):
            for k in range(size_y):
                if j == k:
                    d_C[start_row_index_ACH + j, k] = 1
                else:
                    d_C[start_row_index_ACH + j, k] = 0
        # b = -f_b - D * W^(-1) * f_a
        # compute W^(-1) * f_a = X first
        # compute P * f_a first, the result of the product is saved in d_c_f_a
        matrix_operation_cuda.mat_vec_mul(d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
                                          d_f_a[i, 0: m * (size_y + size_z)],
                                          d_c_f_a[i, 0: m * (size_y + size_z)])
        # first, forward solve the linear system L * (U * X) = (P * f_a), and the result is saved in d_y_f_a
        matrix_factorization_cuda.forward_solve_vec(
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_c_f_a[i, 0: m * (size_y + size_z)],
            d_y_f_a[i, 0: m * (size_y + size_z)], eps)
        # first, backward solve the linear system U * X = Y, and the result is saved in d_x_f_a
        # X = W^(-1) * f_a
        matrix_factorization_cuda.backward_solve_vec(
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_y_f_a[i, 0: m * (size_y + size_z)],
            d_x_f_a[i, 0: m * (size_y + size_z)], eps)
        # perform D * X
        matrix_operation_cuda.mat_vec_mul(d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
                                          d_x_f_a[i, 0: m * (size_y + size_z)],
                                          d_D_W_f_a[i, 0: size_y])
        # final step, b = -f_b - D * W^(-1) * f_a
        # for loop
        for j in range(size_y):
            d_b[i, j] = -d_f_b[i, j] - d_D_W_f_a[i, j]


# finish the implementation of constructing the Jacobian matrix


# start the implementation of partition factorization of the Jacobian


'''
    Partition the Jabobian matrix into M partitions, qr decomposition is performed 
    in each partition and generate the final BABD system to solve.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        M: number of partitions
        N: number of time nodes of the system
        A : the A matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        C : the C matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        H : the H matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_p
            each size_y x size_p corresponds to a matrix block at a time node
        b : the b vector element of the residual in the reduced BABD system in a row dominant matrix form
            dimension: (N - 1) x size_y
            each row vector with size size_y corresponds to a vector block at a time node
    Output:
        index: the index of the time node in each partition
               r_start = index[i] is the start index of the partition
               r_end = index[i + 1] - 1 is the end index of the partition
               index[0] = 1 which is the first node of the mesh
               index[-1] = N - 2 which is the second last node of the mesh
        R: the R matrix element from the partition factorization which contains 
           upper triangular matrix from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the R matrix block at a time node
        E: the E matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the E matrix block at a time node
        J_reduced: the J matrix element from the partition factorization from the qr decomposition 
                   in a row dominant matrix form
                   dimension: (N - 1) * size_y x size_p
                   each size_y x size_p matrix block corresponds to the J matrix block at a time node
        G: the G matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the G matrix block at a time node
        A_tilde: the A_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the A_tilde matrix block 
                 at the boundary of each partition
        C_tilde: the C_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the C_tilde matrix block 
                 at the boundary of each partition
        H_tilde: the H_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_p
                 each size_y x size_p matrix block corresponds to the H_tilde matrix block 
                 at the boundary of each partition
        b_tilde: the b_tilde vector element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M x size_y
                 each size_y vector block corresponds to the b_tilde vector block at the boundary of each partition
        d: the d vector element from the partition factorization from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) x size_y
           each size_y vector block corresponds to the d vector block at the boundary of each partition
        
        q_i : matrix from qr decomposition
            size: 2 * size_y x 2 * size
        q : big row dominated matrix
            size: (N - 1) * 2 * size_y x 2 * size_y
        q_t : container to hold the transpose of each q_i matrix
            size: (N - 1) * 2 * size_y x 2 * size_y
        r_i : matrix from qr decomposition
            size: 2 * size_y x size_y
        r : big row dominated matrix
            size: (N - 1) * 2 * size_y x size_y
'''


def partition_factorization_parallel(size_y, size_p, M, N, A, C, H, b):
    # compute the grid dimension of the warp
    grid_dims = (M + TPB - 1) // TPB
    # number of partitions to use: M
    # integer number of time nodes in each partition
    # M partitions / threads, (N - 1) time nodes divided into M blocks
    num_thread = (N - 1) // M
    indices = []
    for i in range(M):
        indices.append(i * num_thread)
    # final node to be processed is the (N - 2)th node
    # the final node is the (N - 1) = (N - 2) + 1 th node
    indices.append(N - 1)
    index = np.array(indices)
    # transfer the memory from CPU to GPU
    d_index = cuda.to_device(index)
    d_A = cuda.to_device(A)
    d_C = cuda.to_device(C)
    d_H = cuda.to_device(H)
    d_b = cuda.to_device(b)
    # holders for each intermediate matrix
    d_q = cuda.device_array(((N - 1) * 2 * size_y, 2 * size_y), dtype=np.float64)
    d_q_t = cuda.device_array(((N - 1) * 2 * size_y, 2 * size_y), dtype=np.float64)
    d_r = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)
    # holders for output variables
    d_r_j = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    # holders for each intermediate matrix
    d_C_tilde = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_G_tilde = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_H_tilde = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_b_tilde = cuda.device_array(((N - 1), size_y), dtype=np.float64)
    # holders for output variables
    d_E = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_J = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_G = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_d = cuda.device_array(((N - 1), size_y), dtype=np.float64)
    d_A_tilde_r_end = cuda.device_array((M * size_y, size_y), dtype=np.float64)
    d_C_tilde_r_end = cuda.device_array((M * size_y, size_y), dtype=np.float64)
    d_H_tilde_r_end = cuda.device_array((M * size_y, size_p), dtype=np.float64)
    d_b_tilde_r_end = cuda.device_array((M, size_y), dtype=np.float64)
    # container to hold the vstack of the C and A matrices
    d_C_A = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    # container to hold the intermediate variables for qr decomposition
    d_cpy = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    d_v = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    d_vv = cuda.device_array(((N - 1), 2 * size_y), dtype=np.float64)  # row dominated vector
    d_beta = cuda.device_array(((N - 1), size_y), dtype=np.float64)  # row dominated vector
    d_w_t = cuda.device_array(((N - 1), size_y), dtype=np.float64)  # row dominated vector
    d_u_t = cuda.device_array(((N - 1), 2 * size_y), dtype=np.float64)  # row dominated vector
    # machine precision
    eps = sys.float_info.epsilon
    # perform the parallel partition factorization on GPUs
    partition_factorization_kernel[grid_dims, TPB](
        eps, d_index, size_y, size_p, M, d_A, d_C, d_H, d_b, d_q, d_q_t, d_r, d_r_j, d_C_tilde, d_E, d_G, d_J, d_d,
        d_G_tilde, d_H_tilde, d_b_tilde, d_A_tilde_r_end, d_C_tilde_r_end, d_H_tilde_r_end, d_b_tilde_r_end, d_C_A,
        d_cpy, d_v, d_vv, d_beta, d_w_t, d_u_t)
    return index, d_r_j.copy_to_host(), d_E.copy_to_host(), d_J.copy_to_host(), d_G.copy_to_host(), \
           d_d.copy_to_host(), d_A_tilde_r_end.copy_to_host(), d_C_tilde_r_end.copy_to_host(), \
           d_H_tilde_r_end.copy_to_host(), d_b_tilde_r_end.copy_to_host()


'''
    Kernel for performing the partition factorization on each partition.
'''


@cuda.jit()
def partition_factorization_kernel(
        eps, d_index, size_y, size_p, M, d_A, d_C, d_H, d_b, d_q, d_q_t, d_r, d_r_j, d_C_tilde, d_E, d_G, d_J, d_d,
        d_G_tilde, d_H_tilde, d_b_tilde, d_A_tilde_r_end, d_C_tilde_r_end, d_H_tilde_r_end, d_b_tilde_r_end, d_C_A,
        d_cpy, d_v, d_vv, d_beta, d_w_t, d_u_t):
    i = cuda.grid(1)
    if i < M:
        r_start = d_index[i]
        r_end = d_index[i + 1] - 1
        # set the initial condition
        start_row_index = r_start * size_y
        end_row_index = start_row_index + size_y
        # C_tilde_{r_start} = C_{r_start}
        matrix_operation_cuda.set_equal_mat(d_C[start_row_index: end_row_index, 0: size_y],
                                            d_C_tilde[start_row_index: end_row_index, 0: size_y])
        # G_tilde_{r_start} = A_{r_start}
        matrix_operation_cuda.set_equal_mat(d_A[start_row_index: end_row_index, 0: size_y],
                                            d_G_tilde[start_row_index: end_row_index, 0: size_y])
        # H_tilde_{r_start} = H_{r_start}
        matrix_operation_cuda.set_equal_mat(d_H[start_row_index: end_row_index, 0: size_p],
                                            d_H_tilde[start_row_index: end_row_index, 0: size_p])
        # b_tilde_{r_start} = b_{r_start}
        matrix_operation_cuda.set_equal_vec(d_b[r_start, 0: size_y], d_b_tilde[r_start, 0: size_y])
        for j in range(r_start, r_end):
            # index to access matrix A and C
            # start row index for jth element
            start_row_index_cur = j * size_y
            # end row index for jth element
            end_row_index_cur = start_row_index_cur + size_y
            # start row index for (j + 1)th element
            start_row_index_next = (j + 1) * size_y
            # end row index for (j + 1)th element
            end_row_index_next = start_row_index_next + size_y
            # start row index for C_A
            start_row_index_C_A = j * 2 * size_y
            # end row index for C_A
            end_row_index_C_A = start_row_index_C_A + 2 * size_y
            matrix_operation_cuda.vstack(d_C_tilde[start_row_index_cur: end_row_index_cur, 0: size_y],
                                         d_A[start_row_index_next: end_row_index_next, 0: size_y],
                                         d_C_A[start_row_index_C_A: end_row_index_C_A, 0: size_y])
            # qr_cuda(a, cpy, q, r, v, vv, beta, w_t, u_t, eps)
            # qr decomposition of the C_A matrix
            matrix_factorization_cuda.qr(d_C_A[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_cpy[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_q[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                         d_r[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_v[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_vv[j, 0: 2 * size_y],
                                         d_beta[j, 0: size_y],
                                         d_w_t[j, 0: size_y],
                                         d_u_t[j, 0: 2 * size_y],
                                         eps)
            # obtain the transpose of the q matrix
            matrix_operation_cuda.mat_trans(d_q[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                            d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y])
            # [E_j; C_tilde_{j + 1} = Q.T * [0; C_{j + 1}]
            # block_mat_mat_mul_top_zero(d_q, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul_top_zero(
                d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                d_C[start_row_index_next: end_row_index_next, 0: size_y],
                d_E[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_C_tilde[start_row_index_next: end_row_index_next, 0: size_y])
            # [G_j; G_tilde_{j + 1} = Q.T * [G_tilde_{j}; 0]
            # block_mat_mat_mul_bot_zero(d_q, d_a, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul_bot_zero(
                d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                d_G_tilde[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_G[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_G_tilde[start_row_index_next: end_row_index_next, 0: size_y])
            # [J_j; H_tilde_{j + 1} = Q.T * [H_tilde_{j}; H_{j + 1}]
            # block_mat_mat_mul(d_q, d_a, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul(d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                                    d_H_tilde[start_row_index_cur: end_row_index_cur, 0: size_p],
                                                    d_H[start_row_index_next: end_row_index_next, 0: size_p],
                                                    d_J[start_row_index_cur: end_row_index_cur, 0: size_p],
                                                    d_H_tilde[start_row_index_next: end_row_index_next, 0: size_p])
            # [d_j; b_tilde_{j + 1} = Q.T * [b_tilde_{j}; b_{j + 1}]
            # block_mat_vec_mul(d_q, d_a, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_vec_mul(d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                                    d_b_tilde[j, 0: size_y],
                                                    d_b[j + 1, 0: size_y],
                                                    d_d[j, 0: size_y],
                                                    d_b_tilde[j + 1, 0: size_y])
            # save R_j
            matrix_operation_cuda.set_equal_mat(d_r[start_row_index_C_A: start_row_index_C_A + size_y, 0: size_y],
                                                d_r_j[start_row_index_cur: end_row_index_cur, 0: size_y])
        # start index for r_end
        start_row_index_end = r_end * size_y
        # end index for r_end
        end_row_index_end = start_row_index_end + size_y
        # set A_tilde_r_end, C_tilde_r_end, H_tilde_r_end, d_b_tilde_r_end
        matrix_operation_cuda.set_equal_mat(d_G_tilde[start_row_index_end: end_row_index_end, 0: size_y],
                                            d_A_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_y])
        matrix_operation_cuda.set_equal_mat(d_C_tilde[start_row_index_end: end_row_index_end, 0: size_y],
                                            d_C_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_y])
        matrix_operation_cuda.set_equal_mat(d_H_tilde[start_row_index_end: end_row_index_end, 0: size_p],
                                            d_H_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_p])
        matrix_operation_cuda.set_equal_vec(d_b_tilde[r_end, 0: size_y], d_b_tilde_r_end[i, 0: size_y])


# finish the implementation of partition factorization of the Jacobian


# start the implementation of solving the partitioned BABD system in sequential
# this part is copied from the sequential solver and should be updated?


'''
    Construct the reduced BABD system.
    Partition the Jabobian matrix into M partitions, qr decomposition is performed 
    in each partition and generate the final BABD system to solve.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        M: number of partitions
        A_tilde: the A_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the A_tilde matrix block 
                 at the boundary of each partition
        C_tilde: the C_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the C_tilde matrix block 
                 at the boundary of each partition
        H_tilde: the H_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_p
                 each size_y x size_p matrix block corresponds to the H_tilde matrix block 
                 at the boundary of each partition
        b_tilde: the b_tilde vector element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M x size_y
                 each size_y vector block corresponds to the b_tilde vector block at the boundary of each partition
        B_0: derivatives of boundary conditions w.r.t. ODE variables at initial time
             dimension: size_y + size_p x size_y
        B_n: derivatives of boundary conditions w.r.t. ODE variables at final time
             dimension: size_y + size_p x size_y
        V_n: derivatives of boundary conditions w.r.t. parameter varaibels
             dimension: size_y + size_p x size_p
        r_bc : boundary conditions of the system in vector form
            dimension: size_y + size_p
    Output:
        sol: self designed data structure used to solve the BABD system which contains the necessary elements
             in the reduced Jacobian matrix.
'''


def construct_babd(size_y, size_z, size_p, m, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, V_n, r_bc):
    sol = []
    for i in range(M):
        node_i = collocation_node.collocation_node(size_y, size_z, size_p, m)
        node_i.A = A_tilde[i * size_y: (i + 1) * size_y, 0: size_y]
        node_i.C = C_tilde[i * size_y: (i + 1) * size_y, 0: size_y]
        node_i.H = H_tilde[i * size_y: (i + 1) * size_y, 0: size_p]
        node_i.b = b_tilde[i, 0: size_y]
        sol.append(node_i)
    node_N = collocation_node.collocation_node(size_y, size_z, size_p, m)
    sol.append(node_N)
    sol[0].set_B(B_0)
    sol[M].set_B(B_n)
    sol[M].set_VN(V_n)
    sol[M].set_HN(sol[M].V_N)
    sol[M].set_bN(r_bc)
    return sol


'''
    Solve the reduced Jacobian system in sequential with qr decomposition.
    size_s: size of the matrix block in the Jacobian system.
    size_p: size of the parameter block in the Jacobian system.
    N: number of blocks in the Jacobian system.
    sol: self designed data structure with necessary elements of the Jacobian system
'''


def qr_decomposition(size_s, size_p, N, sol):
    sol[0].C_tilda = sol[0].C
    sol[0].G_tilda = sol[0].A
    sol[0].H_tilda = sol[0].H
    sol[0].b_tilda = sol[0].b
    for i in range(N - 2):
        C_tilda_A = np.concatenate((sol[i].C_tilda, sol[i + 1].A), axis=0)
        Q, R = np.linalg.qr(C_tilda_A, mode='complete')
        sol[i].R = R[0: size_s, :]
        zero_C = np.concatenate((np.zeros((size_s, size_s), dtype=np.float64), sol[i + 1].C), axis=0)
        EC = np.dot(Q.T, zero_C)
        sol[i].E = EC[0: size_s, 0: size_s]
        sol[i + 1].C_tilda = EC[size_s: 2 * size_s, 0: size_s]
        G_tilda_zero = np.concatenate((sol[i].G_tilda, np.zeros((size_s, size_s), dtype=np.float64)), axis=0)
        GG = np.dot(Q.T, G_tilda_zero)
        sol[i].G = GG[0: size_s, 0: size_s]
        sol[i + 1].G_tilda = GG[size_s: 2 * size_s, 0: size_s]
        H_tilda_H = np.concatenate((sol[i].H_tilda, sol[i + 1].H), axis=0)
        JH = np.dot(Q.T, H_tilda_H)
        sol[i].K = JH[0: size_s, 0: size_p]
        sol[i + 1].H_tilda = JH[size_s: 2 * size_s, 0: size_p]
        b_tilda_b = np.concatenate((sol[i].b_tilda, sol[i + 1].b), axis=0)
        db = np.dot(Q.T, b_tilda_b)
        sol[i].d = db[0: size_s]
        sol[i + 1].b_tilda = db[size_s: 2 * size_s]
    final_block_up = np.concatenate((sol[N - 2].C_tilda, sol[N - 2].G_tilda, sol[N - 2].H_tilda), axis=1)
    H_N = np.asarray(sol[N - 1].H_N)
    final_block_down = np.concatenate((sol[N - 1].B, sol[0].B, H_N), axis=1)
    final_block = np.concatenate((final_block_up, final_block_down), axis=0)
    Q, R = np.linalg.qr(final_block, mode='complete')
    sol[N - 2].R = R[0: size_s, 0: size_s]
    sol[N - 2].G = R[0: size_s, size_s: 2 * size_s]
    sol[N - 2].K = R[0: size_s, 2 * size_s: 2 * size_s + size_p]
    sol[N - 1].R = R[size_s: 2 * size_s, size_s: 2 * size_s]
    sol[N - 1].K = R[size_s: 2 * size_s, 2 * size_s: 2 * size_s + size_p]
    sol[N - 1].Rp = R[2 * size_s: 2 * size_s + size_p, 2 * size_s: 2 * size_s + size_p]

    b_N = np.asarray(sol[N - 1].b_N)
    b_tilda_b = np.concatenate((sol[N - 2].b_tilda, b_N), axis=0)
    d = np.dot(Q.T, b_tilda_b)
    sol[N - 2].d = d[0: size_s]
    sol[N - 1].d = d[size_s: 2 * size_s]
    sol[N - 1].dp = d[2 * size_s: 2 * size_s + size_p]


'''
    Perform the backward substitution to solve the upper triangular matrix system 
    to get the solution from the linear system of Newton's method.
    N: number of blocks of the system.
    sol: self designed data structure with necessary elements
'''


def backward_substitution(N, sol):
    delta_p = np.linalg.solve(sol[N - 1].Rp, sol[N - 1].dp)
    sol[N - 1].delta_p = delta_p
    delta_y1 = np.linalg.solve(sol[N - 1].R, (sol[N - 1].d - np.dot(sol[N - 1].K, delta_p)))
    sol[0].set_delta_y(delta_y1)
    b_yN = sol[N - 2].d - np.dot(sol[N - 2].G, delta_y1) - np.dot(sol[N - 2].K, delta_p)
    delta_yN = np.linalg.solve(sol[N - 2].R, b_yN)
    sol[N - 1].set_delta_y(delta_yN)
    for i in range(N - 2, 0, -1):
        b_yi = sol[i - 1].d - np.dot(sol[i - 1].E, sol[i + 1].delta_y) - np.dot(sol[i - 1].G, delta_y1) - np.dot(
            sol[i - 1].K, delta_p)
        delta_yi = np.linalg.solve(sol[i - 1].R, b_yi)
        sol[i].set_delta_y(delta_yi)


# finish the implementation of solving the partitioned BABD system in sequential


# start the implementation of the parallel backward substitution


'''
    Obtain the solution of the BABD system using partition backward substitution in parallel.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        M: number of partitions
        N: number of time nodes of the system
        index: the index of the time node in each partition
               r_start = index[i] is the start index of the partition
               r_end = index[i + 1] - 1 is the end index of the partition
               index[0] = 0 which is the first node of the mesh
               index[-1] = N - 2 which is the second last node of the mesh
        delta_s_r: solution to the reduced BABD system which is the solution at the boudary
                   of each partition of the BABD system in a row dominant matrix form
                   dimension: (M + 1) x size_y
                   delta_s_r[0] = delta_s[0] which is the solution at the start of the first partition which is also the
                   first node
                   delta_s_r[1] = delta_s[r_1 + 1] which is the solution at the start of the second partition, which is 
                   also the node after the last node of the first partition
                   delta_s_r[-1] = delta_s[r_M + 1] which is the solution at the node after the final partition, which 
                   is also the final node (index: N - 1)
        delta_p: solution of the parameter variables to the reduced BABD system in vector form
                 dimension: size_p
        R: the R matrix element from the partition factorization which contains 
           upper triangular matrix from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the R matrix block at a time node
        G: the G matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the G matrix block at a time node
        E: the E matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the E matrix block at a time node
        J: the J matrix element from the partition factorization from the qr decomposition 
                   in a row dominant matrix form
                   dimension: (N - 1) * size_y x size_p
                   each size_y x size_p matrix block corresponds to the J matrix block at a time node
        d: the d vector element from the partition factorization from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) x size_y
           each size_y vector block corresponds to the d vector block at the boundary of each partition
    Output:
        delta_s: solution to the BABD system in a row dominant matrix form.
                 dimension: N x size_y
                 each size size_y vector block corresponds to the delta_s vector block at a time node
'''


def partition_backward_substitution_parallel(size_y, size_p, M, N, index, delta_s_r, delta_p, R, G, E, J, d):
    # compute the grid dimension of the voxel model
    grid_dims = (M + TPB - 1) // TPB
    # transfer memory from CPU to GPU
    d_index = cuda.to_device(index)
    d_delta_s_r = cuda.to_device(delta_s_r)
    d_delta_p = cuda.to_device(delta_p)
    d_R = cuda.to_device(R)
    d_G = cuda.to_device(G)
    d_E = cuda.to_device(E)
    d_J = cuda.to_device(J)
    d_d = cuda.to_device(d)
    # holder for output variable
    d_delta_s = cuda.device_array((N, size_y), dtype=np.float64)
    # holder for intermediate matrix vector multiplication variables
    d_G_delta_s = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_E_delta_s = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_J_delta_p = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the right hand side of the linear system to solve in BABD system
    d_vec = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_L = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_U = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_cpy = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_c = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_y = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    partition_backward_substitution_kernel[grid_dims, TPB](eps, size_y, size_p, M, d_index, d_delta_s_r, d_delta_p, d_R,
                                                           d_G, d_E, d_J, d_d, d_G_delta_s, d_E_delta_s, d_J_delta_p,
                                                           d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y, d_delta_s)
    return d_delta_s.copy_to_host()


'''
    Kernel for computing the solution to BABD system which performs the partition backward substitution.
'''


# d_delta_s_r: (M + 1) x size_y
# d_delta_s: N x size_y
@cuda.jit()
def partition_backward_substitution_kernel(eps, size_y, size_p, M, d_index, d_delta_s_r, d_delta_p, d_R, d_G, d_E, d_J,
                                           d_d, d_G_delta_s, d_E_delta_s, d_J_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c,
                                           d_y, d_delta_s):
    i = cuda.grid(1)
    if i < M:
        r_start = d_index[i]
        r_end = d_index[i + 1] - 1
        for k in range(size_y):
            d_delta_s[r_start, k] = d_delta_s_r[i, k]
            d_delta_s[r_end + 1, k] = d_delta_s_r[i + 1, k]
        for j in range(r_end, r_start, -1):
            # set the matrix R_j as the upper triangular
            # eliminate the machine error
            for k in range(size_y):
                for l in range(k):
                    d_R[(j - 1) * size_y + k, l] = 0
            # compute G_j * delta_s[r_start, :]
            # compute E_j * delta_s[j + 1, :]
            # compute J_j * delta_p
            matrix_operation_cuda.mat_vec_mul(d_G[(j - 1) * size_y: j * size_y, 0: size_y],
                                              d_delta_s[r_start, 0: size_y],
                                              d_G_delta_s[j, 0: size_y])
            matrix_operation_cuda.mat_vec_mul(d_E[(j - 1) * size_y: j * size_y, 0: size_y], d_delta_s[j + 1, 0: size_y],
                                              d_E_delta_s[j, 0: size_y])
            matrix_operation_cuda.mat_vec_mul(d_J[(j - 1) * size_y: j * size_y, 0: size_p], d_delta_p,
                                              d_J_delta_p[j, 0: size_y])
            for k in range(size_y):
                d_vec[j, k] = d_d[j - 1, k] - d_G_delta_s[j, k] - d_E_delta_s[j, k] - d_J_delta_p[j, k]
            matrix_factorization_cuda.lu_solve_vec(d_R[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_cpy[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_P[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_L[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_U[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_vec[j, 0: size_y], d_c[j, 0: size_y], d_y[j, 0: size_y],
                                                   d_delta_s[j, 0: size_y], eps)


# finish the implementation of the parallel backward substitution


# start the implementation of the parallel recovering the delta_k


'''
    Recover the delta_k of the search direction from the reduced BABD system.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        N: number of time nodes of the system
        delta_y: solution of the search direction to the BABD system in a row dominant matrix form.
                 dimension: N x size_y
                 each size size_y vector block corresponds to the delta_s vector block at a time node
        delta_p: solution of the search direction of the parameter variables to the reduced BABD system in vector form
                 dimension: size_p
        f_a : matrix of the residual f_a for each time node in row dominant matrix form
            dimension: (N - 1) x m * (size_y + size_z), where each row corresponds the values at each time node
        J: the J matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_y
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
        V: the V matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x size_p
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
        W: the D matrix element in the Jacobian matrix in row dominant matrix form
           dimension: m * (size_y + size_z) * (N - 1) x m * (size_y + size_z)
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
    Output:
        delta_k: solution of the search direction of y_dot and z variables of the system recovered from the reduced BABD
                 system
                 dimension: (N - 1) x m * (size_y + size_z)
                 each size (size_y + size_z) vector corresponds to the search direction at each time node
        delta_y_dot: solution of the search direction of y_dot from corresponding position at delta_k
                     dimension: (N - 1) * m x size_y
                     each size size_y row vector corresponds to the search direction at the corresponding collocation 
                     point. The index for the jth collocation point from the ith time node is i * x + j
        delta_z_tilde: solution of the search direction of z_tilde from corresponding position at delta_k
                     dimension: (N - 1) * m x size_z
                     each size size_z row vector corresponds to the search direction at the corresponding collocation 
                     point. The index for the jth collocation point from the ith time node is i * x + j
'''


def recover_delta_k_parallel(size_y, size_z, m, N, delta_y, delta_p, f_a, J, V, W):
    # compute the grid dimension of the warp function
    grid_dims = ((N - 1) + TPB - 1) // TPB
    # transfer memory from CPU to GPU
    d_delta_y = cuda.to_device(delta_y)
    d_delta_p = cuda.to_device(delta_p)
    d_f_a = cuda.to_device(f_a)
    d_J = cuda.to_device(J)
    d_V = cuda.to_device(V)
    d_W = cuda.to_device(W)
    # holder for output variable
    d_delta_k = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_delta_y_dot = cuda.device_array((m * (N - 1), size_y), dtype=np.float64)
    d_delta_z_tilde = cuda.device_array((m * (N - 1), size_z), dtype=np.float64)
    # holder for intermediate matrix vector multiplication variables
    d_J_delta_y = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_V_delta_p = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    # holder for the right hand side of the linear system to solve in BABD system
    d_vec = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_L = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_U = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_cpy = cuda.device_array(((N - 1) * m * (size_y + size_z), m * (size_y + size_z)), dtype=np.float64)
    d_c = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    d_y = cuda.device_array((N - 1, m * (size_y + size_z)), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    recover_delta_k_kernel[grid_dims, TPB](eps, size_y, size_z, m, N, d_delta_y, d_delta_p, d_f_a, d_J, d_V, d_W,
                                           d_J_delta_y, d_V_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y,
                                           d_delta_k, d_delta_y_dot, d_delta_z_tilde)
    return d_delta_k.copy_to_host(), d_delta_y_dot.copy_to_host(), d_delta_z_tilde.copy_to_host()


'''
    Kernel function of recovering delta_k from the reduced BABD system.
'''


@cuda.jit()
def recover_delta_k_kernel(eps, size_y, size_z, m, N, d_delta_y, d_delta_p, d_f_a, d_J, d_V, d_W,
                           d_J_delta_y, d_V_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y,
                           d_delta_k, d_delta_y_dot, d_delta_z_tilde):
    i = cuda.grid(1)
    if i < (N - 1):
        matrix_operation_cuda.mat_vec_mul(d_J[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z), 0: size_y],
                                          d_delta_y[i, 0: size_y], d_J_delta_y[i, 0: m * (size_y + size_z)])
        matrix_operation_cuda.mat_vec_mul(d_V[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z), :], d_delta_p,
                                          d_V_delta_p[i, 0: m * (size_y + size_z)])
        for j in range(m * (size_y + size_z)):
            d_vec[i, j] = -d_f_a[i, j] - d_J_delta_y[i, j] - d_V_delta_p[i, j]
        matrix_factorization_cuda.lu_solve_vec(d_W[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z),
                                               0: m * (size_y + size_z)],
                                               d_cpy[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z),
                                               0: m * (size_y + size_z)],
                                               d_P[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z),
                                               0: m * (size_y + size_z)],
                                               d_L[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z),
                                               0: m * (size_y + size_z)],
                                               d_U[i * m * (size_y + size_z): (i + 1) * m * (size_y + size_z),
                                               0: m * (size_y + size_z)],
                                               d_vec[i, 0: m * (size_y + size_z)], d_c[i, 0: m * (size_y + size_z)],
                                               d_y[i, 0: m * (size_y + size_z)], d_delta_k[i, 0: m * (size_y + size_z)],
                                               eps)
        for j in range(m):
            start_index_y_collocation = j * (size_y + size_z)
            start_index_z_collocation = start_index_y_collocation + size_y
            for k in range(size_y):
                d_delta_y_dot[i * m + j, k] = d_delta_k[i, start_index_y_collocation + k]
            for k in range(size_z):
                d_delta_z_tilde[i * m + j, k] = d_delta_k[i, start_index_z_collocation + k]


# start the implementation of computing segment residual on each time node


'''
    Input:
        size_y: number of ODE variables.
        m: number of collocation points used
        t: time during the time span
        L: collocation weights vector
           dimension: m
        y_dot: value of the derivative of the ODE variables in time span [t_j, t_(j + 1)]
                dimension: m x size_y
    Output:
        y_dot_ret: returned value of the derivative of the ODE variables at time t
               dimension: size_y
'''


@cuda.jit(device=True)
def get_y_dot(size_y, m, t, L, y_dot, y_dot_ret):
    # ydot = sum_{k=1,m} L_k(t) * ydot_j[k]
    # zero initialization
    for i in range(size_y):
        y_dot_ret[i] = 0
    # compute the collocation weights at time t
    collocation_coefficients.compute_L(m, t, L)
    # perform collocation integration
    for i in range(m):
        for j in range(size_y):
            y_dot_ret[j] += L[i] * y_dot[i, j]
    return


'''
    Input:
        size_z: number of DAE variables.
        m: number of collocation points used
        t: time during the time span
        L: collocation weights vector
           dimension: m
        z_tilde: value of the DAE variables in time span [t_j, t_(j + 1)]
                 dimension: m x size_z
    Output:
        z: returned value of the derivative of the DAE variables at time t
           dimension: size_z
'''


@cuda.jit(device=True)
def get_z(size_z, m, t, L, z_tilde, z):
    # z = sum_{k=1,m} L_k(t) * z_j[k]
    # zero initialization
    for i in range(size_z):
        z[i] = 0
    # compute the collocation weights at time t
    collocation_coefficients.compute_L(m, t, L)
    for i in range(m):
        for j in range(size_z):
            z[j] += L[i] * z_tilde[i, j]
    return


'''
    Input:
        size_y: number of ODE variables.
        m: number of collocation points used
        t: time during the time span
        delta_t: time interval of the time span
        I: collocation weights vector
           dimension: m
        y_dot: value of the derivative of the ODE variables in time span [t_j, t_(j + 1)]
               dimension: m x size_y
    Output:
        y_ret: returned value of the ODE variables at time t
               dimension: size_y
'''


@cuda.jit(device=True)
def get_y(size_y, m, t, delta_t, I, y, y_dot, y_ret):
    # y = sy + delta*sum_{k=1,m} I_k(t) * ydot_jk
    # copy the y values
    for i in range(size_y):
        y_ret[i] = y[i]
    # compute the collocation weights at time t
    collocation_coefficients.compute_I(m, t, I)
    for i in range(m):
        for j in range(size_y):
            y_ret[j] += delta_t * I[i] * y_dot[i, j]
    return


'''
    Compute the segment residual at each time node.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        N: number of time nodes of the system
        rk: collocation coefficients which is runge-kutta coefficients usually
        t_span: time span of the problem
        y: values of the ODE variables in matrix form
           dimension: N x size_y, where each row corresponds the values at each time node
        y_dot: values of the derivatives of ODE variables in row dominant matrix form
            dimension: (N - 1) * m x size_y, where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is i * m + j.
        z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (N - 1) * m x size_z, where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
        p: values of the parameter variables in vector form
           dimension: size_p
        alpha: continuation parameter of the Newton method
        tol: numerical tolerance
'''


def compute_segment_residual_parallel(size_y, size_z, size_p, m, N, t_span, y, y_dot, z_tilde, p, alpha, tol):
    # compute the grid dimension of the warp function
    grid_dims = ((N - 1) + TPB - 1) // TPB
    # tranfer memory from CPU to GPU
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # holder for the output variables
    # remember to zero initialization in the kernel
    d_residual = cuda.device_array(N, dtype=np.float64)
    # holder for the intermediate variables
    d_h_res = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_g_res = cuda.device_array((N - 1, size_z), dtype=np.float64)
    # d_r = cuda.device_array(size_y + size_p, dtype=np.float64)
    d_L = cuda.device_array((N - 1, m), dtype=np.float64)
    d_I = cuda.device_array((N - 1, m), dtype=np.float64)
    d_y_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_z_temp = cuda.device_array((N - 1, size_z), dtype=np.float64)
    d_y_dot_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # need reduction here maybe?
    d_rho_h = cuda.device_array(N - 1, dtype=np.float64)
    d_rho_g = cuda.device_array(N - 1, dtype=np.float64)
    # gaussian coefficients
    gauss_coef = gauss_coefficients.gauss(m + 1)
    # get gaussian quardrature coefficients
    tau = gauss_coef.t
    w = gauss_coef.w
    # transfer the coefficients
    d_tau = cuda.to_device(tau)
    d_w = cuda.to_device(w)
    compute_segment_residual_kernel[grid_dims, TPB](size_y, size_z, m, N, alpha, tol, d_t_span,
                                                    d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                                    d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g, d_tau, d_w,
                                                    residual_type, scale_by_time, scale_by_initial,
                                                    d_residual)
    # copy the memory back to CPU
    rho_h = d_rho_h.copy_to_host()
    rho_g = d_rho_g.copy_to_host()
    residual = d_residual.copy_to_host()
    max_rho_r = 0
    # compute the residual at the boundary
    if (size_y + size_p) > 0:
        r = np.zeros((size_y + size_p), dtype=np.float64)
        _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r)
        max_rho_r = np.linalg.norm(r, np.inf)
        residual[N - 1] = max_rho_r / tol
    max_rho_h = np.amax(rho_h)
    max_rho_g = np.amax(rho_g)
    max_residual = np.amax(residual)
    if residual_type == 2:
        print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(sqrt(max_rho_h) / tol, sqrt(max_rho_g) / tol, max_rho_r / tol))
    else:
        print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(max_rho_h / tol, max_rho_g / tol, max_rho_r / tol))
    return residual, max_residual


'''
    Kernel function for computing segment residual.
'''


@cuda.jit()
def compute_segment_residual_kernel(size_y, size_z, m, N, alpha, tol, d_t_span,
                                    d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                    d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g, d_tau, d_w,
                                    residual_type, scale_by_time, scale_by_initial,
                                    d_residual):
    j = cuda.grid(1)
    if j < (N - 1):
        delta_t_j = d_t_span[j + 1] - d_t_span[j]
        d_rho_h[j] = 0
        d_rho_g[j] = 0
        for i in range(m + 1):
            # compute y_dot at gaussian points
            get_y_dot(size_y, m, d_tau[i], d_L[j, 0: m], d_y_dot[j * m: (j + 1) * m, 0: size_y],
                      d_y_dot_temp[j, 0: size_y])
            # compute y at gaussian points
            get_y(size_y, m, d_tau[i], delta_t_j, d_I[j, 0: m], d_y[j, 0: size_y],
                  d_y_dot[j * m: (j + 1) * m, 0: size_y], d_y_temp[j, 0: size_y])
            # compute z at gaussian points
            get_z(size_z, m, d_tau[i], d_L[j, 0: m], d_z_tilde[j * m: (j + 1) * m, 0: size_z], d_z_temp[j, 0: size_z])
            # compute h
            _abvp_f(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, d_h_res[j, 0: size_y])
            # h(y,z,p) - ydot
            for k in range(size_y):
                d_h_res[j, k] -= d_y_dot_temp[j, k]
            # d_rho_h[j] += np.dot(h_res, h_res) * w[i]
            for k in range(size_y):
                if residual_type == 2:
                    d_rho_h[j] += d_w[i] * d_h_res[j, k] * d_h_res[j, k]
                elif residual_type == 1:
                    d_rho_h[j] += d_w[i] * abs(d_h_res[j, k])
                elif residual_type == 0:
                    d_rho_h[j] = max(d_rho_h[j], d_w[i] * abs(d_h_res[j, k]))
                else:
                    print("\tNorm type invalid!")
                if scale_by_time:
                    d_rho_h[j] *= delta_t_j
                # d_rho_h[j] += delta_t_j * d_w[i] * d_h_res[j, k] * d_h_res[j, k]
            if size_z > 0:
                _abvp_g(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, alpha, d_g_res[j, 0: size_z])
                # rho_g += np.dot(g_res, g_res) * w[i]
                for k in range(size_z):
                    if residual_type == 2:
                        d_rho_g[j] += d_w[i] * d_g_res[j, k] * d_g_res[j, k]
                    elif residual_type == 1:
                        d_rho_g[j] += d_w[i] * abs(d_g_res[j, k])
                    elif residual_type == 0:
                        d_rho_g[j] = max(d_rho_g[j], d_w[i] * abs(d_h_res[j, k]))
                    else:
                        print("\tNorm type invalid!")
                    if scale_by_time:
                        d_rho_g[j] *= delta_t_j
                    # d_rho_g[j] += delta_t_j * d_w[i] * d_g_res[j, k] * d_g_res[j, k]
        if residual_type == 2:
            d_residual[j] = sqrt(d_rho_h[j] + d_rho_g[j]) / tol
        elif residual_type == 1:
            d_residual[j] = (abs(d_rho_h[j]) + abs(d_rho_g[j])) / tol
        elif residual_type == 0:
            d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
    return


def compute_segment_residual_parallel2(size_y, size_z, size_p, m, N, t_span, y, y_dot, z_tilde, p, alpha, tol):
    # compute the grid dimension of the warp function
    grid_dims = ((N - 1) + TPB - 1) // TPB
    # tranfer memory from CPU to GPU
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # holder for the output variables
    # remember to zero initialization in the kernel
    d_residual = cuda.device_array(N, dtype=np.float64)
    # holder for the intermediate variables
    d_h_res = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_g_res = cuda.device_array((N - 1, size_z), dtype=np.float64)
    # d_r = cuda.device_array(size_y + size_p, dtype=np.float64)
    d_L = cuda.device_array((N - 1, m), dtype=np.float64)
    d_I = cuda.device_array((N - 1, m), dtype=np.float64)
    d_y_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_z_temp = cuda.device_array((N - 1, size_z), dtype=np.float64)
    d_y_dot_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # intermediate variable for scaling
    d_y_dot_gauss = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_z_gauss = cuda.device_array((N - 1, size_z), dtype=np.float64)
    # need reduction here maybe?
    d_rho_h = cuda.device_array(N - 1, dtype=np.float64)
    d_rho_g = cuda.device_array(N - 1, dtype=np.float64)
    # gaussian coefficients
    gauss_coef = gauss_coefficients.gauss(m + 1)
    # get gaussian quardrature coefficients
    tau = gauss_coef.t
    w = gauss_coef.w
    # transfer the coefficients
    d_tau = cuda.to_device(tau)
    d_w = cuda.to_device(w)
    compute_segment_residual_kernel2[grid_dims, TPB](size_y, size_z, m, N, alpha, tol, d_t_span,
                                                    d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                                    d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g, d_tau, d_w,
                                                    d_y_dot_gauss, d_z_gauss,
                                                    d_residual)
    # copy the memory back to CPU
    rho_h = d_rho_h.copy_to_host()
    rho_g = d_rho_g.copy_to_host()
    residual = d_residual.copy_to_host()
    max_rho_r = 0
    # compute the residual at the boundary
    if (size_y + size_p) > 0:
        r = np.zeros((size_y + size_p), dtype=np.float64)
        _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r)
        max_rho_r = np.linalg.norm(r, np.inf) / tol
        residual[N - 1] = max_rho_r
    max_rho_h = np.amax(rho_h) / tol
    max_rho_g = np.amax(rho_g) / tol
    max_residual = np.amax(residual)
    print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(max_rho_h, max_rho_g, max_rho_r))
    return residual, max_residual


'''
    Kernel function for computing segment residual.
'''


@cuda.jit()
def compute_segment_residual_kernel2(size_y, size_z, m, N, alpha, tol, d_t_span,
                                    d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                    d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g, d_tau, d_w,
                                    d_y_dot_gauss, d_z_gauss,
                                    d_residual):
    j = cuda.grid(1)
    if j < (N - 1):
        delta_t_j = d_t_span[j + 1] - d_t_span[j]
        d_rho_h[j] = 0
        d_rho_g[j] = 0
        for k in range(size_y):
            d_y_dot_gauss[j, k] = 0
        for k in range(size_z):
            d_z_gauss[j, k] = 0
        # compute variables at gaussian points and record the maximum for each variable among
        # all the gaussian points
        for i in range(m + 1):
            # compute y_dot at gaussian points
            get_y_dot(size_y, m, d_tau[i], d_L[j, 0: m], d_y_dot[j * m: (j + 1) * m, 0: size_y],
                      d_y_dot_temp[j, 0: size_y])
            for k in range(size_y):
                d_y_dot_gauss[j, k] = max(d_y_dot_gauss[j, k], abs(d_y_dot_temp[j, k]))
            # compute z at gaussian points
            get_z(size_z, m, d_tau[i], d_L[j, 0: m], d_z_tilde[j * m: (j + 1) * m, 0: size_z], d_z_temp[j, 0: size_z])
            for k in range(size_z):
                d_z_gauss[j, k] = max(d_z_gauss[j, k], abs(d_z_temp[j, k]))
        cuda.syncthreads()
        # compute the residual at gaussian points
        for i in range(m + 1):
            # compute y_dot at gaussian points
            get_y_dot(size_y, m, d_tau[i], d_L[j, 0: m], d_y_dot[j * m: (j + 1) * m, 0: size_y],
                      d_y_dot_temp[j, 0: size_y])
            # compute y at gaussian points
            get_y(size_y, m, d_tau[i], delta_t_j, d_I[j, 0: m], d_y[j, 0: size_y],
                  d_y_dot[j * m: (j + 1) * m, 0: size_y], d_y_temp[j, 0: size_y])
            # compute z at gaussian points
            get_z(size_z, m, d_tau[i], d_L[j, 0: m], d_z_tilde[j * m: (j + 1) * m, 0: size_z], d_z_temp[j, 0: size_z])
            # compute h
            _abvp_f(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, d_h_res[j, 0: size_y])
            # h(y,z,p) - ydot
            for k in range(size_y):
                # E[j, k] = y_dot_tilde[j, k] - y_dot[j, k]
                d_h_res[j, k] -= d_y_dot_temp[j, k]
                # e^{h, j}_{i, k} = abs(E[j, k]) / (1 + max_{i \in 1,...m} y_dot[i, k])
                # e^h_j = max e^j_{i, k}
                d_rho_h[j] = max(d_rho_h[j], abs(d_h_res[j, k]) / (1 + d_y_dot_gauss[j, k]))
            if size_z > 0:
                _abvp_g(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, alpha, d_g_res[j, 0: size_z])
                for k in range(size_z):
                    # e^{g, j}_{i, k} = abs(E[j, k]) / (1 + max_{i \in 1,...m} z[i, k])
                    # e^g_j = max e^j_{i, k}
                    d_rho_g[j] = max(d_rho_g[j], abs(d_g_res[j, k]) / (1 + d_z_gauss[j, k]))
        # d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
        d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
    return


# finish the implementation of computing segment residual on each time node


# start the implementation of recovering solution


'''
    Recover the solution to the BVP-DAE problem.
    Input:
        size_z: number of DAE variables of the BVP-DAE problem.
        m: number of collocation points used.
        N: number of time nodes in the system.
        z_tilde: values of the DAE variables at the collocation points in a row dominant matrix form.
    Output:
    z: values of the DAE variables only at time nodes.
'''


def recover_solution_parallel(size_z, m, N, z_tilde):
    # compute the grid dimension of the warp function
    grid_dims = (N + TPB - 1) // TPB
    # tranfer memory from CPU to GPU
    d_z_tilde = cuda.to_device(z_tilde)
    # holder for output variables
    d_z = cuda.device_array((N, size_z), dtype=np.float64)
    # holder for intermediate variables
    d_L = cuda.device_array((N, m), dtype=np.float64)
    # execute the kernel function
    recover_solution_kernel[grid_dims, TPB](size_z, m, N, d_z_tilde, d_L, d_z)
    # return the ouput
    return d_z.copy_to_host()


@cuda.jit()
def recover_solution_kernel(size_z, m, N, d_z_tilde, d_L, d_z):
    i = cuda.grid(1)
    if i < (N - 1):
        t = 0
        collocation_coefficients.compute_L(m, t, d_L[i, 0: m])
        # zero initialization
        for k in range(size_z):
            d_z[i, k] = 0
        # loop through all the collocation points
        for j in range(m):
            # loop through all the Z variables
            for k in range(size_z):
                d_z[i, k] += d_L[i, j] * d_z_tilde[i * m + j, k]
    # for the last time node
    if i == (N - 1):
        t = 1
        collocation_coefficients.compute_L(m, t, d_L[i, 0: m])
        # zero initialization
        for k in range(size_z):
            d_z[i, k] = 0
        # loop through all the collocation points
        for j in range(m):
            # loop through all the Z variables
            for k in range(size_z):
                d_z[i, k] += d_L[i, j] * d_z_tilde[(i - 1) * m + j, k]
    return


# finish the implementation of recovering solution


# start the implementation of remesh


'''
 Remesh the problem
 def [N_New, tspan_New, y0_New, z0_New] = remesh(size_y, size_z, N, tspan, y0, z0, residual)
 Input:
        size_y : number of y variables.
        size_z : number of z variables.
        N : number of time nodes.
        m : number of collocation points used.
        tspan : distribution of the time nodes, 1-D array.
        y0 : 2-D matrix of values of the y variables, with each ith row vector corresponding to
             the y values at the ith time node.
        z0 : 2 -D matrix of values of the z variables, with each ith row vector corresponding to
             the y values at the ith time node.
        residual : residual errors at each time node, 1-D array.
 Output:
        N_New : number of time nodes of the new mesh .
        tspan_New : new distribution of the time nodes, 1-D array.
        y0_New : 2-D matrix of the values of the y variables of the new mesh.
        z0_New : 2-D matrix of the values of the z variables of the new mesh.
'''


def remesh(size_y, size_z, N, tspan, y0, z0, residual):
    N_Temp = 0
    tspan_Temp = []
    y0_Temp = []
    z0_Temp = []
    residual_Temp = []

    # Deleting Nodes
    i = 0
    # Record the number of the deleted nodes
    k_D = 0

    thresholdDel = 1e-2
    while i < N - 4:
        res_i = residual[i]
        if res_i <= thresholdDel:
            res_i_Plus1 = residual[i + 1]
            res_i_Plus2 = residual[i + 2]
            res_i_Plus3 = residual[i + 3]
            res_i_Plus4 = residual[i + 4]
            if res_i_Plus1 <= thresholdDel and res_i_Plus2 <= thresholdDel and res_i_Plus3 <= thresholdDel and \
                    res_i_Plus4 <= thresholdDel:
                # append the 1st, 3rd, and 5th node
                # 1st node
                tspan_Temp.append(tspan[i])
                y0_Temp.append(y0[i, :])
                z0_Temp.append(z0[i, :])
                residual_Temp.append(residual[i])
                # 3rd node
                tspan_Temp.append(tspan[i + 2])
                y0_Temp.append(y0[i + 2, :])
                z0_Temp.append(z0[i + 2, :])
                residual_Temp.append(residual[i + 2])
                # 5th node
                tspan_Temp.append(tspan[i + 4])
                y0_Temp.append(y0[i + 4, :])
                z0_Temp.append(z0[i + 4, :])
                residual_Temp.append(residual[i + 4])
                # delete 2 nodes
                k_D += 2
                # add 3 nodes to the total number
                N_Temp += 3
                # ignore those five nodes
                i += 5
            else:
                # directly add the node
                tspan_Temp.append(tspan[i])
                y0_Temp.append(y0[i, :])
                z0_Temp.append(z0[i, :])
                residual_Temp.append(residual[i])
                N_Temp += 1
                i += 1
        else:
            # directly add the node
            tspan_Temp.append(tspan[i])
            y0_Temp.append(y0[i, :])
            z0_Temp.append(z0[i, :])
            residual_Temp.append(residual[i])
            N_Temp += 1
            i += 1
    '''
        if the previous loop stop at the ith node which is bigger than (N - 4), those last
        few nodes left are added manually, if the last few nodes have already been processed,
        the index i should be equal to N, then nothing needs to be done
    '''
    if i < N:
        '''
            add the last few nodes starting from i to N - 1, which
            is a total of (N - i) nodes
        '''
        for j in range(N - i):
            # append the N - 4 + j node
            tspan_Temp.append(tspan[i + j])
            y0_Temp.append(y0[i + j, :])
            z0_Temp.append(z0[i + j, :])
            residual_Temp.append(residual[i + j])
            N_Temp += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_Temp = np.array(tspan_Temp)
    y0_Temp = np.array(y0_Temp)
    z0_Temp = np.array(z0_Temp)
    residual_Temp = np.array(residual_Temp)
    # lists to hold the outputs
    N_New = 0
    tspan_New = []
    y0_New = []
    z0_New = []
    residual_New = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    k_A = 0

    while i < N_Temp - 1:
        res_i = residual_Temp[i]
        if res_i > 1:
            if res_i > 10:
                # add three uniformly spaced nodes
                # add the time point of new nodes
                delta_t = (tspan_Temp[i + 1] - tspan_Temp[i]) / 4
                t_i = tspan_Temp[i]
                t_i_Plus1 = t_i + delta_t
                t_i_Plus2 = t_i + 2 * delta_t
                t_i_Plus3 = t_i + 3 * delta_t
                tspan_New.append(t_i)
                tspan_New.append(t_i_Plus1)
                tspan_New.append(t_i_Plus2)
                tspan_New.append(t_i_Plus3)
                # add the residuals of the new nodes
                delta_res = (residual_Temp[i + 1] - residual_Temp[i]) / 4
                res_i_Plus1 = res_i + delta_res
                res_i_Plus2 = res_i + 2 * delta_res
                res_i_Plus3 = res_i + 3 * delta_res
                residual_New.append(res_i)
                residual_New.append(res_i_Plus1)
                residual_New.append(res_i_Plus2)
                residual_New.append(res_i_Plus3)
                # add the ys of the new nodes
                y0_i = y0_Temp[i, :]
                y0_i_Next = y0_Temp[i + 1, :]
                delta_y0 = (y0_i_Next - y0_i) / 4
                y0_i_Plus1 = y0_i + delta_y0
                y0_i_Plus2 = y0_i + 2 * delta_y0
                y0_i_Plus3 = y0_i + 3 * delta_y0
                y0_New.append(y0_i)
                y0_New.append(y0_i_Plus1)
                y0_New.append(y0_i_Plus2)
                y0_New.append(y0_i_Plus3)
                # add the zs of the new nodes
                z0_i = z0_Temp[i, :]
                z0_i_Next = z0_Temp[i + 1, :]
                delta_z0 = (z0_i_Next - z0_i) / 4
                z0_i_Plus1 = z0_i + delta_z0
                z0_i_Plus2 = z0_i + 2 * delta_z0
                z0_i_Plus3 = z0_i + 3 * delta_z0
                z0_New.append(z0_i)
                z0_New.append(z0_i_Plus1)
                z0_New.append(z0_i_Plus2)
                z0_New.append(z0_i_Plus3)
                # update the index
                # 1 original node + 3 newly added nodes
                N_New += 4
                k_A += 3
                i += 1
            else:
                # add one node to the middle
                # add the time point of the new node
                delta_t = (tspan_Temp[i + 1] - tspan_Temp[i]) / 2
                t_i = tspan_Temp[i]
                t_i_Plus1 = t_i + delta_t
                tspan_New.append(t_i)
                tspan_New.append(t_i_Plus1)
                # add the residual of the new node
                delta_res = (residual_Temp[i + 1] - residual_Temp[i]) / 2
                res_i_Plus1 = res_i + delta_res
                residual_New.append(res_i)
                residual_New.append(res_i_Plus1)
                # add the y of the new node
                y0_i = y0_Temp[i, :]
                y0_i_Next = y0_Temp[i + 1, :]
                delta_y0 = (y0_i_Next - y0_i) / 2
                y0_i_Plus1 = y0_i + delta_y0
                y0_New.append(y0_i)
                y0_New.append(y0_i_Plus1)
                # add the z of the new node
                z0_i = z0_Temp[i, :]
                z0_i_Next = z0_Temp[i + 1, :]
                delta_z0 = (z0_i_Next - z0_i) / 2
                z0_i_Plus1 = z0_i + delta_z0
                z0_New.append(z0_i)
                z0_New.append(z0_i_Plus1)
                # update the index
                # 1 original node + 1 newly added node
                N_New += 2
                k_A += 1
                i += 1
        else:
            # add the current node only
            # add the time node of the current node
            t_i = tspan_Temp[i]
            tspan_New.append(t_i)
            # add the residual of the current node
            residual_New.append(res_i)
            # add the y of the current node
            y0_i = y0_Temp[i, :]
            y0_New.append(y0_i)
            # add the z of the current node
            z0_i = z0_Temp[i, :]
            z0_New.append(z0_i)
            # update the index
            # 1 original node only
            N_New += 1
            i += 1
    # add the final node
    tspan_New.append(tspan_Temp[N_Temp - 1])
    y0_New.append(y0_Temp[N_Temp - 1, :])
    z0_New.append(z0_Temp[N_Temp - 1, :])
    residual_New.append(residual_Temp[N_Temp - 1])
    N_New += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_New = np.array(tspan_New)
    y0_New = np.array(y0_New)
    z0_New = np.array(z0_New)
    print("\tDelete nodes: {}; Add nodes: {}; Number of nodes after mesh: {}".format(k_D, k_A, N_New))
    # return the output
    return N_New, tspan_New, y0_New, z0_New


# finish the implementation of remesh


# start the implementation of plot


def plot_result(size_y, size_z, t_span, y, z):
    for i in range(size_y):
        fig, ax = plt.subplots()
        ax.plot(t_span, y[:, i])
        ax.set(xlabel='time', ylabel='ODE variable %s' % (i + 1),
               title='{}_{}'.format('ODE variable', (i + 1)))
        ax.grid()
        plt.show()
    for i in range(size_z):
        fig, ax = plt.subplots()
        ax.plot(t_span, z[:, i])
        ax.set(xlabel='time', ylabel='DAE variable %s' % (i + 1),
               title='{}_{}'.format('DAE variable', (i + 1)))
        ax.grid()
        plt.show()


# finish the implementation of plot


def write_benchmark_result(fname,
                           initial_input_time, initial_input_count,
                           residual_time, residual_count,
                           jacobian_time, jacobian_count,
                           reduce_jacobian_time, reduce_jacobian_count,
                           recover_babd_time, recover_babd_count,
                           segment_residual_time, segment_residual_count,
                           total_time):
    try:
        with open(fname, 'w') as f:
            f.write('Initial input time: {}\n'.format(initial_input_time))
            f.write('Initial input counts: {}\n'.format(initial_input_count))
            f.write('Residual time: {}\n'.format(residual_time))
            f.write('Residual counts: {}\n'.format(residual_count))
            f.write('Jacobian time: {}\n'.format(jacobian_time))
            f.write('Jacobian counts: {}\n'.format(jacobian_count))
            f.write('Reduce Jacobian time: {}\n'.format(reduce_jacobian_time))
            f.write('Reduce Jacobian counts: {}\n'.format(reduce_jacobian_count))
            f.write('Recover BABD time: {}\n'.format(recover_babd_time))
            f.write('Recover BABD counts: {}\n'.format(recover_babd_count))
            f.write('Segment Residual time: {}\n'.format(segment_residual_time))
            f.write('Segment Residual counts: {}\n'.format(segment_residual_count))
            f.write('Total time: {}\n'.format(total_time))
    except OSError:
        print('Write time failed!')


def mesh_sanity_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
    if mesh_it > max_mesh:
        print("\tReach maximum number of mesh refinements allowed.")
        return True
    if N > max_nodes:
        print('\tReach maximum number of nodes allowed.')
        return True
    elif N < min_nodes:
        print('\tReach minimum number of nodes allowed.')
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use the CUDA based collocation solver to solve the "
                                                 "optimal control problem.")
    parser.add_argument("-m", type=int, choices=[3, 4, 5, 6, 7, 8, 9, 10], default=4,
                        help="number of collocation points used to solve the problem")
    args = parser.parse_args()
    m = args.m

    print("Running '{}'".format(__file__))
    print("Solve the optimal control problem with {} collocation points.".format(m))
    collocation_solver_parallel(m=m)
