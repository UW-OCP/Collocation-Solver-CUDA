import numpy as np
import scipy.linalg


class collocation_node:
    """
        Node class for collocation solver, save the required DAE variables
        at each time node
    """

    '''
        Input: size_y - size of the ODE variables
               size_z - size of the DAE variables
               size_p - size of the parameters
               m: stages of the lobatto coefficients 
        Construct the empty collocation node with all the 
        fields set to zero.
    '''
    def __init__(self, size_y, size_z, size_p, m):
        self.size_y = size_y
        self.size_z = size_z
        self.size_p = size_p
        self.m = m
        self.delta_t = 0
        self.tspan = []
        self.y = np.zeros((size_y), dtype = np.float64)
        self.z = np.zeros((size_z), dtype = np.float64)
        self.y_dot = np.zeros((size_y, m), dtype = np.float64)
        self.z_tilda = np.zeros((size_z, m), dtype = np.float64)
        self.p = np.zeros((size_p), dtype = np.float64)
        self.y_tilda = np.zeros((size_y, m), dtype = np.float64)
        self.f_a = np.zeros(((size_y + size_z) * m), dtype = np.float64)
        self.f_b = np.zeros((size_y), dtype = np.float64)
        self.f_N = []
        self.J = np.zeros(((size_y + size_z) * m, size_y), dtype = np.float64)
        self.W = np.zeros(((size_y + size_z) * m, (size_y + size_z) * m), dtype = np.float64)
        self.V = np.zeros(((size_y + size_z) * m, size_p), dtype = np.float64)
        self.V_N = []
        self.D = np.zeros((size_y, (size_y + size_z) * m), dtype = np.float64)
        self.B = np.zeros((size_y + size_p, size_y), dtype = np.float64)
        self.A = np.zeros((size_y, size_y), dtype = np.float64)
        self.C = np.zeros((size_y, size_y), dtype = np.float64)
        self.H = np.zeros((size_y, size_p), dtype=np.float64)
        self.H_N = []
        self.b = np.zeros((size_y), dtype=np.float64)
        self.b_N = []
        self.delta_y = np.zeros(size_y, dtype=np.float64)
        self.delta_k = np.zeros(((size_y + size_z) * m), dtype = np.float64)
        self.delta_p = np.zeros(size_p, dtype=np.float64)
        '''
        self.C_tilda = np.zeros((size_y, size_y), dtype = np.float64)
        self.G_tilda = np.zeros((size_y, size_y), dtype = np.float64)
        self.H_tilda = np.zeros((size_y, size_p), dtype = np.float64)
        self.b_tilda = np.zeros((size_y), dtype = np.float64)
        self.R = np.zeros()
        self.E = np.zeros()
        self.G = np.zeros()
        self.K = np.zeros()
        self.d = np.zeros()
        self.Rp = np.zeros()
        self.dp = np.zeros()
        self.delta_p = np.zeros()
        self.delta_y = np.zeros()
        self.delta_k = np.zeros()
        '''

    '''
        Input: y - a size_y vector of the value of the ODE variables
        Set the ODE variable y.
    '''
    def set_y(self, y):
        for i in range(self.size_y):
            self.y[i] = y[i]

    '''
        Input: z - a size_z vector of the value of the DAE variables
        Set the DAE variable z.
    '''
    def set_z(self, z):
        for i in range(self.size_z):
            self.z[i] = z[i]

    '''
        Input: p - a size_p x 1 vector of the value of the parameter variables
        Set the parameter variable p.
    '''
    def set_p(self, p):
        for i in range(self.size_p):
            self.p[i] = p[i]

    '''
        Input: delta_t - a double representing the interval of the time 
                         span between the current node and the next node
        Set the time interval delta_t.
    '''
    def set_delta_t(self, delta_t):
        self.delta_t = delta_t

    '''
        Input: tspan representing the time span between the current node and the next node
        Set the time interval tspan.
    '''
    def set_tspan(self, tspan):
        for i in range(tspan.shape[0]):
            self.tspan.append(tspan[i])

    '''
        Input:
                y_dot : value of the derivative of the ODE variable y
                j : index of the collocation point
        Set the y_dot at the jth collocation point
    '''
    def set_y_dot(self, y_dot, j):
        for i in range(self.size_y):
            self.y_dot[i][j] = y_dot[i]

    '''
        Input:
                z_tilda : value of the DAE variable z
                j : index of the collocation point
        Set the z_tilda at the jth collocation point
    '''
    def set_z_tilda(self, z_tilda, j):
        for i in range(self.size_z):
            self.z_tilda[i][j] = z_tilda[i]

    '''
        Input:
                y_tilda : value of the ODE variable y
                j : index of the collocation point
        Set the y_tilda at the jth collocation point
    '''
    def set_y_tilda(self, y_tilda, j):
        for i in range(self.size_y):
            self.y_tilda[i][j] = y_tilda[i]

    '''
        Input:
                f_a : value of the residual of all the collocation points of the time interval
        Set the residual f_a of the time interval
    '''
    def set_f_a(self, f_a):
        for i in range((self.size_y + self.size_z) * self.m):
            self.f_a[i] = f_a[i]

    '''
        Input:
                f_b : value of the residual of at the node
        Set the residual f_b of the node
    '''
    def set_f_b(self, f_b):
        for i in range(self.size_y):
            self.f_b[i] = f_b[i]

    # self.f_N = np.zeros((size_y + size_p), dtype = np.float64)
    def set_f_N(self, f_N):
        for i in range(self.size_y + self.size_p):
            self.f_N.append(f_N[i])

    def get_y(self):
        y = np.zeros((self.size_y), dtype = np.float64)
        for i in range(self.size_y):
            y[i] = self.y[i]
        return y

    def get_y_tilda(self, j):
        y = np.zeros((self.size_y), dtype = np.float64)
        for i in range(self.size_y):
            y[i] = self.y_tilda[i][j]
        return y

    def get_z_tilda(self, j):
        z = np.zeros((self.size_z), dtype = np.float64)
        for i in range(self.size_z):
            z[i] = self.z_tilda[i][j]
        return z

    def set_B(self, B):
        for i in range(self.size_y + self.size_p):
            for j in range(self.size_y):
                self.B[i][j] = B[i][j]

    # self.VN = np.zeros((size_y + size_p, size_p), dtype = np.float64)
    def set_VN(self, VN):
        for i in range(self.size_y + self.size_p):
            V_row = []
            for j in range(self.size_p):
                V_row.append(VN[i][j])
            self.V_N.append(V_row)

    # self.HN = np.zeros((size_y + size_p, size_p), dtype = np.float64)
    def set_HN(self, VN):
        for i in range(self.size_y + self.size_p):
            H_row = []
            for j in range(self.size_p):
                H_row.append(VN[i][j])
            self.H_N.append(H_row)

    # self.b_N = np.zeros((size_y + size_p), dtype = np.float64)
    def set_bN(self, f_b):
        for i in range(self.size_y + self.size_p):
            self.b_N.append(-f_b[i])

    def set_delta_y(self, delta_y):
        for i in range(self.size_y):
            self.delta_y[i] = delta_y[i]

    def set_delta_k(self, delta_k):
        for i in range((self.size_y + self.size_z) * self.m):
            self.delta_k[i] = delta_k[i]

    # j_col : jth collocation node
    def set_Jacobian(self, a, b, Dh, Dg, j_col):
        '''
        hy = np.zeros((self.size_y, self.size_y), dtype = np.float64)
        hz = np.zeros((self.size_y, self.size_z), dtype = np.float64)
        hp = np.zeros((self.size_y, self.size_p), dtype = np.float64)
        gy = np.zeros((self.size_z, self.size_y), dtype = np.float64)
        gz = np.zeros((self.size_z, self.size_z), dtype = np.float64)
        gp = np.zeros((self.size_z, self.size_p), dtype = np.float64)
        '''
        hy = Dh[0 : , 0 : self.size_y]
        hz = Dh[0 : , self.size_y : self.size_y + self.size_z]
        hp = Dh[0 : , self.size_y + self.size_z : ]
        gy = Dg[0 : , 0 : self.size_y]
        gz = Dg[0 : , self.size_y : self.size_y + self.size_z]
        gp = Dg[0 : , self.size_y + self.size_z : ]
        '''
        for i in range(self.size_y):
            for j in range(self.size_y):
                hy[i][j] = Dh[i][j]
            for j in range(self.size_z):
                hz[i][j] = Dh[i][j + self.size_y]
            for j in range(self.size_p):
                hp[i][j] = Dh[i][j + (self.size_y + self.size_z)]
        for i in range(self.size_z):
            for j in range(self.size_y):
                gy[i][j] = Dg[i][j]
            for j in range(self.size_z):
                gz[i][j] = Dg[i][j + self.size_y]
            for j in range(self.size_p):
                gp[i][j] = Dg[i][j + (self.size_y + self.size_z)]
        '''
        start_row_index_h = j_col * (self.size_y + self.size_z)
        self.J[start_row_index_h : start_row_index_h + self.size_y, 0 : self.size_y] = hy
        self.V[start_row_index_h : start_row_index_h + self.size_y, 0 : self.size_p] = hp
        '''
        for i in range(self.size_y):
            for j in range(self.size_y):
                self.J[start_row_index_h + i][j] = hy[i][j]
            for j in range(self.size_p):
                self.V[start_row_index_h + i][j] = hp[i][j]
        '''
        start_row_index_g = start_row_index_h + self.size_y
        self.J[start_row_index_g : start_row_index_g + self.size_z, 0 : self.size_y] = gy
        self.V[start_row_index_g : start_row_index_g + self.size_z, 0 : self.size_p] = gp
        '''
        for i in range(self.size_z):
            for j in range(self.size_y):
                self.J[start_row_index_g + i][j] = gy[i][j]
            for j in range(self.size_p):
                self.V[start_row_index_g + i][j] = gp[i][j]
        '''
        self.D[0 : self.size_y, j_col * (self.size_y + self.size_z) : j_col * (self.size_y + self.size_z) + self.size_y] =  self.delta_t * b * np.eye(self.size_y, dtype = np.float64)
        '''
        for i in range(self.size_y):
            self.D[i][i + j_col * (self.size_y + self.size_z)] = self.delta_t * b * 1
        '''
          
        # for each row block j_col
        # loop through all the column block  
        for i in range(self.m):
            start_row_index = j_col * (self.size_y + self.size_z)
            w_tmp = np.zeros(((self.size_y + self.size_z), (self.size_y + self.size_z)), dtype = np.float64)
            if i == j_col:
                start_col_index = i * (self.size_y + self.size_z)
                identity = np.eye(self.size_y, dtype = np.float64)
                w_tmp[0 : self.size_y, 0 : self.size_y] = -identity + self.delta_t * a[j_col, j_col] * hy
                w_tmp[0 : self.size_y, self.size_y : ] = hz
                w_tmp[self.size_y : , 0 : self.size_y] = self.delta_t * a[j_col, j_col] * gy
                w_tmp[self.size_y : , self.size_y : ] = gz
                self.W[start_row_index : start_row_index + (self.size_y + self.size_z), start_col_index : start_col_index + (self.size_y + self.size_z)] = w_tmp
                '''
                for j in range(self.size_y):
                    for k in range(self.size_y):
                        w_tmp[j][k] = -identity[j][k] + self.delta_t * a[j_col][j_col] * hy[j][k]
                    for k in range(self.size_z):
                        w_tmp[j][k + self.size_y] = hz[j][k]
                for j in range(self.size_z):
                    for k in range(self.size_y):
                        w_tmp[j + self.size_y][k] = self.delta_t * a[j_col][j_col] * gy[j][k]
                    for k in range(self.size_z):
                        w_tmp[j + self.size_y][k + self.size_y] = gz[j][k]
                for j in range(self.size_y + self.size_z):
                    for k in range(self.size_y + self.size_z):
                        self.W[start_row_index + j][start_col_index + k] = w_tmp[j][k]
                '''
            else:
                start_col_index = i * (self.size_y + self.size_z)
                w_tmp[0 : self.size_y, 0 : self.size_y] = self.delta_t * a[j_col, i] * hy
                w_tmp[0 : self.size_y, self.size_y : ] = np.zeros((self.size_y, self.size_z), dtype = np.float64)
                w_tmp[self.size_y : , 0 : self.size_y] = self.delta_t * a[j_col, i] * gy
                w_tmp[self.size_y : , self.size_y : ] = np.zeros((self.size_z, self.size_z), dtype = np.float64)
                self.W[start_row_index : start_row_index + (self.size_y + self.size_z), start_col_index : start_col_index + (self.size_y + self.size_z)] = w_tmp
                '''
                for j in range(self.size_y):
                    for k in range(self.size_y):
                        w_tmp[j][k] = self.delta_t * a[j_col][i] * hy[j][k]
                    for k in range(self.size_z):
                        w_tmp[j][k + self.size_y] = 0
                for j in range(self.size_z):
                    for k in range(self.size_y):
                        w_tmp[j + self.size_y][k] = self.delta_t * a[j_col][i] * gy[j][k]
                    for k in range(self.size_z):
                        w_tmp[j + self.size_y][k + self.size_y] = 0
                for j in range(self.size_y + self.size_z):
                    for k in range(self.size_y + self.size_z):
                        self.W[start_row_index + j][start_col_index + k] = w_tmp[j][k]
                '''

    def update_Jacobian(self):
        # W_inv = np.linalg.inv(self.W)
        # identity = np.eye(self.size_y, dtype=np.float64)
        # D_W_inv = np.dot(self.D, W_inv)
        # self.A = -identity + np.dot(D_W_inv, self.J)
        # self.C = np.eye(self.size_y)
        # self.H = np.dot(D_W_inv, self.V)
        # self.b = -self.f_b - np.dot(D_W_inv, self.f_a)
        identity = np.eye(self.size_y, dtype=np.float64)
        # P * W = L * U
        P, L, U = scipy.linalg.lu(self.W)
        # A = -I + D * W^(-1) * J
        # X = np.dot(P, self.J)
        X = np.linalg.solve(P, self.J)
        L_inv_J = np.linalg.solve(L, X)
        W_inv_J = np.linalg.solve(U, L_inv_J)
        D_W_inv_J = np.dot(self.D, W_inv_J)
        self.A = -identity + D_W_inv_J
        # H = D * W^(-1) * V
        X = np.linalg.solve(P, self.V)
        L_inv_V = np.linalg.solve(L, X)
        W_inv_V = np.linalg.solve(U, L_inv_V)
        self.H = np.dot(self.D, W_inv_V)
        # C = I
        self.C = identity
        # b = -f_b - D * W^(-1) * f_a
        X = np.linalg.solve(P, self.f_a)
        L_inv_f_a = np.linalg.solve(L, X)
        W_inv_f_a = np.linalg.solve(U, L_inv_f_a)
        D_W_inv_f_a = np.dot(self.D, W_inv_f_a)
        self.b = -self.f_b - D_W_inv_f_a
