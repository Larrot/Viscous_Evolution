import numpy as np
import scipy

def General_Pringle(D, u_init, u_left, u_right, t_end, R_in, R_out, tau, N, epsilon, M):
    """
    Implicit scheme for numerical solution of the Dirichle 
    problem for one-dimensional parabolic equation.


    Parameters
    ----------
    D : function
        diffusion coefficient D=D(u, r, t).
    u_init : function
        initial condition u=u(0,x).
    u_left : function
        Left boundary condition u=u(t,0).
    u_right : function
        Right boundary condition u=u(t,L).
    t_end : float 
        End time of computations.
    R_in : float 
        Inner boundary
    R_out : float 
        Outside boundary.
    tau : float 
        the length of time step.
    N : int 
        the number of nodes.
    epsilon : float 
        Accuracy of iterative process.    

    Returns function u(x, t_end)
    -------
    """
    def Tridiagonal_solver(a, b, c, d):
        """
         Algorithm to solve tridiagonal systems of equations

         Parameters
         ----------
         a : array
             Lower diagonal.
         b : array
             Middle diagonal.
         c : array
             Upper diagonal.
         d : array
             Right side of equations.

         Returns solution array x
         -------
        """
        size = len(d)
        x = np.zeros(size)
        # Forward steps
        for i in range(size-1):
            w = a[i]/b[i]
            b[i+1] -= w*c[i]
            d[i+1] -= w*d[i]
        # Backward steps
        x[size-1] = d[size-1]/b[size-1]
        for i in range(size-2, -1, -1):
            x[i] = (d[i]-c[i]*x[i+1])/b[i]

        return x

    # Spatial grid
    # x = np.logspace(R_in, R_out, N+1)
    x = np.logspace(R_in, R_out, N+1)
    # x = np.linspace(0.01, 1000, N+1)
    
    x_s = np.array([(x[i+1]+x[i])/2 for i in range(N)])
    
    # Initial time
    t = 0

    # C = 9*tau*10**4
    C = 6*tau
    # C = 9*tau*10**4
    # C = 9*tau

    # Solution function
    u = np.zeros(N)
    # Array for finding error
    error = np.zeros(N)
    # Initial  ondition
    for i in range(N):
        # u[i] = u_init(x[i]/2+x[i+1]/2)
        u[i] = u_init(x_s[i])

    # Boundary conditions
    # Left
    # u[0] = u_left(u, x[0], x[1], t)
    # Right
    # u[N-1] = u_right(u, x[N-1], t)

    # Array for iterative process
    u_iter = u.copy()

    # Array for RSE
    RP = np.zeros(N-2)
    # Arrays for tridiagonal matrix
    a = np.zeros(N-3)
    b = np.zeros(N-2)
    c = np.zeros(N-3)

    U =np.zeros((int(t_end/tau)+1, N))
    U[0][:] = u
    k = 1
    # The coefficient at the term Sigma_{k+1}
    def q(k):
        # a1 = (x[k]/2+x[k+1]/2)
        # a2 = (x[k-1]/2+x[k]/2)
        # a = 1/(x[k]**2-x[k-1]**2)
        
        # b = a1**(1/2)/(x[k+1]-x[k])
        

        # q = b/a*x[k+1]**(1/2)
        
        # b = (x[k]**(1/2)*a1**(1/2))/(a1-a2)
        
        q = x[k+1]**(1/2)*(x[k+2]+x[k+1])/(x[k+1]**2-x[k]**2)*1/(x[k+2]-x[k])

        return q

    # The coefficient at the term Sigma_{k}
    def w(k):
        # a1 = (x[k]/2+x[k+1]/2)
        # a2 = (x[k-1]/2+x[k]/2)
        # a3 = (x[k-1]/2+x[k-2]/2)
        # a = 1/(x[k]**2-x[k-1]**2)        
        # b = a1**(1/2)/(x[k+1]-x[k])
        # c = a2**(1/2)/(x[k]-x[k-1])
        
        # w = (b+c)/a*x[k]**(1/2)
        # b = (x[k]**(1/2)*a2**(1/2))/(a2-a1)
        # c = (x[k-1]**(1/2)*a2**(1/2))/(a2-a3)
        
        w1 = x[k+1]**(1/2)*(x[k+1]+x[k])/(x[k+1]**2-x[k]**2)*1/(x[k+2]-x[k])
        w2 = x[k]**(1/2)*(x[k+1]+x[k])/(x[k+1]**2-x[k]**2)*1/(x[k+1]-x[k-1])
        w = w1 + w2
        return w

    # The coefficient at the term Sigma_{k-1}
    def e(k):
        # a2 = (x[k-1]/2+x[k]/2)
        # a3 = (x[k-1]/2+x[k-2]/2)
        # a = 1/(x[k]**2-x[k-1]**2)        
        # c = a2**(1/2)/(x[k]-x[k-1])
        
        # e = c/a*x[k-1]**(1/2)
        
        # c = (x[k-1]**(1/2)*a3**(1/2))/(a2-a3)
        
        e = x[k]**(1/2)*(x[k-1]+x[k])/(x[k+1]**2-x[k]**2)*1/(x[k+1]-x[k-1])
        return e

    # while t < t_end:
    for _ in range(M-1):
        # Iterative process
        while True:
            # Array for comparison of accuracy
            u_iter_old = u_iter.copy()

            # Filling RSE
            for i in range(1, N-3):
                RP[i] = u[i+1]
                
            RP[0] = u[1]+C*e(2)*D(u_iter[0], x[0], t+tau) * \
                u_left(u, x[0], x[1], t+tau)
                
            RP[N-3] = u[N-2]+C*q(N-3)*D(u_iter[N-1], x[N-1],
                                        t+tau)*u_right(u, x[N-1], t+tau)

            # Filling lower diagonal
            for i in range(N-3):
                a[i] = -C*e(i+2)*D(u_iter[i+1], x[i+1], t+tau)

            # Filling middle diagonal
            for i in range(N-2):
                b[i] = 1 + C*w(i+1)*D(u_iter[i+1], x[i+1], t+tau)
            # Filling upper diagonal
            for i in range(N-3):
                c[i] = -C*q(i+1)*D(u_iter[i+2], x[i+2], t+tau)

            # Solving system of equations
            u_iter[1:N-1] = Tridiagonal_solver(a, b, c, RP)
            
            # Linear system of equations solver
           
            # LES = np.diag(b,0) + np.diag(c,1) + np.diag(a,-1) 
            # u_iter[1:N-1] = np.linalg.solve(LES, RP)
            # u_iter[1:N-1] = scipy.linalg.solve(LES, RP)
            
            # if np.allclose(np.dot(LES, u_iter[1:N-1]), RP):
                # print('OK')
            # else:
                # print('NOT')

            # Finding error
            for i in range(N):
                # error[i] = abs(u_iter_old[i]-u_iter[i])
                error[i] = abs(u_iter_old[i]-u_iter[i])

            # Loop exit
            if max(error) < epsilon:
                break

        # New boundary conditions
        u_iter[0] = u_left(u, x[0], x[1], t+tau)
        # u_iter[N-1] = u_right(u, x[N-1], t+tau)
        u_iter[N-1] = u_right(u, x[N-1]/2+x[N-2]/2, t+tau)

        # Array on the next time layer
        u = u_iter.copy()
        # u = u_iter
        U[k][:] = u
        # t += tau
        k+=1
        
    return U
