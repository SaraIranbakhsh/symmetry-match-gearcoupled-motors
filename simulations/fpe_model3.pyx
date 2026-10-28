# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, sin, cos
from cython.parallel import prange

# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

# =============================================================================
# =============================================================================
# ==========
# ========== INTERFACES
# ==========
# =============================================================================
# =============================================================================

def launchpad_reference(
    double n0, double n1,
    double phase,
    double[:] positions,
    double[:, :] prob, double[:, :] p_now,
    double[:, :] p_last, double[:, :] p_last_ref,
    double[:, :] potential_at_pos,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, unsigned int check_step,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double dt, double m1, double m2, double beta, double gamma1, double gamma2
    ):

    cdef:
        double     Z = 0.0

        Py_ssize_t i, j # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        for j in range(N):
            potential_at_pos[i, j] = potential(
                positions[i], positions[j],
                n0, n1, phase, E0, Ecouple, E1
                )

    # calculate the partition function
    for i in range(N):
        for j in range(N):
            Z += exp(-beta*potential_at_pos[i, j])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        for j in range(N):
            prob[i, j] = exp(-beta*potential_at_pos[i, j])/Z

    # build the drift vector
    construct_drift_vec_func(
        drift_at_pos, positions,
        E0, Ecouple, E1, mu_Hp, mu_atp,
        phase, n0, n1, m1, m2,
        gamma1, gamma2, N
        )

    # build diffusion tensor
    construct_diffusion_tensor_func(
        diffusion_at_pos, positions, m1, m2, beta, gamma1, gamma2, N
        )

    # initialize the simulation to steady state distribution
    # start with Gibbs-Boltzmann distribution as initial guess
    for i in range(N):
        for j in range(N):
            p_now[i, j] = prob[i, j]

    steady_state_initialize(
        p_now, p_last, p_last_ref,
        drift_at_pos, diffusion_at_pos,
        N, dx, dt, check_step
    )

def construct_drift_vec(
    double[:, :, :] drift_at_pos, double[:] positions,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double phase, double n0, double n1, double m1, double m2,
    double gamma1, double gamma2, int N
    ):

    construct_drift_vec_func(
        drift_at_pos, positions,
        E0, Ecouple, E1, mu_Hp, mu_atp,
        phase, n0, n1, m1, m2,
        gamma1, gamma2, N
        )

def construct_diffusion_tensor(
    double[:, :, :] diffusion_at_pos, double[:] positions,
    double m1, double m2, double beta, double gamma1, double gamma2, int N
    ):

    construct_diffusion_tensor_func(
        diffusion_at_pos, positions, m1, m2, beta, gamma1, gamma2, N
        )

# =============================================================================
# =============================================================================
# ==========
# ========== CONSTRUCTOR FUNCTIONS
# ==========
# =============================================================================
# =============================================================================

cdef void construct_drift_vec_func(
    double[:, :, :] drift_at_pos, double[:] positions,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double phase, double n0, double n1, double m1, double m2,
    double gamma1, double gamma2, int N
    ) noexcept nogil:

    cdef Py_ssize_t i, j

    for i in range(N):
        for j in range(N):
            drift_at_pos[0, i, j] = drift1(
                positions[i], positions[j],
                n0, n1,phase, E0, Ecouple, mu_Hp,
                m1, gamma1
                )
            drift_at_pos[1, i, j] = drift2(
                positions[i], positions[j],
                n0, n1, E1, Ecouple, mu_atp,
                m2, gamma2
                )

cdef void construct_diffusion_tensor_func(
    double[:, :, :] diffusion_at_pos, double[:] positions,
    double m1, double m2, double beta, double gamma1, double gamma2,
    int N
    ) noexcept nogil:

    cdef Py_ssize_t i, j

    for i in range(N):
        for j in range(N):
            # diffusion tensor is diagonal: only [0] and [1] are non-zero
            # diffusion02 and diffusion11 are identically 0.0 — not stored
            diffusion_at_pos[0, i, j] = diffusion01(
                positions[i], positions[j],
                m1, beta, gamma1
                )
            diffusion_at_pos[1, i, j] = diffusion12(
                positions[i], positions[j],
                m2, beta, gamma2
                )

# =============================================================================
# =============================================================================
# ==========
# ========== ENERGETICS AND FORCES
# ==========
# =============================================================================
# =============================================================================

cdef double potential(
    double position0, double position1,
    double n0, double n1,
    double phase, double E0, double Ecouple, double E1
    ) noexcept nogil:
    #model 3,4
    return 0.5*(
        E0*(1-cos(n0*(position0-phase)))
        + Ecouple*(1-cos(n0*position0-n1*position1))
        + E1*(1-cos((n1*position1)))
        )
    #model 1,2
    '''return 0.5*(
        E0*(1-cos(n0*(position0-phase)))
        + Ecouple*(1-cos(position0-position1))
        + E1*(1-cos((n1*position1)))
        )'''

cdef double drift1(
    double position0, double position1, double n0, double n1,
    double phase, double E0, double Ecouple, double mu_Hp,
    double m1, double gamma1
    ) noexcept nogil:
    # Returns the force on system F0. H+ chemical potential set up so that
    # postive values of chemical potential returns postive values of the flux
    # for F0
    #model 3
    return (-1.0/(m1*gamma1))*((0.5)*(
        Ecouple*n0*sin(n0*position0-n1*position1)
        + (n0*E0*sin(n0*(position0-phase)))
        ) - mu_Hp)
    #model 1
    '''return (-1.0/(m1*gamma1))*((0.5)*(
        Ecouple*sin(position0-position1)
        + (n0*E0*sin(n0*(position0-phase)))
        ) - mu_Hp)'''
    #model 2
    '''return (-1.0/(m1*gamma1))*((0.5)*(
        Ecouple*sin(position0-position1)
        + (n0*E0*sin(n0*(position0-phase)))
        ) - (mu_Hp*n0/8.0))'''
    #model 4
    '''return (-1.0/(m1*gamma1))*((0.5)*(
        Ecouple*n0*sin(n0*position0-n1*position1)
        + (n0*E0*sin(n0*(position0-phase)))
        ) - (mu_Hp*n0/8.0))'''

cdef double drift2(
    double position0, double position1, double n0, double n1,
    double E1, double Ecouple, double mu_atp,
    double m2, double gamma2
    ) noexcept nogil:
    # Returns the force on system F1. Chemical potential \mu_{ATP} set up so
    # that postive values of chemical potential returns positive values of
    # the of the flux for F1
    #model 3
    return (-1.0/(m2*gamma2))*((0.5)*(
        (-1.0)*n1*Ecouple*sin(n0*position0-n1*position1)
        + (n1*E1*sin(n1*position1))
        ) - mu_atp)
    #model 1
    '''return (-1.0/(m2*gamma2))*((0.5)*(
        (-1.0)*Ecouple*sin(position0-position1)
        + (n1*E1*sin(n1*position1))
        ) - mu_atp)'''
    #model 2
    '''return (-1.0/(m2*gamma2))*((0.5)*(
        (-1.0)*Ecouple*sin(position0-position1)
        + (n1*E1*sin(n1*position1))
        ) - (mu_atp*n1/3.0))'''
    #model 4
    '''return (-1.0/(m2*gamma2))*((0.5)*(
        (-1.0)*n1*Ecouple*sin(n0*position0-n1*position1)
        + (n1*E1*sin(n1*position1))
        ) - (mu_atp*n1/3.0))'''

cdef double diffusion01(
    double position0, double position1,
    double m1, double beta, double gamma1
    ) noexcept nogil:
    # return the 11 element of the diffusion tensor
    return 1.0/(beta*m1*gamma1)

cdef double diffusion12(
    double position0, double position1,
    double m2, double beta, double gamma2
    ) noexcept nogil:
    # return the 22 element of the diffusion tensor
    return 1.0/(beta*m2*gamma2)

# =============================================================================
# =============================================================================
# ==========
# ========== INITIALIZATIONS
# ==========
# =============================================================================
# =============================================================================

cdef void steady_state_initialize(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] p_last_ref,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, double dt, unsigned int check_step,
    ) noexcept nogil:

    cdef:
        double              tot_var_dist       = 0.0
        int                 continue_condition = 1

        # counters
        Py_ssize_t          i, j, ii, jj
        unsigned long       step_counter       = 0

    while continue_condition:

        for i in range(N):
            for j in range(N):
                # save previous distribution
                p_last[i, j] = p_now[i, j]
                # reset to zero
                p_now[i, j] = 0.0

        # advance probability one time step
        update_probability_full(
            p_now,
            p_last,
            drift_at_pos,
            diffusion_at_pos,
            N, dx, dt
            )

        if step_counter == check_step:

            for i in range(N):
                for j in range(N):
                    tot_var_dist += 0.5*fabs(p_last_ref[i, j]-p_now[i, j])

            # check convergence criteria
            if tot_var_dist < float64_eps:
                continue_condition = 0
            else:
                tot_var_dist = 0.0 # reset total variation distance
                step_counter = 0 # reset step counter
                # make current distribution the reference distribution
                for i in range(N):
                    for j in range(N):
                        p_last_ref[i, j] = p_now[i, j]

        step_counter += 1

cdef void update_probability_full(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, double dt
    ) noexcept nogil:

    # update the distribution using FTCS method.
    # diffusion tensor is diagonal: diffusion_at_pos[0] = D11, diffusion_at_pos[1] = D22
    # cross terms (D12, D21) are identically zero and have been removed.

    # declare iterator variables
    cdef Py_ssize_t i, j

    # Periodic boundary conditions:
    # Explicit update FPE for the corners
    p_now[0, 0] = p_last[0, 0] + dt*(
        -(drift_at_pos[0, 1, 0]*p_last[1, 0]-drift_at_pos[0, N-1, 0]*p_last[N-1, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 1, 0]*p_last[1, 0]-2.0*diffusion_at_pos[0, 0, 0]*p_last[0, 0]+diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0])/(dx*dx)
        -(drift_at_pos[1, 0, 1]*p_last[0, 1]-drift_at_pos[1, 0, N-1]*p_last[0, N-1])/(2.0*dx)
        +(diffusion_at_pos[1, 0, 1]*p_last[0, 1]-2.0*diffusion_at_pos[1, 0, 0]*p_last[0, 0]+diffusion_at_pos[1, 0, N-1]*p_last[0, N-1])/(dx*dx)
        )
    p_now[0, N-1] = p_last[0, N-1] + dt*(
        -(drift_at_pos[0, 1, N-1]*p_last[1, N-1]-drift_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 1, N-1]*p_last[1, N-1]-2.0*diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        -(drift_at_pos[1, 0, 0]*p_last[0, 0]-drift_at_pos[1, 0, N-2]*p_last[0, N-2])/(2.0*dx)
        +(diffusion_at_pos[1, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[1, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[1, 0, N-2]*p_last[0, N-2])/(dx*dx)
        )
    p_now[N-1, 0] = p_last[N-1, 0] + dt*(
        -(drift_at_pos[0, 0, 0]*p_last[0, 0]-drift_at_pos[0, N-2, 0]*p_last[N-2, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[0, N-2, 0]*p_last[N-2, 0])/(dx*dx)
        -(drift_at_pos[1, N-1, 1]*p_last[N-1, 1]-drift_at_pos[1, N-1, N-1]*p_last[N-1, N-1])/(2.0*dx)
        +(diffusion_at_pos[1, N-1, 1]*p_last[N-1, 1]-2.0*diffusion_at_pos[1, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[1, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        )
    p_now[N-1, N-1] = p_last[N-1, N-1] + dt*(
        -(drift_at_pos[0, 0, N-1]*p_last[0, N-1]-drift_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]-2.0*diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(dx*dx)
        -(drift_at_pos[1, N-1, 0]*p_last[N-1, 0]-drift_at_pos[1, N-1, N-2]*p_last[N-1, N-2])/(2.0*dx)
        +(diffusion_at_pos[1, N-1, 0]*p_last[N-1, 0]-2.0*diffusion_at_pos[1, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[1, N-1, N-2]*p_last[N-1, N-2])/(dx*dx)
        )

    # iterate through all the coordinates (not corners) for both variables
    for i in prange(1, N-1):
        # Periodic boundary conditions:
        # Explicitly update FPE for edges of grid (not corners)
        p_now[0, i] = p_last[0, i] + dt*(
            -(drift_at_pos[0, 1, i]*p_last[1, i]-drift_at_pos[0, N-1, i]*p_last[N-1, i])/(2.0*dx)
            +(diffusion_at_pos[0, 1, i]*p_last[1, i]-2.0*diffusion_at_pos[0, 0, i]*p_last[0, i]+diffusion_at_pos[0, N-1, i]*p_last[N-1, i])/(dx*dx)
            -(drift_at_pos[1, 0, i+1]*p_last[0, i+1]-drift_at_pos[1, 0, i-1]*p_last[0, i-1])/(2.0*dx)
            +(diffusion_at_pos[1, 0, i+1]*p_last[0, i+1]-2.0*diffusion_at_pos[1, 0, i]*p_last[0, i]+diffusion_at_pos[1, 0, i-1]*p_last[0, i-1])/(dx*dx)
            )
        p_now[i, 0] = p_last[i, 0] + dt*(
            -(drift_at_pos[0, i+1, 0]*p_last[i+1, 0]-drift_at_pos[0, i-1, 0]*p_last[i-1, 0])/(2.0*dx)
            +(diffusion_at_pos[0, i+1, 0]*p_last[i+1, 0]-2.0*diffusion_at_pos[0, i, 0]*p_last[i, 0]+diffusion_at_pos[0, i-1, 0]*p_last[i-1, 0])/(dx*dx)
            -(drift_at_pos[1, i, 1]*p_last[i, 1]-drift_at_pos[1, i, N-1]*p_last[i, N-1])/(2.0*dx)
            +(diffusion_at_pos[1, i, 1]*p_last[i, 1]-2.0*diffusion_at_pos[1, i, 0]*p_last[i, 0]+diffusion_at_pos[1, i, N-1]*p_last[i, N-1])/(dx*dx)
            )
        p_now[N-1, i] = p_last[N-1, i] + dt*(
            -(drift_at_pos[0, 0, i]*p_last[0, i]-drift_at_pos[0, N-2, i]*p_last[N-2, i])/(2.0*dx)
            +(diffusion_at_pos[0, 0, i]*p_last[0, i]-2.0*diffusion_at_pos[0, N-1, i]*p_last[N-1, i]+diffusion_at_pos[0, N-2, i]*p_last[N-2, i])/(dx*dx)
            -(drift_at_pos[1, N-1, i+1]*p_last[N-1, i+1]-drift_at_pos[1, N-1, i-1]*p_last[N-1, i-1])/(2.0*dx)
            +(diffusion_at_pos[1, N-1, i+1]*p_last[N-1, i+1]-2.0*diffusion_at_pos[1, N-1, i]*p_last[N-1, i]+diffusion_at_pos[1, N-1, i-1]*p_last[N-1, i-1])/(dx*dx)
            )
        p_now[i, N-1] = p_last[i, N-1] + dt*(
            -(drift_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-drift_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(2.0*dx)
            +(diffusion_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-2.0*diffusion_at_pos[0, i, N-1]*p_last[i, N-1]+diffusion_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(dx*dx)
            -(drift_at_pos[1, i, 0]*p_last[i, 0]-drift_at_pos[1, i, N-2]*p_last[i, N-2])/(2.0*dx)
            +(diffusion_at_pos[1, i, 0]*p_last[i, 0]-2.0*diffusion_at_pos[1, i, N-1]*p_last[i, N-1]+diffusion_at_pos[1, i, N-2]*p_last[i, N-2])/(dx*dx)
            )

        # all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = p_last[i, j] + dt*(
                -(drift_at_pos[0, i+1, j]*p_last[i+1, j]-drift_at_pos[0, i-1, j]*p_last[i-1, j])/(2.0*dx)
                +(diffusion_at_pos[0, i+1, j]*p_last[i+1, j]-2.0*diffusion_at_pos[0, i, j]*p_last[i, j]+diffusion_at_pos[0, i-1, j]*p_last[i-1, j])/(dx*dx)
                -(drift_at_pos[1, i, j+1]*p_last[i, j+1]-drift_at_pos[1, i, j-1]*p_last[i, j-1])/(2.0*dx)
                +(diffusion_at_pos[1, i, j+1]*p_last[i, j+1]-2.0*diffusion_at_pos[1, i, j]*p_last[i, j]+diffusion_at_pos[1, i, j-1]*p_last[i, j-1])/(dx*dx)
                )
