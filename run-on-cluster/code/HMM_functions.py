import numpy as np
from collections import Counter

### Forward algorithm (p.262 and p.272 of Rabiner, also read https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec14.pdf)
### target: sequence of numbers representing the nucleotide sequence of a putative recombinant
### ref: matrix of allele frequencies in each cluster, looks like:
# [[1 'A' 0.0 ... 0.0 0.0 1.0]
#  [1 'T' 0.0 ... 1.0 0.0 0.0]
#  [1 'C' 1.0 ... 0.0 0.0 0.0]
#  ...
#  [100 'T' 0.0 ... 1.0 0.0 0.0]
#  [100 'C' 1.0 ... 0.0 0.0 0.0]
#  [100 'G' 0.0 ... 0.0 1.0 0.0]]
def forward_scaled_matrix(target, ref, e, s, pi = None):
    # Run function to organize allele frequencies into emission matrix
    ref = calculate_emission(ref, e)
    
    # N: number of clusters    
    N = int(ref.shape[0]/4)
    # T: length of genome
    T = len(target)
    
    # pi is the initial cluster probabilities, which we set to be equal for every cluster
    if pi is None:                       # keep current default
        pi_vec = np.full(N, 1 / N)
    else:
        pi_vec = np.asarray(pi, dtype=float)
        if pi_vec.size != N:
            raise ValueError(f"pi length {pi_vec.size} ≠ number of states {N}")
        if not np.isclose(pi_vec.sum(), 1.0):
            pi_vec = pi_vec / pi_vec.sum()

    # A: transition matrix, N by N, diagonal entries are s, non-diag. entries are (1-s)/(N-1)
    A = get_transition_prob(N, s)
    
    # Initialize three matrices used for forward algorithm 
    alpha = np.zeros((N, T))
    alpha_hat = np.zeros((N, T))
    alpha_tilde = np.zeros((N, T))
    
    # Scaling factor
    g = np.zeros(T)

    # First, we follow Equation 19 in Rabiner
    for i in range(N):
        alpha[i,0] = pi_vec[i]*get_emission_prob(i, 0, target[0], ref)
    # Get scaling factor for first position
    g[0] = sum(alpha[:,0])
    # Divide by scaling factor to get alpha_hat
    alpha_hat[:,0] = alpha[:,0]/g[0]

    # Useful for matrix calculations
    alpha_hat = np.ascontiguousarray(alpha_hat)
    A = np.ascontiguousarray(A)

    # Induction step
    for t in range(1,T):
        # For each t, iterate over clusters, and get vector of emission probabilities, each corresponding to a cluster
        b_jt = np.array(list(map(lambda j: get_emission_prob(j, t, target[t], ref), list(range(N)))))
        # Obtain new alpha_hat
        # We're actually not using alpha_tilde, but might be useful to check calculations later
        alpha_tilde[:,t], g[t], alpha_hat[:,t] = matrix_calculations_fast(t, alpha_hat, b_jt, s, N)

    # The scaling factor can be used to obtain the log-likelihood (see https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec14.pdf)    
    return g, alpha_hat

### Function for backward probabilities (only used to calculate the MMPP)
def backward_scaled_matrix(target, ref, e, s, g):
    ref = calculate_emission(ref, e)
    N = int(ref.shape[0]/4)
    T = len(target)
    A = get_transition_prob(N, s)
    
    beta_hat = np.zeros((N, T))
    beta_hat[:, -1] = 1  # Initialization (scaled)
    
    for t in range(T-2, -1, -1):
        b_jt1 = np.array([get_emission_prob(j, t+1, target[t+1], ref) for j in range(N)])
        beta_hat[:, t] = A @ (b_jt1 * beta_hat[:, t+1]) / g[t]
    
    return beta_hat

### Organize allele frequencies to emission matrix
def calculate_emission(original_ref, e):
    ref = original_ref.copy()
    # Remove first two columns of the matrix
    allele_counts = ref[:, 2:].astype(float)
    # Add pseudocount to matrix 
    allele_counts += e
    # Get proportions by dividing every entry by 1+4*e
    proportions = allele_counts/(1 + 4*e)
    return(proportions)

### Get transition matrix, diagonal entries are s, non-diag. entries are (1-s)/(N-1)
def get_transition_prob(N, s):
    # diagonal entries are s
    # non-diag. entries are (1-s)/(N-1)
    A = np.full((N, N), (1-s)/(N-1)) #N by N matrix
    np.fill_diagonal(A, s)
    return(A)
    
### Function to locate emission probabilities from matrix
### n: which cluster (starts from 0), t: which position (starts from 0), target_allele: which allele, ref: emission matrix
def get_emission_prob(n, t, target_allele, ref):
    target_allele = int(target_allele)
    # Get a vector of allele frequencies corresponding to the cluster and position
    alleles = ref[(n*4):(n*4+4),t]
    if target_allele != 4:
        return(alleles[target_allele])
    # If allele is unknown on the putative recombinant, return 1
    else:
        return(1)

### Function for induction step of forward algorithm    
def matrix_calculations(t, alpha_hat, A, b_jt):
    # a_ij corresponds to transition probability between clusters i and j
    # alpha_hat[:,t-1] is a vector of probabilities, [alpha_hat_{t-1}(1), alpha_hat_{t-1}(2), alpha_hat_{t-1}(3),...]
    # b_jt is a vector of emission probabilities corresponding to each cluster 
    res = np.dot(alpha_hat[:,t-1], A)*b_jt
    # The third position that is returned corresponds to the new alpha_hat, but the denominator (second position) represents the scaling factor at t
    return(res, sum(res), res/sum(res))

def matrix_calculations_fast(t, alpha_hat, b_jt, s, N):
    """
    Fast forward step for T_ii = s, T_ij = (1-s)/(N-1), j!=i.
    Computes:
      res = ((A @ alpha_hat[:,t-1]) * b_jt)
      g_t = sum(res)
      alpha_hat[:,t] = res / g_t
    """
    u = (1.0 - s) / (N - 1.0)         # off-diagonal value
    tmp = (s - u) * alpha_hat[:, t-1] + u   # (A @ alpha_hat_{t-1})
    res = tmp * b_jt
    gt = res.sum()
    return (res, gt, res / gt)

### Function used in run_HMM.py to optimize e and s parameters by minimizing the negative log-likelihood of the sequence
### sample: sequence of numbers representing the nucleotide sequence of a putative recombinant
### allele_counts: matrix of allele frequencies in each cluster, looks like:
# [[1 'A' 0.0 ... 0.0 0.0 1.0]
#  [1 'T' 0.0 ... 1.0 0.0 0.0]
#  [1 'C' 1.0 ... 0.0 0.0 0.0]
#  ...
#  [100 'T' 0.0 ... 1.0 0.0 0.0]
#  [100 'C' 1.0 ... 0.0 0.0 0.0]
#  [100 'G' 0.0 ... 0.0 1.0 0.0]]
def nll(params, sample, allele_counts, pi=None):
    log_e, k = params
    L = len(sample)                               # use len(), not length()
    e = np.exp(log_e)
    s = 1 - k / (L - 1)

    g, _ = forward_scaled_matrix(sample, allele_counts, e, s, pi)
    return -np.sum(np.log(g))                     # np.sum is fine but not required

def nll_sigma_1(params, sample, allele_counts, pi=None):
    log_e = params[0] if np.ndim(params) else params  # minimise may pass array
    e = np.exp(log_e)

    g, _ = forward_scaled_matrix(sample, allele_counts, e, 1.0, pi)  # s fixed at 1
    return -np.sum(np.log(g))

### Function to run Viterbi algorithm (see p.264 of Rabiner)
def viterbi_haplotype_states(target, ref, s, e, pi=None):
    # Get emission matrix
    ref = calculate_emission(ref, e)
    
    # This part is the same as the forward algorithm
    N = int(ref.shape[0]/4)
    T = len(target)
    
    if pi is None:
        pi_vec = np.full(N, 1 / N)
    else:
        pi_vec = np.asarray(pi, dtype=float)
        if pi_vec.size != N:                     # <- add back
            raise ValueError(f"pi length {pi_vec.size} ≠ number of states {N}")
        if not np.isclose(pi_vec.sum(), 1.0):
            pi_vec /= pi_vec.sum()

    A = get_transition_prob(N, s)
    delta = np.zeros((N, T)) # N by T matrix
    psi = np.zeros((N, T), dtype = int) # N by T matrix

    # Initialization step
    # For each cluster
    for i in range(N):
        # Obtain delta and psi (see p.264)
        # delta is logged in this implementation
        delta[i, 0] = np.log(pi_vec[i]*get_emission_prob(i, 0, target[0], ref))
        psi[i, 0] = 0
    
    # Recursive step
    for t in range(1, T):
        for j in range(N):
            # delta is already logged
            temp = delta[:,t-1] + np.log(A[:,j])
            delta[j, t] = max(temp) + np.log(get_emission_prob(j, t, target[t], ref))
            psi[j, t] = np.argmax(temp)
    
    # Get p star and q star (see p.264)        
    P = max(delta[:,T-1])
    q_T = np.argmax(delta[:,T-1])
    q_star = np.zeros(T)
    q_star[T-1] = q_T
    
    for t in range(T-2, -1, -1):
        q_star[t] = psi[int(q_star[t+1]), t+1]
        
    return q_star, delta

### Function to compute the MMPP
def compute_mmpp(alpha_hat, beta_hat):
    posterior = alpha_hat * beta_hat  # elementwise multiplication
    posterior /= posterior.sum(axis=0)  # normalize each column to sum to 1
    
    max_posteriors = np.max(posterior, axis=0)  # max over lineages at each position
    mmpp = np.mean(max_posteriors)
    return mmpp

### Wrapper function to compute the MMPP
def get_mmpp(target, ref, s, e, pi=None):
    g, alpha_hat = forward_scaled_matrix(target, ref, e, s, pi)
    beta_hat = backward_scaled_matrix(target, ref, e, s, g)
    mmpp = compute_mmpp(alpha_hat, beta_hat)
    return mmpp

