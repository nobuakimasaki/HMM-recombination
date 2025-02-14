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
def forward_scaled_matrix(target, ref, e, s):
    # Run function to organize allele frequencies into emission matrix
    ref = calculate_emission(ref, e)
    
    # N: number of clusters    
    N = int(ref.shape[0]/4)
    # T: length of genome
    T = len(target)
    
    # pi is the initial cluster probabilities, which we set to be equal for every cluster
    pi = np.full(N, 1/N) 
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
        alpha[i,0] = pi[i]*get_emission_prob(i, 0, target[0], ref)
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
        alpha_tilde[:,t], g[t], alpha_hat[:,t] = matrix_calculations(t, alpha_hat, A, b_jt)
    
    # The scaling factor can be used to obtain the log-likelihood (see https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec14.pdf)    
    return g, alpha_hat

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
def nll(params, sample, allele_counts):
    e, s = params
    # Obtain negative log-likelihood by using the forward algorithm
    g, alpha_hat = forward_scaled_matrix(sample, allele_counts, e, s)
    # Using scaling factor to obtain nll (see https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec14.pdf)
    nll = -sum(np.log(g))
    return(nll)

def nll_sigma_1(params, sample, allele_counts):
    e = params
    # Obtain negative log-likelihood by using the forward algorithm
    g, alpha_hat = forward_scaled_matrix(sample, allele_counts, e, 1)
    # Using scaling factor to obtain nll (see https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec14.pdf)
    nll = -sum(np.log(g))
    return(nll)

### Function to run Viterbi algorithm (see p.264 of Rabiner)
def viterbi_haplotype_states(target, ref, s, e):
    # Get emission matrix
    ref = calculate_emission(ref, e)
    
    # This part is the same as the forward algorithm
    N = int(ref.shape[0]/4)
    T = len(target)
    
    pi = np.full(N, 1/N) 
    A = get_transition_prob(N, s)
    
    pi = np.full(N, 1/N) # vector of length N
    delta = np.zeros((N, T)) # N by T matrix
    psi = np.zeros((N, T)) # N by T matrix

    # Initialization step
    # For each cluster
    for i in range(N):
        # Obtain delta and psi (see p.264)
        # delta is logged in this implementation
        delta[i, 0] = np.log(pi[i]*get_emission_prob(i, 0, target[0], ref))
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