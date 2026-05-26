# %%
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Setup & Data Generation
# ==========================================
torch.manual_seed(42)

n_annotators = 2000  # Number of rankings to generate
m_items = 10         # Slate size per ranking
k_clusters = 3       # Number of latent clusters / features

# Item features: X is (m, k)
X = torch.randn(m_items, k_clusters)

# True Theta is an Identity matrix: Cluster i strictly uses Feature i
true_Theta = torch.eye(k_clusters)
true_alpha = torch.ones(k_clusters) / k_clusters # Uniform mixing weights

print("Generating synthetic rankings...")
rankings = []
true_assignments = []

for _ in range(n_annotators):
    # Sample a cluster assignment for this annotator
    c = torch.multinomial(true_alpha, 1).item()
    true_assignments.append(c)
    
    # Calculate true utilities: U = X @ theta_c
    U = X @ true_Theta[c]
    
    # Plackett-Luce generation via Gumbel trick
    gumbel_noise = -torch.empty_like(U).exponential_().log()
    scores = U + gumbel_noise
    
    # Sort descending to get the ranking (item indices)
    ranking = torch.argsort(scores, descending=True)
    rankings.append(ranking)

rankings = torch.stack(rankings) # Shape: (n, m)

# %%
# ==========================================
# 2. EM Algorithm (Mixture of Plackett-Luce)
# ==========================================
def get_pl_log_probs(rankings, theta, X):
    """Computes the Plackett-Luce log-likelihood for a batch of rankings."""
    U = X @ theta # Shape: (m,)
    U_ranked = U[rankings] # Gather utilities in ranked order, Shape: (n, m)
    
    log_probs = 0
    # Sequential choice probability: P(item t | remaining items t...m)
    for i in range(m_items - 1):
        logits = U_ranked[:, i:]
        log_probs += logits[:, 0] - torch.logsumexp(logits, dim=1)
    return log_probs

# Initialize parameters randomly
hat_Theta = torch.randn(k_clusters, k_clusters, requires_grad=True)
hat_alpha = torch.randn_like(true_alpha)
hat_alpha = torch.softmax(hat_alpha, dim=0) # Ensure mixing weights sum to 1
optimizer = torch.optim.Adam([hat_Theta], lr=0.1)

n_em_steps = 20
inner_m_steps = 15

print("\nStarting EM Algorithm...")
for step in range(n_em_steps):
    
    # --- E-STEP: Compute responsibilities (gamma) ---
    with torch.no_grad():
        log_P = torch.zeros(n_annotators, k_clusters)
        for c in range(k_clusters):
            log_P[:, c] = torch.log(hat_alpha[c]) + get_pl_log_probs(rankings, hat_Theta[c], X)
        
        # Log-sum-exp trick to prevent underflow
        log_marginal = torch.logsumexp(log_P, dim=1, keepdim=True)
        gamma = torch.exp(log_P - log_marginal) # Posterior probabilities (n, k)
        
        # Update mixing weights
        hat_alpha = gamma.mean(dim=0)
        
    # --- M-STEP: Maximize weighted log-likelihood ---
    for _ in range(inner_m_steps):
        optimizer.zero_grad()
        loss = 0
        for c in range(k_clusters):
            log_probs = get_pl_log_probs(rankings, hat_Theta[c], X)
            # Weight the log likelihood by the E-step posteriors
            loss -= (gamma[:, c] * log_probs).sum()
        
        # Add a small L2 penalty to anchor the scale (since PL utilities are scale-invariant 
        # relative to the implied Gumbel noise variance of 1.0)
        loss += 0.5 * (hat_Theta ** 2).sum()
        
        loss.backward()
        optimizer.step()

    if (step + 1) % 5 == 0:
        print(f"EM Step {step + 1:2d} | Negative Log-Likelihood: {loss.item():.2f}")

    print(hat_Theta)

# %%
# ==========================================
# 3. Evaluation & Cluster Alignment
# ==========================================
learned_Theta = hat_Theta.detach().numpy()
true_Theta_np = true_Theta.numpy()

# Use Hungarian Matching to align learned clusters to true clusters
# based on the negative cosine similarity (cost) between the vectors
cost_matrix = np.zeros((k_clusters, k_clusters))
for i in range(k_clusters):
    for j in range(k_clusters):
        sim = np.dot(learned_Theta[i], true_Theta_np[j]) / (np.linalg.norm(learned_Theta[i]) * np.linalg.norm(true_Theta_np[j]))
        cost_matrix[i, j] = -sim

row_ind, col_ind = linear_sum_assignment(cost_matrix)
aligned_Theta = learned_Theta[row_ind]

print("\n=== Results ===")
print("True Theta (Identity Matrix):")
print(np.round(true_Theta_np, 2))

print("\nLearned Theta (Aligned & Normalized):")
# Normalize the learned vectors to norm 1.0 to easily compare against the one-hot truth
normalized_Theta = aligned_Theta / np.linalg.norm(aligned_Theta, axis=1, keepdims=True)
print(np.round(normalized_Theta, 2))