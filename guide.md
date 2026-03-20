WMSS: Replication Guide
Here's a complete guide to replicate the Weak-Driven Learning method from this paper.
Core Idea
Instead of continuing standard SFT when a model plateaus, use historical weak checkpoints to inject corrective signals. The weak model's confusion on hard negatives amplifies gradients that have vanished in the strong model, enabling continued learning.

Method Overview: Three-Stage Pipeline
Stage 1: Initialize Weak & Strong Agents

Start with base model $M_0$
Perform standard SFT on your dataset $D$ → get $M_1$
Set:

$M_{\text{weak}} \leftarrow M_0$ (base checkpoint - frozen)
$M_{\text{strong}} \leftarrow M_1$ (SFT checkpoint - trainable)

Stage 2: Curriculum-Enhanced Data Activation
This stage identifies which samples need more attention based on entropy dynamics.
Step 2.1: Compute Entropy for Each Sample
For each training sample $x_i$:
Python# Run forward passes (no gradient)
with torch.no_grad():
    logits_weak = M_weak(x_i)
    logits_strong = M_strong(x_i)
    
    # Compute predictive entropy
    H_weak = entropy(softmax(logits_weak))
    H_strong = entropy(softmax(logits_strong))
    
    # Entropy change
    ΔH_i = H_strong - H_weak

Step 2.2: Construct Curriculum Weights
For each sample, compute sampling weight:
$$
p_i \propto \alpha \cdot H(M_{\text{weak}}; x_i) + \beta \cdot [-\Delta H_i]_+ + \gamma \cdot [\Delta H_i]_+
$$
where $[u]_+ = \max(u, 0)$
Hyperparameters (from paper):

$\alpha = 0.1$ (base difficulty)
$\beta = 0.8$ (consolidation - samples where model became more certain)
$\gamma = 0.1$ (regression repair - samples where model became less certain)

Python# Compute weights
base_difficulty = alpha * H_weak
consolidation = beta * torch.clamp(-ΔH, min=0)
regression_repair = gamma * torch.clamp(ΔH, min=0)

p_i = base_difficulty + consolidation + regression_repair

# Normalize to get sampling probabilities
p_i = p_i / p_i.sum()

Step 2.3: Sample Active Dataset
Python# Weighted sampling for training
D_active = weighted_sample(D, weights=p_i, num_samples=len(D))


Stage 3: Joint Training via Logit Mixing
This is the core innovation. Train the strong model using mixed logits.
Implementation
Python# Hyperparameters
λ = 0.5  # mixing coefficient (paper recommends 0.42-0.48 range)
lr = 1e-5
max_seq_len = 8192

# Training loop
for batch (x, y) in D_active:
    # Forward pass - weak model FROZEN
    with torch.no_grad():
        z_weak = M_weak(x)  # logits from weak model
    
    z_strong = M_strong(x)  # logits from strong model (trainable)
    
    # Mix logits
    z_mix = λ * z_strong + (1 - λ) * z_weak
    
    # Compute loss on MIXED logits
    loss = CrossEntropyLoss(z_mix, y)
    
    # Backward pass - only M_strong gets updated
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

Critical details:

Weak model is completely frozen (no gradient computation)
Loss is computed on mixed logits $z_{\text{mix}}$
Gradients backprop through mixing operation to update only $M_{\text{strong}}$
Use the same learning rate as standard SFT


Complete Algorithm (Pseudocode)
Pythondef WMSS(M_0, dataset_D, num_iterations=4):
    # Stage 1: Initialization
    M_1 = standard_SFT(M_0, D)
    M_weak = M_0.copy()
    M_strong = M_1.copy()
    
    # Stage 2 & 3: Iterative improvement
    for iteration in range(num_iterations):
        # === Stage 2: Curriculum Data Activation ===
        ΔH = compute_entropy_dynamics(M_strong, M_weak, D)
        p = compute_curriculum_weights(ΔH, α=0.1, β=0.8, γ=0.1)
        D_active = weighted_sample(D, p)
        
        # === Stage 3: Joint Training ===
        M_strong = joint_train(M_strong, M_weak, D_active, λ=0.5)
        
        # Update weak reference for next iteration
        M_weak = M_strong.copy()
    
    return M_strong

def joint_train(M_strong, M_weak, D_active, λ=0.5):
    M_weak.eval()  # Freeze weak model
    M_strong.train()
    
    for batch (x, y) in D_active:
        with torch.no_grad():
            z_weak = M_weak(x)
        
        z_strong = M_strong(x)
        z_mix = λ * z_strong + (1 - λ) * z_weak
        
        loss = cross_entropy(z_mix, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return M_strong


Key Hyperparameters

ParameterValueDescription$\alpha$0.1Base difficulty weight$\beta$0.8Consolidation weight$\gamma$0.1Regression repair weight$\lambda$0.5Logit mixing coefficient (optimal: 0.42-0.48)Learning rate1×10⁻⁵Same as standard SFTSequence length8192For trainingIterations3-4More causes overfitting (see Fig 3)

Experimental Setup (for replication)
Dataset

Math: AM-1.4M dataset (filtered with math_verify → 111,709 samples)
Code: AM-1.4M code subset (execution-filtered → 104,077 samples)
Total: ~215k high-quality samples

Models Tested

Qwen3-4B-Base
Qwen3-8B-Base

Training Infrastructure

8× NVIDIA H800 GPUs
TRL library + Hugging Face transformers
Global batch size: standard for your model size

Evaluation Benchmarks

Math: AIME2025, MATH500, AMC23, AQuA, GSM8K, MAWPS, SVAMP
Code: HumanEval, MBPP

Critical Implementation Details
1. Entropy Computation
Pythondef entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

2. Stopping Criterion

Monitor validation performance per epoch
Stop at Epoch 3-4 (paper shows regression after this)
AMC2023 shows sharp drop after Epoch 3 (Figure 3)

3. Logit Mixing Detail

Mix happens in logit space, not probability space
Then apply softmax once on mixed logits
❌ Wrong: softmax(λ*z_strong) + (1-λ)*softmax(z_weak)
✅ Correct: softmax(λ*z_strong + (1-λ)*z_weak)

4. Weak Model Updates
In iterative training (multiple iterations), you can optionally update the weak reference:

After each iteration, copy current $M_{\text{strong}}$ to $M_{\text{weak}}$
This creates a moving reference that tracks improvement

Expected Results
Math Reasoning (Qwen3-4B-Base)

MethodAIME2025MATH500GSM8KAvgSFT12.266.183.954.1WMSS20.071.388.559.9
Key win: ~2× improvement on hardest benchmark (AIME2025)

Why It Works (Intuition)


Saturation bottleneck: Standard SFT stops improving when model is already confident (gradient vanishes on non-target tokens)


Weak signal injection: Historical weak checkpoint has higher probability on "plausible mistakes"


Gradient amplification: Mixing reintroduces mass on hard negatives → larger gradients → continued learning


Suppression-dominant: Strong model learns by suppressing distractors (non-target logit mean: 2.09 → 0.90) rather than boosting already-high target logits