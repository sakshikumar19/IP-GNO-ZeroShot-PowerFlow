## Research Gap
Nobody has successfully combined the mesh-free, resolution-invariant properties of Neural Operators with Physics-Informed loss for zero-shot transfer across different scale power grids (e.g., training on 33-bus and testing on 123-bus).

## Target Venues & Deadlines

1. **ICML 2026 Workshops (Seoul, South Korea)**  
   Workshop Proposals Accepted: Mid-March 2026  
   Estimated Paper Submission Deadline: April 24 - May 15, 2026

2. **NeurIPS 2026 Workshops (Sydney, Australia)**  
   Main Conference Deadline: Mid-May 2026  
   Estimated Workshop Paper Deadline: Mid-September 2026

3. **IEEE Transactions on Smart Grid (TSG)**:  
   Submit the extended 8-10 page version of the accepted workshop paper in August/September 2026.

4. **Uncertainty in Artificial Intelligence (UAI workshops)**

5. **MLCloud/MLcon/industry ML events** - Great for demo + application stories (not full‐papers)

## Plan
The work cannot just be "a GNN applied to a grid."  
It must be framed as a fundamental ML contribution that happens to use power systems as its proof of concept.

### Step 1: Shift from Message Passing to Operator Learning
Standard GNNs (baseline Inductive GNNs) learn a discrete mapping from node to node.  
If you add 90 new nodes, the discrete weights fail.  
Implement a Graph Neural Operator (GNO) or Fourier Neural Operator (FNO) that learns the integral operator of the power flow Partial Differential Equation (PDE).  
Because operators learn the continuous function space, they are inherently "grid-resolution invariant."

### Step 2: Prove Zero-Shot Scale Transfer
Reviewers see a hundred papers a day doing 33-bus predictions.  
You need to shock them.  
Train your IP-GNO strictly on your 33-bus dataset.  
Then, evaluate it directly on the IEEE 123-bus and the IEEE 8500-node feeder without fine-tuning a single weight.  
If the operator has truly learned the underlying physics mapping, it will scale.

### Step 3: Inject Physics-Informed Loss (The Safety Net)
Neural Operators can hallucinate physically impossible states.  
Do not just rely on MSE against OpenDSS targets.  
Add a secondary loss term that calculates the active and reactive power mismatches (deltaP, deltaQ) using the predicted voltage and angle.

### Step 4: Energy Profiling
To cement the CS/Systems appeal, incorporate an energy profiling metric.  
Measure the actual computational footprint (Joules or kWh) of running your IP-GNO forward pass against running the traditional OpenDSS Newton-Raphson solver.  
Proving that your model is 100x faster and uses 90% less energy is a massive selling point for "Sustainable AI" workshops.

## NOTE

### Baselines & theory:
Strengthen comparisons with existing neural operator methods (FNO, GNO, DeepONet).  
Demonstrate why they fail without physics loss.

### Ablations:
Show impact of each component (operator vs discrete; with/without physics loss; energy metrics) to isolate contributions.

### Theoretical insights:
Provide at least some proof/intuition why operator learning enables scale invariance theoretically.  
Reviewers appreciate theory more even if experiments are strong.

Your biggest differentiator is NOT that you use a neural operator.  
Your biggest differentiator is: Zero-shot generalization across graph size without retraining.