# MAppAnalysis


Run `scripts/compute_transition_matrices.py` to compute the transition matrices for the different apps. 
The script will create a folder `bkp/compute_transition_matrices` and store the transition matrices there.

Run `notebooks/main.ipynb` to simulate a user activity and train an active-inference agent on top.

- `notebooks/main2.ipynb`: similar to `main`, but the initial beliefs are 
NOT a uniform distribution (random jitter in Dirichlet's alpha between 0 and 1).
- `notebooks/main11.ipynb`: Using hierarchical GMM to model the user activity 
and build the transition matrices.  
- `notebooks/main12.ipynb`: Similar to 12 but began to clean
- `notebooks/main13.ipynb`: Simplified version of 12 