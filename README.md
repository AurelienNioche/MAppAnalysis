# MAppAnalysis


Run `scripts/compute_transition_matrices.py` to compute the transition matrices for the different apps. 
The script will create a folder `bkp/compute_transition_matrices` and store the transition matrices there.

Run `notebooks/main.ipynb` to simulate a user activity and train an active-inference agent on top.

The difference between `notebooks/main.ipynb` and `notebooks/main2.ipynb` is that in the former, the initial beliefs are uniform distribution, while not in the second (random jitter in Dirichlet's alpha between 0 and 1).