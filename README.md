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


## Ideas for baseline

- Compare groups with pre-defined reward conditions (A/B testing)
- Compare groups with pre-defined transition matrices (pre-defined beliefs over user activity)
- Post-hoc: looks where it is doing worse
- Bayesian optimisation (obs: the number of steps at the end of the day)
- Look at papers about when to intervene