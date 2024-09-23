# QMST.jl
Package to compute lower bounds on the quadratic minimum spanning tree problem.
The quadratic minimum spanning tree problem is to find a spanning tree $T$ in a graph $G$ minimizing the cost function
```math
\sum_{e \in E(T)} \sum_{f \in E(T)} Q_{ef},
```
where the cost matrix $Q \in \mathbb{R}^{m \times m}$, modelling the costs of edges and the interaction cost between pairs of edges, is given.


### Installation
To enter the package mode press ```]```.
```julia
pkg> add https://github.com/melaniesi/QMST.jl.git
```
To exit the package mode press ```backspace```.

### Example
```julia
julia> using QMST
julia> using QMST.GraphIO
julia> instancefilepath = "path/to/instance/qmstp_CP10_100_10_10.dat"; # set path instance
julia> n, m, edges, Q = readInput_qmstp(instancefilepath);
julia> params = Parameters(get_param_beta(Q), 0.9, 1, 1e-4, 10800, max_newRLTcuts=m, min_newRLTcuts=10, epsilon_cutviolations=1e-3);
julia> result = run_admm(Q, n, edges, params; trace_constraint=true, PRSM=true, frequ_output=30);
```

Further examples can be found in the folder [`examples/`](examples/) of this project.

### References
This package is part of the publication

Frank de Meijer, Melanie Siebenhofer, Renata Sotirov, Angelika Wiegele. (2024). _Integer Semidefinite Programming for the Quadratic Minimum Spanning Tree problem._ [Manuscript in preparation].