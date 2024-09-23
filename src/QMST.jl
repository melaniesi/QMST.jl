# ==========================================================================
#   This file is part of QMST.jl
# --------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   QMST.jl is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   QMST.jl is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==========================================================================

module QMST

include("GraphIO.jl")
using .GraphIO

using DataStructures
using Dates
using Dictionaries
using Documenter
using Graphs
using HiGHS
using JuMP
using LinearAlgebra
using MKL # BLAS.get_config() to check
using Printf

export Parameters, get_param_beta, run_admm 

intmax = Int(maxintfloat())

"""
    Parameters(beta, gamma1, gamma2, epsilon, max_time=intmax; max_outerloops=15, epsilon_cutviolations=1e-3,
                max_newRLTcuts=500, min_newRLTcuts=30, epsilon_dykstra=epsilon/10, epsilon_lbimprov=1e-3,
                max_iterationstotal=intmax)

Structure containing parameters for the PRSM to compute a lower bound on the QMSTP.

# Fields:
-    `beta::Float64`: penalty parameter of the augmented Lagranian
-    `gamma1::Float64`: stepsize first update, in the range `[-1,1]`
-    `gamma2::Float64`: stepsize second update, in the range `(0, (1 + sqrt(5))/2)`, fulfills `gamma1 + gamma2 > 0`
                        and `|gamma1| < 1 + gamma2 - gamma2^2`.
-    `epsilon::Float64`: epsilon for primal and dual relative error to stop algorithm
-    `max_time::Int`: maximum execution time in seconds
-    `max_outerloops::Int`: maximum number of outer loops
-    `epsilon_violation::Float64`: threshold for RLT cut-set constraints to be considered violated
-    `max_newRLTcuts::Int`: maximum number of new cuts to be added in one outer loop
-    `min_newRLTcuts::Int`: minimum number of new cuts to be added to continue iterating the outer loop (separate new
                            violated constraints)
-    `epsilon_dykstra::Float64`: threshold to stop Dykstra's algorithm when the norm of two consecutive matrices is below that value,
                                 should be smaller than `epsilon`.
-    `epsilon_lbimprov::Float64`: minimum improvement of the valid lower bound of two consecutive outer loops to keep
                                  iterating in the outer loop
-    `max_iterationstotal::Int`:  maximum number of total ADMM iterations
"""
struct Parameters
    beta::Float64
    gamma1::Float64
    gamma2::Float64
    epsilon::Float64
    max_time::Int
    max_outerloops::Int
    epsilon_violation::Float64
    max_newRLTcuts::Int
    min_newRLTcuts::Int
    epsilon_dykstra::Float64
    epsilon_lbimprov::Float64
    max_iterationstotal::Int
    Parameters(beta, gamma1, gamma2, epsilon, max_time=intmax; max_outerloops=15, epsilon_cutviolations=1e-3, max_newRLTcuts=500, min_newRLTcuts=30,
                epsilon_dykstra=epsilon/10, epsilon_lbimprov=1e-3, max_iterationstotal=intmax) = 
            new(beta, gamma1, gamma2, epsilon, max_time, max_outerloops, epsilon_cutviolations, max_newRLTcuts, min_newRLTcuts,
                epsilon_dykstra, epsilon_lbimprov, max_iterationstotal)
end

"""
    get_param_beta(Q)

Compute a good estimate on the penalty parameter β based on the cost matrix `Q`.
"""
function get_param_beta(Q)
    mp1 = size(Q,1)
    normQ = norm(Q)
    trQ = tr(Q)
    tmp = max(normQ,trQ)/min(normQ,trQ)
    guessbeta = tmp < 1.2 ? sqrt(normQ * min(normQ,trQ)/mp1) : sqrt(tmp * norm(Q))
    return guessbeta
end

"""
    run_admm(Q, n::Int, edges, params::Parameters)

Compute a lower bound on the quadratic minimum spanning tree problem using an ADMM algorithm.

# Arguments:
- `Q::Matrix`: cost matrix of the interaction costs between edges in the graph
- `n::Int`: the number of vertices in the graph
- `edges::Vector{Tuple{Int,Int}}`: list of edges in graph
- `params::Parameters`: parameters for the algorithm

# Keyword Arguments:
- `trace_constraint::Bool=true`: Indicate whether the trace constraint is included in \$\\mathcal\{Y\}\$.
- `PRSM=true`: if `PRSM=true`, the PRSM is used and else, the Douglas Rachford algorithm is applied.
- `frequ_output=20`: The number of iterations after which a line of output is printed.
- `ub=Inf`: It is possible to provide an upper bound on the QMSTP.

# Output:
Returns a dictionary with information on the lower bound and the DNN lower bound without cuts and the time needed 
to compute the lower bound, the number of iterations, the matrices Y, R and S.
The (not valid) primal and dual objective values. The number of outer iterations, the solution status,
the number of cuts added, and the number of Dykstra clusters.
"""
function run_admm(Q, n::Int, edges, params::Parameters; trace_constraint::Bool=true, PRSM=true,
                                                        frequ_output=20, ub=Inf)
    mp1 = size(Q, 1)
    m = mp1 - 1

    # parameters
    eps = params.epsilon
    epsilon_dykstra = params.epsilon_dykstra
    gamma1 = params.gamma1
    gamma2 = params.gamma2
    beta = params.beta
    max_iterationstotal = params.max_iterationstotal
    @assert beta > 0
    if PRSM
        @assert -1 < gamma1 < 1 && 0 < gamma2 < (1 + sqrt(5))/2
        @assert gamma1 + gamma2 > 0
        @assert abs(gamma1) < 1 + gamma2 - gamma2^2
    else
        gamma1 = gamma2 = max(gamma1, gamma2)
        @assert 0 < gamma2 < (1 + sqrt(5)) / 2
    end
    # more parameters set here
    max_exectime = params.max_time
    max_outerloops = params.max_outerloops
    min_newRLTcuts = params.min_newRLTcuts
    max_newRLTcuts = params.max_newRLTcuts
    nRLTcuts = 0
    eps_lbimprov = params.epsilon_lbimprov
    eps_violation = params.epsilon_violation
    

    # initialize matrices V, Y, R, S
    V = vcat((n-1) * Matrix(I, m, m), ones(1,m))
    V = Matrix(qr(V).Q) # orthonormalization
    Y, S = initialize_matricesADMM(n, m)
    U_R = d_R = missing  # R = U_R * diagm(d_R) * U_R'
    Yold = copy(Y)
    
    # helper variables
    continue_innerloop = true
    continue_outerloop = true
    counter_outerloops = 0
    counter_iterationstotal = 0
    counter_iterationsinnerloop = 0
    counter_stagnation = 0
    timelimit_reached = false
    output = true
    
    lower_bound = -Inf
    lower_bound_rounded = -Inf
    lower_bound_old = -Inf
    dnn_lower_bound = -Inf
    primal_obj = -Inf
    primal_obj_old = -Inf
    dual_obj = -Inf
    solutionstatus = missing
    dnn_time = 0

    edges_adjtoi = missing
    G = missing
    violated_cuts = missing
    dykstra_clusters = missing
    dykstra_iterations = 0
    

    # ADMM iterations
    println("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓") 
    println("┃    primal         dual   err_p_rel    err_d_rel   iteration    time_elapsed       ┃") 
    println("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛") 

    start_time = now()
    while continue_outerloop
        counter_outerloops += 1
        while continue_innerloop
            counter_iterationstotal += 1
            U_R, d_R = trace_constraint ? projection_PSD_cone_trace(V' * (Symmetric(Y + 1/beta * S) * V), n) : 
                                projection_PSD_cone(V' * (Symmetric(Y + 1/beta * S) * V))
            VU_R = V * U_R
            VRVt = Symmetric(VU_R * diagm(d_R) * VU_R')  #VRVt = V * R * V'
            # if PRSM
            if PRSM
                S += (gamma1 * beta) * (Y - VRVt)
            end
            if counter_outerloops > 1
                Y, dykstra_iterations = projection_polyhedral_withviolatedRLT(VRVt - (1/beta .* (Q + S)), n, edges_adjtoi, dykstra_clusters,
                                                                trace_constraint=trace_constraint, eps_error=epsilon_dykstra,
                                                                dykstra_iterations=dykstra_iterations, output=false)
                Y = Symmetric(Y)
            else
                Y = trace_constraint ? projection_polyhedralY_withtrace(VRVt - (1/beta .* (Q + S)), n) :
                                    projection_polyhedralY!(VRVt - (1/beta .* (Q + S)))
            end
            primal_residual = Y - VRVt
            S += (gamma2 * beta) * primal_residual

            primal_obj = dot(Q, Y)
            if counter_iterationsinnerloop > 100 && abs(primal_obj - primal_obj_old) < 1e-6
                counter_stagnation += 1
            end
            primal_obj_old = primal_obj
            dual_obj = dot(Q, VRVt)
            dual_residual = V' * (Symmetric(Yold - Y) * V) # times beta
            Yold = copy(Y)
            # rel_primal_residual = symm_norm(primal_residual) / (sqrt(mp1) + max(symm_norm(Y), symm_norm(VRVt)))
            rel_primal_residual = symm_norm(primal_residual) / (1 + symm_norm(Y))
            # VtSV = Symmetric(V' * (S * V))
            # rel_dual_residual = beta * symm_norm(Symmetric(dual_residual)) / (sqrt(m) + symm_norm(VtSV))
            rel_dual_residual = beta * symm_norm(Symmetric(dual_residual)) / (1 + symm_norm(Symmetric(S)))
            time_elapsed_s = Dates.value(Millisecond(now() - start_time)) / 10^3 # time elapsed in seconds
            if time_elapsed_s ≥ max_exectime timelimit_reached = true; end
            if (rel_primal_residual < eps && rel_dual_residual < eps) || counter_iterationstotal == max_iterationstotal || timelimit_reached
                continue_innerloop = false  
            end
            
            output = counter_iterationstotal == 1 || counter_iterationstotal % frequ_output == 0 || !continue_innerloop
            if output
                @printf("%11.5f  %11.5f  %10.7f   %10.7f   %8d   %10.2f s\n",
                        primal_obj, dual_obj, rel_primal_residual, rel_dual_residual,
                        counter_iterationstotal, time_elapsed_s)
            end
            
        end           
        # postprocessing after inner loops
        lower_bound_new = compute_safe_lowerbound_new(Q, n, S, V, Symmetric(V' * (S * V)); trace_constraint=trace_constraint,
                                                    violated_cuts=violated_cuts, edges_adjtoi=edges_adjtoi)
        if lower_bound_new > lower_bound
            lower_bound = lower_bound_new
            lower_bound_rounded = Int(ceil(lower_bound - 1e-5))
        end
        if counter_outerloops == 1
            dnn_lower_bound = lower_bound
            dnn_time = Dates.value(Millisecond(now() - start_time)) / 10^3  # time elapsed in seconds
        end 
        # decision what next
        # stop or check for new violated cuts:
        println("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        if counter_iterationstotal == max_iterationstotal
            println("  ⏹ Maximum number of iterations reached.")
            solutionstatus = "MAX_ITS_REACHED"
            @goto final_output
        elseif timelimit_reached
            println("  ⏹ Timelimit reached")
            solutionstatus = "TIMELIMIT_REACHED"
            @goto final_output
        end 

        @printf("  Current lower bound:       %14.5f\n", lower_bound)
        @printf("  Lower bound (rounded):     %8d\n", lower_bound_rounded)       
        if abs(lower_bound - lower_bound_old) < eps_lbimprov || counter_outerloops ≥ max_outerloops || lower_bound_rounded ≥ ub || timelimit_reached
            continue_outerloop = false
            if abs(lower_bound - lower_bound_old) < eps_lbimprov
                solutionstatus = "SLOW_IMPROVEMENT"
                println("  ⏹ Improvement of lower bound too slow.")
            elseif counter_outerloops ≥ max_outerloops
                solutionstatus = "MAX_OUTERLOOPS_REACHED"
                println("  ⏹ Maximum number of outer loops reached.")
            elseif lower_bound_rounded ≥ ub
                solutionstatus = "GAP_CLOSED"
                println("  ⏹ Gap to upper bound closed.")           
            end
        else  # search for violated cuts to add
            println("  → Search for (new) violated cuts.")
            if counter_outerloops == 1
                # initialize for adding violated cuts
                edges_adjtoi = [findall(e -> i in e, edges) for i in 1:n]
                G = SimpleGraph(Edge.(edges))
                violated_cuts = SortedSet{Tuple{Int,Int}}(DataStructures.FasterForward())
            end
            nnewcuts = add_newviolatedcuts!(violated_cuts, max_newRLTcuts, Y, n, edges_adjtoi, eps_viol=eps_violation)
            if nnewcuts < min_newRLTcuts
                continue_outerloop = false
                solutionstatus = "FEW_VIOLATIONS_FOUND"
                println("  ⏹ Found $nnewcuts < $min_newRLTcuts new violated cuts.")
            else
                nRLTcuts = length(violated_cuts)
                dykstra_clusters = get_dykstraclusters(violated_cuts, G, edges)
                println("  ✓ Found $nnewcuts new violated cuts.")
                @printf("  Number of cuts:            %8d (%d new)\n", nRLTcuts, nnewcuts)
                @printf("  Number of clusters:        %8d\n", length(dykstra_clusters))
                println("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
            end
            # reset variables for next run
            lower_bound_old = lower_bound
            counter_iterationsinnerloop = 0
            continue_innerloop = true
        end
    end
    @label final_output
    total_time = Dates.value(Millisecond(now() - start_time)) / 10^3  # time elapsed in seconds
    
    println("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓") 
    @printf("┃ Lower bound:               %14.5f                                         ┃\n", lower_bound)
    @printf("┃ Lower bound (rounded):     %8d                                               ┃\n", lower_bound_rounded)
    @printf("┃ DNN lower bound:           %14.5f                                         ┃\n", dnn_lower_bound)
    @printf("┃ DNN lower bound (rounded): %8d                                               ┃\n", Int(ceil(dnn_lower_bound - 1e-5)))
    @printf("┃ Time:                      %12.3f s                                         ┃\n", total_time)
    @printf("┃ Iterations:                %8d                                               ┃\n", counter_iterationstotal)
    @printf("┃ Stagnations:               %8d                                               ┃\n", counter_stagnation)
    println("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    results = Dict{String, Any}("lb" => lower_bound, "DNN-lb" => dnn_lower_bound, "time-wc" => total_time,
                                "iterations" => counter_iterationstotal)
    results["Y"] = Y; results["R"] = U_R * diagm(d_R) * U_R'; results["S"] = S;
    results["primal-notvalid"] = primal_obj; results["dual-notvalid"] = dual_obj;
    results["solutionstatus"] = solutionstatus;
    results["outer-iterations"] = counter_outerloops;
    results["lb-rounded"] = lower_bound_rounded;
    results["dnn-lb-rounded"] = Int(ceil(dnn_lower_bound - 1e-5));
    results["stagnations"] = counter_stagnation;
    results["ncuts"] = ismissing(violated_cuts) ? 0 : nRLTcuts;
    results["nclusters"] = ismissing(dykstra_clusters) ? 0 : length(dykstra_clusters)
    results["dnn-time"] = dnn_time

    return results
end

"""
    compute_safe_lowerbound_new(Q, n, Sout, V, VtSoutV)

Computes a safe lower bound on the QMSTP based on the output `Sout` from the PRSM algorithm.

# Arguments:
- `Q::Matrix`: the cost matrix of dimension `|E| × |E|`.
- `n::Int`: the number of vertices in the graph
- `Sout::Matrix`: the matrix obtained at the end of the PRSM
- `V::Matrix`: the matrix used for facial reduction
- `VtSoutV`: the matrix `V' * Sout * V`

# Keyword Arguments:
- `trace_constraint=true`: Indicate whether the trace constraint is included in \$\\mathcal\{Y\}\$.
- `violated_cuts=missing`: list of all the violated cuts added in the form of tuples `(f,i) ∈ E × V`.
- `edges_adjtoi::Vector{Vector{Int}}`: vector of length `n` containing at position `i` a list of the edges adjacent
                                        to the vertex `i`.
"""
function compute_safe_lowerbound_new(Q, n, Sout, V, VtSoutV; trace_constraint=true, violated_cuts=missing, edges_adjtoi=missing)
    mp1 = size(Q,1)
    m = mp1 - 1
    model = Model(HiGHS.Optimizer)
    set_attribute(model, "output_flag", false)
    @variable(model, 0 <= Y[1:mp1,1:mp1] <= 1, Symmetric)
    @constraint(model, Y[mp1,mp1] == 1)
    @constraint(model, [e = 1:m], Y[e,e] == Y[e,mp1])
    if trace_constraint
        @constraint(model, tr(Y) == n)
        λmax = eigmax(VtSoutV)
        @objective(model, Min, dot(Q + Sout, Y) - n * λmax)
    else
        ev, W = eigen(Symmetric(VtSoutV))
        ind1 = findfirst(>(0), ev)
        if isnothing(ind1)
            VMVt = zeros(mp1,mp1)
        else
            idx = ind1:length(ev)
            W = W[:,idx]
            VW = V * W
            VMVt = VW * diagm(ev[idx]) * VW'
        end
        @objective(model, Min, dot(Q + Sout - VMVt, Y))
    end
    if !ismissing(violated_cuts)
        if ismissing(edges_adjtoi) @error "there are violated_cuts but edges_adjtoi not provided"; end
        for (f,i) in violated_cuts
            @constraint(model, sum(Y[edges_adjtoi[i],f]) >= Y[f,f])
        end
    end
    optimize!(model)
    return objective_value(model)
end

"""
    symm_norm(A::Symmetric)

Compute the Frobenius norm of a symmetric matrix.
"""
function symm_norm(A::Symmetric)
    return sqrt(dot(A,A))
end

"""
    initialize_matricesADMM(n, m)

Initialize the matrices `Y` and `S` for the PRSM algorithm.

Returns `(Y, S)`.
"""
function initialize_matricesADMM(n, m)
    Y = (n - 1) * (n - 2) / (m * (m - 1)) * ones(m + 1, m + 1)
    entry = (n - 1) / m
    Y[:,end] .= entry
    Y[end,:] .= entry
    Y[diagind(Y)] .= entry
    Y[end,end] = 1
    # R = zeros(m,m)
    S = zeros(m + 1,m + 1)
    return Y, S
end

"""
    add_newviolatedcuts!(violated_cuts::SortedSet, max_newcuts, Y, n, edges_adjtoi)

Separate new violated cuts from `Y`, add them to `violated_cuts` and return the number
of new cuts added.

The `max_newcuts` most violated cuts are added to `violated_cuts` in the form (f, i)
representing the RLT type cut-set constraint \$ \\sum_{e ∈ δ(i)} y_e ≥ y_f \$.

# Arguments:
- `violated_cuts::SortedSet`: sorted set containing tuples of the form (f, i) representing constraints already added
                              to the problem
- `max_newcuts`: the maximum number of new cuts to be added
- `Y`:           matrix of dimension `m × m` used for separation
- `edges_adjtoi::Vector{Vector{Int}}`: contains at position `i` a list of the edges adjacent
                                       to the vertex `i`. 

# Keyword Arguments:
- `eps_viol=1e-4`: threshold of violation for cuts to be considered violated
"""
function add_newviolatedcuts!(violated_cuts::SortedSet, max_newcuts, Y, n, edges_adjtoi; eps_viol=1e-4)
    if max_newcuts ≤ 0 return 0; end
    m = size(Y, 1) - 1
    viol = 0.0
    n_addedcuts = 0
    new_cuts = PriorityQueue{Tuple{Int64,Int64}, Float64}()
    for f in 1:m
        for i in 1:n
            viol =  Y[f,f] - sum(Y[f,edges_adjtoi[i]]) 
            if viol > eps_viol && (f,i) ∉ violated_cuts
                if n_addedcuts < max_newcuts
                    n_addedcuts += 1
                    enqueue!(new_cuts, (f,i), viol)
                elseif peek(new_cuts)[2] < viol
                    dequeue!(new_cuts) # remove cut with least violation
                    enqueue!(new_cuts, (f,i), viol)
                end
            end
        end
    end
    if n_addedcuts > 0
        push!(violated_cuts, collect(keys(new_cuts))...)
    end
    return n_addedcuts
end

"""
    get_lists_violatededgesvertices(violation_tuples::SortedSet)

Separate `violation_tuples` into a list of the edges `f` and the vertices `i` of the cuts in `violation_tuples`.

# Arguments:
- `violation_tuples`: Set of constraints stored in the form of tuples `(f,i) ∈ E × V`.

# Output:
Returns `edges_viol`, `verts_viol` that is two vectors of the same length as the cardinality of `violation_tuples`
and for each index `k`, the tuple `(edges_viol[k], verts_viol[k])` is contained in `violation_tuples`.
"""
function get_lists_violatededgesvertices(violation_tuples::SortedSet)
    # works if violation_tuples is sorted
    edges_viol = Int64[]
    verts_viol = []
    ind = 0
    f_viol_cur = -1
    for (f_viol, v_f_viol) in violation_tuples
        if f_viol != f_viol_cur
            f_viol_cur = f_viol
            ind += 1
            push!(edges_viol, f_viol)
            push!(verts_viol, [v_f_viol])
        else
            push!(verts_viol[ind], v_f_viol)
        end
    end
    return edges_viol, verts_viol
end

"""
    get_dykstraclusters(violation_tuples, G::Graph, edges)

Cluster the cuts in `violation_tuples` such that in each cluster the vertices in the same column `f` form an independent set.

# Arguments:
- `violation_tuples::SortedSet`: Set of constraints stored in the form of tuples `(f,i) ∈ E × V`.
- `G::Graph`: the underlying graph of the QMSTP
- `edges`: list of edges (stored as tuple of two vertices) in `G`

# Output:
- `dykstra_clusters::Vector{Vector{Tuple{Int64, Vector{Int64}}}}`: vector containing in each position `k` one
                    dykstra cluster as list of Tuples `(f, vertices)`, where `f` denotes the row/column of the violated cuts
                    and `vertices` is an independet set of vertices in the underlying graph representing for each vertex `i` in
                    `vertices`, the RLT type cut-set constraint (i,f), that is `sum(y_\{e,f\} for e ∈ δ(i)) ≥ y_f`,.
                    For each edge `f` there is at most one entry in `dykstra_clusters[k]`.
"""
function get_dykstraclusters(violation_tuples, G::Graph, edges)
    edges_viol, verts_viol = get_lists_violatededgesvertices(violation_tuples)
    dykstra_clusters = Vector{Vector{Tuple{Int64, Vector{Int64}}}}()
    dykstra_its = 0
    for (f_viol, verts_f_viol) in zip(edges_viol, verts_viol)
        indep_sets_f_viol = undef
        if length(verts_f_viol) == 1
            indep_sets_f_viol = [[verts_f_viol[1]]]
        elseif length(verts_f_viol) == 2
            indep_sets_f_viol = any(e->e in edges, Tuple.([verts_f_viol, reverse(verts_f_viol)])) ? 
                                    [[verts_f_viol[1]], [verts_f_viol[2]]] : [[verts_f_viol[1], verts_f_viol[2]]]
        else
            G_ind, v_map = induced_subgraph(G, verts_f_viol)
            coloring = Graphs.Parallel.greedy_color(G_ind)
            indep_sets_f_viol = [v_map[findall(==(c), coloring.colors)] for c in 1:coloring.num_colors]
        end
        for ind in 1:min(dykstra_its, length(indep_sets_f_viol))
            push!(dykstra_clusters[ind], (f_viol, indep_sets_f_viol[ind]))
        end
        for ind in (dykstra_its + 1):length(indep_sets_f_viol)
            push!(dykstra_clusters, [(f_viol, indep_sets_f_viol[ind])])
            dykstra_its += 1
        end
    end
    return dykstra_clusters
end



"""
    projection_RLT_col!(a, m, pos_f, pos_edges_partition_k, normal_mat_col)

Project the column `a` onto the polyhedral set \$ T_f^k \$.

The polyhedral set is defined as
\$ \\{ y ∈ \\mathbb{R}^{m + 2} : y_f = y_{m+1} = y_{m+2}, \\sum_{e ∈ δ(i)} y_e ≥ y_f ∀ i ∈ K_k \\} \$

The projection is stored in `a` and the normal vector in `normal_mat_col`.
The normal vector equals the original `a` minus its projection onto \$ T_f^k \$.

# Arguments:
- `a`: the vector of size `m + 1` instead of `m + 1` (`a[m + 1] = 1/2 * (a[m + 1] + a[m + 2])` is assumed to hold)
- `m`: the number of edges
- `pos_f`: the position of edge f in `a`
- `pos_edges_partition_k`: a list containing for each vertex \$i ∈ K_k\$  a list with positions of the edges in δ(i)
- `normal_mat_col`: vector of dimension `m + 1`
"""
function projection_RLT_col!(a, m, pos_f, pos_edges_adjtoverts, normalmat_col_f)
    #normal_mat = zeros(Float64, size(a))
    af_avg = 1/3 * (a[pos_f] + 2*a[m+1])

    pq = PriorityQueue{Int, Float64}(DataStructures.FasterReverse())
    for (i, incident_toi) in enumerate(pos_edges_adjtoverts)
        if pos_f in incident_toi       # check whether f ∈ δ(i)
            c = sum(a[incident_toi]) - a[pos_f]
            if c < 0
                indices = filter(!=(pos_f), incident_toi)
                entry = c/(length(incident_toi) - 1)
                normalmat_col_f[indices] .= entry
                a[indices] .-= entry
            end
        else
            enqueue!(pq, i, af_avg - sum(a[incident_toi]))
        end
    end
    ωi_n = 0
    ωi_d = 3
    Kstar_vals = Vector{Tuple{Int64,Float64,Int64}}()
    while length(pq) > 0
        i, gi = dequeue_pair!(pq)
        di = length(pos_edges_adjtoverts[i])
        ωi_n += gi/di
        ωi_d += 1/di
        if gi > ωi_n/ωi_d
            push!(Kstar_vals, (i,gi,di))
        else
            ωi_n -= gi/di
            ωi_d -= 1/di
            break
        end
    end
    ωi = ωi_n/ωi_d
    for (i,gi,di) in Kstar_vals
        entry = (ωi - gi)/di
        normalmat_col_f[pos_edges_adjtoverts[i]] .= entry
        a[pos_edges_adjtoverts[i]] .-= entry
    end
    normalmat_col_f[pos_f] = a[pos_f] - af_avg + ωi
    normalmat_col_f[m+1] = a[m+1] - af_avg + ωi
    a[pos_f] = a[m+1] = af_avg - ωi
end

"""
    projection_RLT_cluster_k_part!(M, edges_adjtoi, dykstra_clusterk_chunk, normal_mat)

Project a subset of columns of matrix `M` onto \$\\mathcal\{Y\}_\{\\mathcal\{C\}_k\} \$.

The projection `X` of `M` gets stored in the matrix `M`.
The normal matrix, that is `M - X`, is stored in `normal_mat`.

# Arguments:
- `edges_adjtoi::Vector{Vector{Int}}`: contains at position `i` a list of the edges adjacent
                                        to the vertex `i`.
- `dykstra_clusters_chunk::Vector{Tuple{Int64, Vector{Int64}}}`: contains a subset of
                            `dykstra_clusters_k` which represents \$\\mathcal\{C\}_k \$.
                            The vector `dykstra_clusters_chunk` is a list of Tuples `(f, vertices)`, where `f` denotes the column 
                            and `vertices` is an independet set of vertices in the underlying graph representing for each vertex `i` in
                            `vertices`, the RLT type cut-set constraint (i,f), that is `sum(y_\{e,f\} for e ∈ δ(i)) ≥ y_f`.
                            For each edge `f` there is at most one entry in `dykstra_clusters_chunk`.
"""
function projection_RLT_cluster_k_part!(M, edges_adjtoi, dykstra_clusterk_chunk, normal_mat)
    mp1 = size(M, 1)
    for (f, verts_f_viol) in dykstra_clusterk_chunk
        pos_edges_adjtoverts = [edges_adjtoi[i] for i in verts_f_viol]
        Mff = M[f,f]
        Mfmp1 = M[f,mp1]
        Mmp1f = M[mp1,f]

        M[mp1,f] = 1/2 * (M[f,mp1] + M[mp1,f])
        projection_RLT_col!(view(M,:,f), (mp1 -1), f, pos_edges_adjtoverts, view(normal_mat,:,f))
        M[f,mp1] = M[mp1,f]

        normal_mat[f,f] = Mff - M[f,f]
        normal_mat[mp1,f] = Mmp1f - M[mp1,f]
        normal_mat[f,mp1] = Mfmp1 - M[f,mp1]
    end
end

"""
    projection_RLT_cluster_k!(M, edges_adjtoi, dykstra_cluster_k, normal_mat)

Project matrix `M` onto \$\\mathcal\{Y\}_\{\\mathcal\{C\}_k\} \$.

The projection `X` of `M` gets stored in the matrix `M`.
The normal matrix, that is `M - X`, is stored in `normal_mat`.
The projection is multithreaded over the columns. 

# Arguments:
- `edges_adjtoi::Vector{Vector{Int}}`: vector that contains at position `i` a list of the edges adjacent
                                       to the vertex `i`.
- `dykstra_clusters_k::Vector{Tuple{Int64, Vector{Int64}}}`: vector that represents \$\\mathcal\{C\}_k \$
                and is a list of Tuples `(f, vertices)`, where `f` denotes the row/column of the violated cuts 
                and `vertices` is an independet set of vertices in the underlying graph representing for each vertex `i` in
                `vertices`, the RLT type cut-set constraint (i,f), that is `sum(y_\{e,f\} for e ∈ δ(i)) ≥ y_f`.
                For each edge `f` there is at most one entry in `dykstra_clusters`.
"""
function projection_RLT_cluster_k!(M, edges_adjtoi, dykstra_cluster_k, normal_mat)
    mp1 = size(M, 1)

    fill!(normal_mat, 0)
    normal_mat[mp1,mp1] = M[mp1,mp1] - 1
    M[mp1,mp1] = 1

    # parallelize: split up dykstra_cluster_k
    dykstra_k_chunks = Iterators.partition(dykstra_cluster_k, Int(ceil(length(dykstra_cluster_k) / (Threads.nthreads() ))))
    # do the following part then in parallel
    threadhandles = []
    for chunk in dykstra_k_chunks
        # put this part in a function to call with Threads.@spawn
        t = Threads.@spawn projection_RLT_cluster_k_part!(M, edges_adjtoi, chunk, normal_mat)
        push!(threadhandles, t)
    end
    for t in threadhandles
        wait(t)
    end
end

"""
    projection_polyhedral_withviolatedRLT(M, n, edges_adjtoi, dykstra_clusters)

Project `M` onto \$\\mathcal\{Y\}_\{RLT\} \$.

# Arguments
- `edges_adjtoi::Vector{Vector{Int}}`: vector is of length `n` and containing at position `i` a list of the edges adjacent
                                        to the vertex `i`.
- `dykstra_clusters::Vector{Vector{Tuple{Int64, Vector{Int64}}}}`: vector containing in each position `k` one
                    dykstra cluster as list of Tuples `(f, vertices)`, where `f` denotes the row/column of the violated cuts
                    and `vertices` is an independet set of vertices in the underlying graph representing for each vertex `i` in
                    `vertices`, the RLT type cut-set constraint (i,f), that is `sum(y_\{e,f\} for e ∈ δ(i)) ≥ y_f`,.
                    For each edge `f` there is at most one entry in `dykstra_clusters[k]`.

## Keyword arguments:
- `eps_error=1e-8`: Algorithm stops as soon as `norm(Xold - X) < eps_error`.
- `trace_constraint::Bool=true`: Indicate whether the trace constraint is included in \$\\mathcal\{Y\}\$.
- `dykstra_iterations=0`: The norm of `Xold - X` gets checked after `0.85 * dykstra_iterations` iterations.
- `output=false`: If `output=true`, the number of Dykstra iterations is printed at the end.

## Output:
Returns the projection `X` of `M` onto \$\\mathcal\{Y\}_\{RLT\} \$ and the number of Dykstra iterations needed.
"""
function projection_polyhedral_withviolatedRLT(M, n, edges_adjtoi, dykstra_clusters; eps_error=1e-8, trace_constraint::Bool=true,
                                        dykstra_iterations=0, output=false)
    X = copy(Matrix(M))
    Nmatrices = Dictionary{Int64, Matrix}(1:length(dykstra_clusters),
                                            [zeros(Float64,size(M)) for i in 1:length(dykstra_clusters)])
    N = zeros(size(M))
    diff = 1 + eps_error
    threshold_normcomputation = floor(0.85 * dykstra_iterations)
    ct = 0
    while diff > eps_error
        X_old = copy(X)
        ct += 1
        X += N # X = X + 1 * N
        if trace_constraint
            X = projection_polyhedralY_withtrace(X, n)
        else
            X = projection_polyhedralY(X)
        end
        N += (X_old - X)
        # diff = norm(X_old - X)^2
        for (ind, dykstra_cluster_k) in enumerate(dykstra_clusters)
            X += Nmatrices[ind] # X = X + Nmatrices[ind]
            projection_RLT_cluster_k!(X, edges_adjtoi, dykstra_cluster_k, Nmatrices[ind])
            # diff += norm(Nmatrices[ind] - Nmold)^2
        end
        
        if ct > threshold_normcomputation diff = norm(X_old - X); end
    end
    if output println("Number of dykstra iterations: $ct, threshold was: $threshold_normcomputation"); end
    return (X, ct)
end


"""
    projection_cappedsimplex(y, k)

Return the projection x onto the capped simplex,
that is
\$\\arg\\min \\lVert x - y \\rVert s.t.: e^\\top x = k\$

Alorithm of
    Weiran Wang and Canyi Lu,
    Projection onto the Capped Simplex,
    arXiv preprint arXiv:1503.01002, 2015
    
Translation of projection.m from
https://github.com/canyilu/Projection-onto-the-capped-simplex
"""
function projection_cappedsimplex(y, k)
    n = length(y)
    x = zeros(Float64, n)
    if k < 0 || k > n throw(DomainError(k, "argument must be between 0 and length(y)")) end
    if k == 0 return x end
    if k == n return ones(Float64, n) end
    idx = sortperm(y)
    ys = sort(y)

    if k == round(k) # if k is integer
        b = Int(n - k)
        if ys[b+1] - ys[b] >= 1
            x[idx[b+1:end]] .= 1
            return x
        end
    end

    # assume a=0
    s = cumsum(ys)
    ys = vcat(ys, Inf)
    for b = 1:n
        gamma = (k + b - n - s[b]) / b # hypothesized gamma
        if ys[1] + gamma > 0 && ys[b] + gamma < 1 && ys[b+1] + gamma ≥ 1
            x[idx] = vcat(ys[1:b] .+ gamma, ones(n-b))
            return x
        end
    end

    # assume a ≥ 1
    for a = 1:n
        for b = a+1:n
            if b == a
                @warn "Projection capped simplex: reached problematic case! b = a."
            end
            # hypothesized gamma
            gamma = (k + b - n + s[a] - s[b]) / (b - a)
            if ys[a] + gamma ≤ 0 && ys[a+1] + gamma > 0 && ys[b] + gamma < 1 && ys[b+1] + gamma ≥ 1
                x[idx] = vcat(zeros(a), ys[a+1:b] .+ gamma, ones(n-b))
                return x
            end
        end
    end
    @warn "Problem projecting onto capped simplex, did not find root. Use approximate algorithm."
    return projection_cappedsimplex_approx(y, k)
end


"""
    projection_cappedsimplex_approx(y, k)

Return the approximate projection x onto the capped simplex,
that is
\$\\arg\\min \\lVert x - y \\rVert s.t.: e^\\top x = k\$
with an approximation error of `|sum(x)-k| < eps`.
The solution is guaranteed to be bounded by 0 and 1.

Implementation of Algorithm 1 of
    Andersen Ang, Jianzhu Ma, Nianjun Liu, Kun Huang, Yijie Wang
    Fast Projection onto the Capped Simplex with Applications to
    Sparse Regression in Bioinformatics,
    Advances in Neural Information Processing Systems 34 (NeurIPS 2021) 
"""
function projection_cappedsimplex_approx(y, k, eps=1e-12)
    n = length(y)
    if k == 0
        return zeros(n)
    elseif k == n
        return ones(n)
    elseif k < 0 || k > n
        throw(DomainError(k, "argument must be between 0 and length(y)"))
    end
    gamma = minimum(y) - 0.5
    v = y .- gamma
    w1 = k - sum(v)
    ct = 0
    while abs(w1) > eps
        ct += 1
        w1 = k
        w2 = 0
        for i=1:n
            if v[i] > 0
                if v[i] < 1
                    w1 -= v[i]
                    w2 += 1
                else
                    w1 -= 1
                end
            end
        end
        if w2 == 0
            @warn "k probably too small/large for algorithm, w''(γ) = 0"
            return projection_cappedsimplex(y, k)
        else
            v .+= w1/w2
        end
    end
    for i=1:n
        if v[i] < 0
            v[i] = 0
        elseif v[i] > 1
            v[i] = 1
        end
    end
    return v
end

"""
    projection_polyhedralY_withtrace(M, n)

Compute the projection of a symmetric matrix `M`
onto the polyhedral set \$\\mathcal{Y}\$.

In more detail, \$\\mathcal{Y}\$ is the set of symmetric matrices
with entries between 0 and 1, the diagonal of the matrix equals the
last column and the bottom right entry equals 1 and trace = n.
"""
function projection_polyhedralY_withtrace(M, n)
    X = 1/2 * (M + M')
    mp1 = size(M, 1)
    m = mp1 - 1
    X[end,end] = 1
    v = 2/3 * X[1:m,end] + 1/3 * diag(X[1:m,1:m])
    v_proj = projection_cappedsimplex(v, (n - 1))
    for j = 1:m
        for i = 1:(j - 1)
            if X[i,j] < 0
                X[i,j] = 0
            elseif X[i,j] > 1
                X[i,j] = 1
            end
        end
        X[j,end] = v_proj[j]
        X[j,j] = v_proj[j]
    end
    return Symmetric(X)
end


"""
    projection_polyhedralY(M)

Compute the projection of a symmetric matrix `M`
onto the polyhedral set \$\\mathcal{Y}\$.

In more detail, \$\\mathcal{Y}\$ is the set of symmetric matrices
with entries between 0 and 1, the diagonal of the matrix equals the
last column and the bottom right entry equals 1.
"""
function projection_polyhedralY(M)
    Y = 1/2 * (M + M')
    mp1 = size(Y, 1)
    for i = 1:(mp1 - 1)
        Y[i,mp1] = 2/3 * Y[i,mp1] + 1/3 * Y[i,i]
        Y[i,i] = Y[i,mp1]
    end
    Y[mp1,mp1] = 1
    projection_box!(Y)
    return Symmetric(Y)
end


"""
    projection_polyhedralY!(M)

Compute the projection of a symmetric matrix `M`
onto the polyhedral set \$\\mathcal{Y}\$.

In more detail, \$\\mathcal{Y}\$ is the set of symmetric matrices
with entries between 0 and 1, the diagonal of the matrix equals the
last column and the bottom right entry equals 1.

## Output
The upper triangular part of the matrix `M` after the function call
is equal to the projection of `M` onto \$\\mathcal{Y}\$.
The matrix `Symmetric(M)` is returned.
"""
function projection_polyhedralY!(M)
    mp1 = size(M, 1)
    for i = 1:(mp1 - 1)
        M[i,mp1] = 2/3 * M[i,mp1] + 1/3 * M[i,i]
        M[i,i] = M[i,mp1]
    end
    M[mp1,mp1] = 1
    projection_box!(M)
    return Symmetric(M)
end



"""
    projection_box!(M)

Project each entry of the matrix `M` onto the box [0,1].
"""
function projection_box!(M)
    mp1 = size(M, 1)
    for j = 1:mp1
        for i = 1:j
            if M[i,j] < 0
                M[i,j] = 0
            elseif M[i,j] > 1
                M[i,j] = 1
            end
        end
    end
end


"""
    projection_PSD_cone(M)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices.

## Output
Returns `U::Matrix` and `v::Vector` such that
the projection equals `U * diagm(v) * U'`.
"""
function projection_PSD_cone(M)
    ev, U = try
        eigen(Symmetric(M))
    catch e
        println(e)
        eigen(Symmetric(Float32.(M))) # cf.: https://github.com/JuliaLang/julia/issues/40260#issuecomment-812303318
    end
    ind1 = findfirst(>(0), ev)
    if isnothing(ind1)
        return zeros(size(M,1),1), [0]
        return zeros(size(M))
    end
    idx = ind1:length(ev)
    U = U[:,idx]
    return U, ev[idx] # projection = Symmetric(U * diagm(ev[idx]) * U')
end


"""
    projection_PSD_cone_trace(M, n)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices with trace equal to `α`.

## Output
Returns `U::Matrix` and `v::Vector` such that
the projection equals `U * diagm(v) * U'`.
"""
function projection_PSD_cone_trace(M, α)
    ev, U = try
        eigen(Symmetric(M))
    catch e
        println(e)
        eigen(Symmetric(Float32.(M))) # cf.: https://github.com/JuliaLang/julia/issues/40260#issuecomment-812303318
    end

    # project vector of eigenvalues onto the
    # simplex Δ_α
    # do not set entries who change to 0
    # (that is the smallest entries 1:(index - 1))
    cum_sum = 0
    index = length(ev)
    for k = 1:length(ev)
        if cum_sum + ev[index] - α ≥ k * ev[index]
            index += 1
            ev[index:end] .-= (cum_sum - α) / (k - 1)            
            break
        else
            cum_sum += ev[index]
            index -= 1
        end
    end
    if index == 0
        index = 1
        ev[1:end] .-= (cum_sum - α) / length(ev)
    end

    idx = index:length(ev)
    U = U[:,idx]
    return U, ev[idx]   # projection = Symmetric(U * diagm(ev[idx]) * U')
end


end # module
