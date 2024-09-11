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

using QMST
using QMST.GraphIO

using JSON

# PROVIDE PATHS TO DIRECTORIES CONTAINING INSTANCES HERE
# run code first for a small instance to exclude compile time from runtime numerics
paths0 = ["path/to/QMSTPInstances/test_small/"] 
paths1 = ["path/to/QMSTPInstances/CP$(i)/" for i in [1, 2, 4, 3]]
paths2 = ["path/to/QMSTPInstances/SV/"]
paths3 = ["path/to/QMSTPInstances/OP"*x*"sym/" for x in ["", "e", "v"]]


function get_params(Q, beta_fun)
    m = size(Q,1) - 1
    beta = beta_fun(Q)
    gamma1 = 0.9
    gamma2 = 1
    eps = 1e-4
    epsdykstra = 1e-5
    params = Parameters(beta, gamma1, gamma2, eps, 10800,
                        max_iterationstotal=30000, max_outerloops=15,
                        epsilon_cutviolations=1e-3, max_newRLTcuts=m, min_newRLTcuts=10, #Int(floor(size(Q,1)/20)),
                        epsilon_dykstra=epsdykstra, epsilon_lbimprov=1e-3
                        ) # β, γ1, γ2, ε, maxtime
    return params
end

function evaluate_admm(filepath, writelogfile=true, movetoprocessed=true)
    n, m, edges, Q = readInput_qmstp(filepath)
    Q = hcat(vcat(Q, zeros(1, m)), zeros(m+1,1))
    params = get_params(Q, get_param_beta)
    result = run_admm(Q, n, edges, params; trace_constraint=true, sPRSM=true, frequ_output=100)
    if writelogfile || movetoprocessed
        filename = split(filepath, "/")[end]
        instancename = split(filename, ".", keepempty=false)[begin]
        pathdir = split(filepath, instancename)[begin]
        if writelogfile
            if !isdir(pathdir*"logs") mkdir(pathdir*"logs"); end
            io = open(pathdir * "logs/" * instancename* ".json", "w")
            JSON.print(io, result, 4)
            close(io)
        end
        if movetoprocessed
            if !isdir(pathdir*"processed") mkdir(pathdir*"processed"); end
            mv(filepath, pathdir*"processed/"*filename, force=true)
        end
    end
    return result
end


function evaluate_all(paths)
    for pathdir in paths
        evaluate_allindir(pathdir)
    end
end

function evaluate_allindir(pathdir)
    graphFiles = filter(x->(endswith(x,r".dat|txt")), readdir(pathdir, sort=false))
    perm = get_sortperm_graphFiles(graphFiles, pathdir)
    for graphfile in graphFiles[perm]
        evaluate_admm(pathdir*graphfile, true, true)
    end
end

function get_sortperm_graphFiles(graphFiles, path_dir, threshold=100000)
    nedges = similar(graphFiles, Int64)
    for (i, filename) in enumerate(graphFiles)
        filepath = path_dir * filename
        _, m, _, _ = readInput_qmstp(filepath, threshold)
        nedges[i] = m
    end
    return sortperm(nedges)
end

# EVALUATE
for paths in [paths0, paths1, paths2, paths3]
    evaluate_all(paths)
end


# WRITE A SUMMARY OF ALL RESULTS
for paths in [paths0, paths1, paths2, paths3]
    for path in paths
        if !isdir(path*"logs/") continue; end # no log files in this directory - go to next path
        jsonFiles = filter(x->endswith(x, ".json"), readdir(path*"logs/",sort=false))
        foldername = split(path, "/", keepempty=false)[end]
        io = open(path*"logs/summary-$foldername.csv", "w")
        write(io,"instance,n,m,dnn,dnnrounded,dnntime,lb,lbrounded,time,ncuts,nclusters,iterations,outerloops,solutionstatus\n")
        for instance in jsonFiles
            println(instance)
            instancename = split(instance, ".")[begin]
            instancepath_nofileending = path * "processed/" * instancename
            instancepath = isfile(instancepath_nofileending*".dat") ? instancepath_nofileending*".dat" : instancepath_nofileending*".txt"
            nverts, nedges, _, _ = readInput_qmstp(instancepath)
            
            results = JSON.parsefile(path*"logs/"*instance)

            lower_bound = results["lb"]
            dnn_lower_bound = results["DNN-lb"]
            dnn_time = results["dnn-time"]
            total_time = results["time-wc"]
            iterations = results["iterations"]
            primal_ob = results["primal-notvalid"] 
            dual_obj = results["dual-notvalid"]
            solutionstatus = results["solutionstatus"]
            outerloops = results["outer-iterations"]
            lb_rounded = results["lb-rounded"]
            dnn_rounded = results["dnn-lb-rounded"]
            stagnations = results["stagnations"]
            ncuts = results["ncuts"]
            nclusters = results["nclusters"]

            write(io, "$instancename,$nverts,$nedges,$dnn_lower_bound,$dnn_rounded,$dnn_time,$lower_bound,$lb_rounded,$total_time,$ncuts,$nclusters,$iterations,$outerloops,$solutionstatus\n")
        end
        close(io)
    end
end
