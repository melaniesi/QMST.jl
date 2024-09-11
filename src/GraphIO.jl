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

module GraphIO

using SparseArrays

export readInput_qmstp

"""
    readInput_qmstp(filepath, threshold=100000)

Read input of QMST instances of one of the three formats and return
the number of vertices, the number of edges, a list of Tuples representing the edges
and the cost matrix `Q`.

## Format 1
```
m n
i1 j1
i2 j2
.  .
.  .
.  .
im jm
q11 q12 ... q1m
 .   .       .
 .      .    .
 .         . .
qm1 qm2 ... qmm
```
## Format 2
(m = n(n-1)/2, non-existing edges have weights `threshold`)
```
m n
q11 q12 ... q1m
 .   .       .
 .      .    .
 .         . .
qm1 qm2 ... qmm
```

## Format 3
```
param n := 4 ;
param m := 5 ;
set Edges := (1,2) (1,3) (1,4) (2,4) (3,4);
param c := [1,2] 3 [1,3] 2 [1,4] 1 [2,4] 3 [3,4] 10;
param q := [1,2,1,3] 7 [1,2,1,4] 2 [1,2,2,4] 8 [1,2,3,4] 3 [1,3,1,4] 7 [1,3,2,4] 5 
[1,3,3,4] 1 [1,4,2,4] 5 [1,4,3,4] 8 [2,4,3,4] 4;
```
In this format, the edge costs are given in `c`  and the interaction costs between
two edges is given in `q`.
"""
function readInput_qmstp(filepath, threshold=100000)
    io = open(filepath, "r")
    readline(io)
    secondline = split(readline(io), " ", keepempty=false)
    formatSV = length(secondline) == 2 ? true : false
    formataqmstp = secondline[begin] == "param"
    if formatSV
        return readInput_SVFormat(filepath)
    elseif formataqmstp
        return readInput_aqmstp4qmstp(filepath)
    else
        n, Q_full = readInput_givenCostMatrix(filepath)
        Q_small, edges = getSmallerQAndEdgelist(Q_full, n, threshold)
        m = length(edges)
        return n, m, edges, Q_small
    end
end

"""
    readInput_SVFormat(filepath)

Read the input of QMSTP instances of the format
```
m n
i1 j1
i2 j2
.  .
.  .
.  .
im jm
q11 q12 ... q1m
 .   .       .
 .      .    .
 .         . .
qm1 qm2 ... qmm
```
stored in `filepath` and return the number of vertices, the number of edges,
a list of the edges represented as Tuples of two integers and the cost matrix `Q`.
"""
function readInput_SVFormat(filepath)
    io = open(filepath, "r")
    n, m = parse.(Int64, split(readline(io), " ", keepempty=false))
    edges = Tuple{Int,Int}[]
    for _ in 1:m
        i,j = parse.(Int64, split(readline(io), " ", keepempty=false))
        if i > j
            i, j = j, i
        end
        push!(edges, (i,j))
    end
    Q_small = zeros(Int64, m, m)
    i, j = 1, 1
    while !eof(io)
        entries = parse.(Int64,split(readline(io), " ", keepempty=false))
        nrNewEntries = length(entries)
        if nrNewEntries != 0
            Q_small[i,j:j+nrNewEntries-1] .= entries
        end
        if j + nrNewEntries > m
            i += 1
            j = 1
        else
            j += nrNewEntries
        end
    end
    Q_small = 0.5 * (Q_small + Q_small')
    close(io)
    return n, m, edges, Q_small
end

"""
    readInput_givenCostMatrix(filepath)

Read the input of QMSTP instances of the format
m n
q11 q12 ... q1m
 .   .       .
 .      .    .
 .         . .
qm1 qm2 ... qmm
stored in `filepath` and return `n` and `Q`.
It is assumed that the graph is complete, hence
m = n(n-1)/2.
"""
function readInput_givenCostMatrix(filepath)
    io = open(filepath, "r")
    m, n = parse.(Int64, filter!(x->(!isempty(x) && isdigit(x[1])), split(readline(io), " ")))
    if m < n - 1  # make sure m and n are in correct order!
        m, n = n, m
    end
    nC2 = binomial(n,2) # not sure whether I want to trust m
    Q = zeros(Int64, nC2, nC2)
    i, j = 1, 1
    while !eof(io)
        entries = parse.(Int64,filter!(x->(!isempty(x) && isdigit(x[1])), split(readline(io), " ")))
        nrNewEntries = length(entries)
        if nrNewEntries != 0
            Q[i,j:j+nrNewEntries-1] .= entries
        end
        if j + nrNewEntries > nC2
            i += 1
            j = 1
        else
            j += nrNewEntries
        end
    end
    close(io)
    return (n, Q)
end

"""
    getSmallerQAndEdgelist(Q, n, threshold)

Reduce the quadratic cost matrix by removing all
rows and columns corresponding to edges for which all
quadratic costs with other edges are equal to
`threshold`. Returns the reduces matrix `Q`, and a list
of the remaining edges represented as tuples of vertices
with the first entry being the smaller one.

It is assumed that `Q` is the quadratic cost matrix
of the complete graph, i.e., of dimension `m × m` with
`m = n(n - 1)/2`.
"""
function getSmallerQAndEdgelist(Q, n, threshold)
    nC2 = size(Q,1)
    indices = findall([!all(Q[i,:] .== threshold) for i=1:nC2])
    return (Q[indices,indices], [(i,j) for i = 1:n for j = i+1:n][indices])
end

"""
    readInput_givenEdgesAndCostMatrix(filepath)

Read the input of QMSTP instances of the format
m n
edge_1_i edge_1_j
edge_2_i edge_2_j
    ...
edge_m_i edge_m_j
q11 q12 ... q1m
 .   .       .
 .      .    .
 .         . .
qm1 qm2 ... qmm
stored in `filepath` and return `n`, `Q` and a list 
`edges` of tuples representing the edges.
"""
function readInput_givenEdgesAndCostMatrix(filepath)
    io = open(filepath, "r")
    m, n = parse.(Int64, split(readline(io), " "))
    if m < n  # make sure m and n are in correct order!
        m, n = n, m
    end
    edges = []
    for i = 1:m
        append!(edges, Tuple(parse.(Int64, filter!(x->(!isempty(x) && isdigit(x[1])), split(readline(io), " ")))))
    end
    Q = zeros(Int64, m, m)
    i, j = 1, 1
    while !eof(io)
        entries = parse.(Int64,filter!(x->(!isempty(x) && isdigit(x[1])), split(readline(io), " ")))
        nrNewEntries = length(entries)
        Q[i,j:j+nrNewEntries-1] .= entries
        if j + nrNewEntries > m
            i += 1
            j = 1
        else
            j += nrNewEntries
        end
    end
    close(io)
    return (n, Q, edges)
end

"""
    readInput_aqmstp4qmstp(filepath)

Read the input of AQMSTP instances provided by Pereira at
https://github.com/dilsonpereira/AQMSTP_Instances
and return the number of vertices `n`, the number of edges `m`,
a list of the edges and the quadratic cost matrix `Q` of dimension
`m × m`.

We assume that edges are always given as Tuple
that is edges are always given as (i,j) with i < j
and we assume that Q[e,f] and Q[f,e] are in the file.
"""
function readInput_aqmstp4qmstp(filepath)
    # we assume that edges are always given as Tuple
    # that is edges are always given as (i,j) with i < j
    # we assume that Q[e,f] and Q[f,e] are in the file
    io = open(filepath, "r")
    # read n and m
    n = parse(Int, split(split(readline(io), "=")[2], " ", keepempty=false)[1])
    m = parse(Int, split(split(readline(io), "=")[2], " ", keepempty=false)[1])

    # read edge information
    edges = Tuple{Int64,Int64}[]
    pos_edges = Dict{Tuple{Int64,Int64}, Int64}()
    input = split(split(readline(io), "=")[2], " ", keepempty=false)
    pos = 0
    i = 0
    while input[pos+1] != ";" # end of definition edges is ;
        pos += 1
        i += 1
        push!(edges, eval(Meta.parse(input[pos])))
        pos_edges[eval(Meta.parse(input[pos]))] = i
        if pos == length(input) # read next line from io
            input = split(readline(io), " ", keepempty=false)
            pos = 0
        end
    end
    @assert length(edges) == m

    Q = zeros(m,m)

    # read costs on edges
    input = split(split(readline(io), '=')[2], (' ', '[', ']'), keepempty=false)
    pos = 0
    while input[pos+1] != ";" # end of definition c is ;
        pos_edge = pos_edges[eval(Meta.parse(input[pos+1]))]
        Q[pos_edge,pos_edge] = parse(Int64, input[pos+2])
        pos += 2
        if pos == length(input)
            input = split(readline(io), (' ', '[', ']'), keepempty=false)
            pos = 0
        end
    end

    # read quadratic costs on pairs of adjacent edges
    input = split(split(readline(io), '=')[2], (' ', '[', ']'), keepempty=false)
    pos = 0
    while input[pos+1] != ";"
        edgespair = parse.(Int64,split(input[pos+1],",", keepempty=false))
        i = pos_edges[(edgespair[1],edgespair[2])]
        j = pos_edges[(edgespair[3],edgespair[4])]
        Q[i,j] = parse(Int64, input[pos+2])
        #Q[j,i] = Q[i,j]
        pos += 2
        if pos == length(input)
            input = split(readline(io), (' ', '[', ']'), keepempty=false)
            pos = 0
        end
    end
    close(io)

    @assert Q == Q'
    return n, m, edges, Q
end

"""
    readInput_AQMSTP(filepath)

Read the input of AQMSTP instances provided by Pereira at
https://github.com/dilsonpereira/AQMSTP_Instances
and return the number of vertices `n`,
a dictionary `edges` of the enumerated edges in the graph
and the edge tuple as key, linear costs `c` on the edges
and quadratic costs `Q`.
"""
function readInput_AQMSTP(filepath)
    # we assume that edges are always given as Tuple
    # that is edges are always given as (i,j) with i < j
    # we assume that Q[e,f] and Q[f,e] are in the file

    io = open(filepath, "r")
    # read n and m
    n = parse(Int, split(split(readline(io), "=")[2], " ", keepempty=false)[1])
    m = parse(Int, split(split(readline(io), "=")[2], " ", keepempty=false)[1])

    # read edge information
    edges = Dict{Tuple{Int64,Int64}, Int64}()
    input = split(split(readline(io), "=")[2], " ", keepempty=false)
    pos = 0
    i = 0
    while input[pos+1] != ";"
        pos += 1
        i += 1
        edges[eval(Meta.parse(input[pos]))] = i #push!(edges, eval(Meta.parse(input[pos])))
        if pos == length(input) # read next line
            input = split(readline(io), " ", keepempty=false)
            pos = 0
        end
    end

    # read costs on edges
    c = zeros(Int64, m)
    input = split(split(readline(io), '=')[2], (' ', '[', ']'), keepempty=false)
    pos = 0
    while input[pos+1] != ";"
        c[edges[eval(Meta.parse(input[pos+1]))]] = parse(Int64, input[pos+2])
        pos += 2
        if pos == length(input)
            input = split(readline(io), (' ', '[', ']'), keepempty=false)
            pos = 0
        end
    end

    # read quadratic costs on pairs of adjacent edges
    Q = spzeros(Int64,m,m)
    input = split(split(readline(io), '=')[2], (' ', '[', ']'), keepempty=false)
    pos = 0
    zeroscounter = 0
    while input[pos+1] != ";"
        twoedge = parse.(Int64,split(input[pos+1],",", keepempty=false))
        i = edges[(twoedge[1],twoedge[2])]
        j = edges[(twoedge[3],twoedge[4])]
        Q[i,j] = parse(Int64, input[pos+2])
        if Q[i,j] == 0
            zeroscounter += 1
        end
        #Q[j,i] = Q[i,j]
        pos += 2
        if pos == length(input)
            input = split(readline(io), (' ', '[', ']'), keepempty=false)
            pos = 0
        end
    end
    close(io)
    return (n, edges, c, Q)
end

end # module