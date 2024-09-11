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

using Gurobi
using JuMP
using LinearAlgebra
using Random
using Test

@testset "QMST.jl" begin

    @testset "projection_rlt_col" begin
        function projection_RLT_col_gurobi(a, m, pos_f, pos_edges_partition_k)
            a_dim_mp2 = copy(a)
            append!(a_dim_mp2, a[m+1])
            model = Model(Gurobi.Optimizer)
            yvar = @variable(model, yvar[1:(m + 2)])
            @constraint(model, yvar[pos_f] == yvar[m+1])
            @constraint(model, yvar[m+1] == yvar[m+2])
            for adj_toi in pos_edges_partition_k
                @constraint(model, sum(yvar[adj_toi]) >= yvar[pos_f])
            end
            @objective(model, Min, dot(yvar-a_dim_mp2,yvar-a_dim_mp2))
            optimize!(model)
            return value.(yvar)[1:m+1]
        end

        m = 30
        indices = collect(1:m)
        shuffle!(indices)
        pos_edges1 = [indices[1:3], indices[4:6], indices[7:8], indices[9:17]]
        pos_f = indices[18]
        pos_edges2 = deepcopy(pos_edges1)
        append!(pos_edges2[2], pos_f)

        a = 1 .- 2.5 * rand(m+1)
        ac1 = copy(a)
        ac2 = copy(a)

        nmc1 = zeros(size(a))
        nmc2 = zeros(size(a))
        QMST.projection_RLT_col!(ac1, m, pos_f, pos_edges1, nmc1)
        QMST.projection_RLT_col!(ac2, m, pos_f, pos_edges2, nmc2)

        @test norm(ac1 - projection_RLT_col_gurobi(a, m, pos_f, pos_edges1)) < 1e-5
        @test norm(ac2 - projection_RLT_col_gurobi(a, m, pos_f, pos_edges2)) < 1e-5
        @test norm(a - ac1 - nmc1) < 1e-5
        @test norm(a - ac2 - nmc2) < 1e-5

        a[pos_edges1[1][1]] += a[pos_f]
        ac1 = copy(a)
        ac2 = copy(a)

        nmc1 = zeros(size(a))
        QMST.projection_RLT_col!(ac1, m, pos_f, pos_edges1, nmc1)
        nmc2 = zeros(size(a))
        QMST.projection_RLT_col!(ac2, m, pos_f, pos_edges2, nmc2)
        @test norm(ac1 - projection_RLT_col_gurobi(a, m, pos_f, pos_edges1)) < 1e-5
        @test norm(ac2 - projection_RLT_col_gurobi(a, m, pos_f, pos_edges2)) < 1e-5
        @test norm(a - ac1 - nmc1) < 1e-5
        @test norm(a - ac2 - nmc2) < 1e-5
    end

    @testset "projection-polyhedralY" begin
        function projection_polyhedralY_gurobi(M)
            m = size(M,1) - 1
            model = Model(Gurobi.Optimizer)
            Y = @variable(model, 0 <= Y[1:(m + 1),1:(m + 1)] <= 1, Symmetric)
            @constraint(model, [i = 1:m], Y[i,i] == Y[i,end])
            @constraint(model, Y[end,end] == 1)
            @objective(model, Min, dot(Y-M,Y-M))
            optimize!(model)
            return value.(Y)
        end

        function projection_polyhedralY_gurobi_withtrace(M, n)
            m = size(M,1) - 1
            model = Model(Gurobi.Optimizer)
            Y = @variable(model, 0 <= Y[1:(m + 1),1:(m + 1)] <= 1, Symmetric)
            @constraint(model, [i = 1:m], Y[i,i] == Y[i,end])
            @constraint(model, Y[end,end] == 1)
            @constraint(model, tr(Y) == n)
            @objective(model, Min, dot(Y-M,Y-M))
            optimize!(model)
            return value.(Y)
        end

        m = 30
        M = Symmetric(1 .- 2.5 * rand(m+1,m+1))
        @test norm(QMST.projection_polyhedralY(Matrix(M)) - projection_polyhedralY_gurobi(M)) < 1e-4

        n = 8
        @test norm(QMST.projection_polyhedralY_withtrace(Matrix(M), n) - projection_polyhedralY_gurobi_withtrace(M, n)) < 1e-4

    end


    @testset "projection-capped-simplex" begin
        y = rand(100)
        @test sum(QMST.projection_cappedsimplex(y, 0)) == 0
        @test sum(QMST.projection_cappedsimplex(y, 100)) == 100

        y = []

        @test QMST.projection_cappedsimplex([3, 2, 1, -10, -2, -3], 4) ==
        [1.0, 1.0, 1.0, 0, 1.0, 0]

        y = [2, 0, 0, 0, 0, 1, 2, 2, 1, 0]
        xopt = [1, 0.8, 0.8, 0.8, 0.8, 1, 1, 1, 1, 0.8]
        x = QMST.projection_cappedsimplex(y, 9)
        @test norm(x - xopt) < 1e-10

        y = [1,1,1,1,0,0,0,1]
        xopt = [0.8, 0.8, 0.8, 0.8, 0, 0, 0, 0.8]
        x = QMST.projection_cappedsimplex(y, 4)
        @test norm(x - xopt) < 1e-10

        y = rand(400)
        x1 = QMST.projection_cappedsimplex(y, 111)
        x2 = QMST.projection_cappedsimplex_approx(y, 111, 1e-12)
        @test norm(x1 - x2) < 1e-8
    end

end
