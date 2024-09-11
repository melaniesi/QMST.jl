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

instancefilepath = "../../04_Data/QMSTPInstances/CP3/qmstp_CP10_100_100_10.dat"
n, m, edges, Q = readInput_qmstp(instancefilepath);
params = Parameters(get_param_beta(Q), 0.9, 1, 1e-4, 10800, max_newRLTcuts=m, min_newRLTcuts=10, epsilon_cutviolations=1e-3);
result = run_admm(Q, n, edges, params; trace_constraint=true, sPRSM=true, frequ_output=30)
