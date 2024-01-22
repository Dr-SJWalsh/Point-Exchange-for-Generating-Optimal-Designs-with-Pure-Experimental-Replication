## ============================================================================================
#
#       Point Exchange (PEXCH) Algorithm for generating optimal designs without and with
#         user specified replication structure
#
#       Author: Stephen J. Walsh
#               Utah State University
#
#       January 19, 2024
#
#       Supplementary to: Walsh, Bolton, Borkowski (2024) Generating Optimal Designs with
#                          User Specified Replication Structure
#
## ============================================================================================

## ============================================================================================
## change the working directory
print(pwd())
cd("C:/Users/Dr. Stephen J. Walsh/OneDrive - USU/Desktop/julia_hypercube_v0.0")

## required Julia packages
using Distributions, LinearAlgebra, Random,  DataFrames, CSV, Plots

## --------------------------------------------------------------------------------------------

## ============================================================================================
## Build the discrete candidate set for K = 2 designs
#  Currently utilizing a 31^2 factorial, i.e. 961 candidate points

# For K = 2, 31pts in one D gives a grid with 961 pts
Nl  = 31
x1  = collect(range(-1, stop = 1, length = Nl))
Xg2  = vec(collect(Iterators.product(x1, x1)))

# For K = 3, 31^3 = 29791 ---- THIS WILL NOT BE FEASIBLE, SO REDUCE GRID DENSITY
Nl  = 21
x1  = collect(range(-1, stop = 1, length = Nl))
# K = 3 grid has 9261 pts
Xg3  = vec(collect(Iterators.product(x1, x1, x1)))

# For K = 3, 31^3 = 923521 ---- THIS WILL NOT BE FEASIBLE, SO REDUCE GRID DENSITY
Nl  = 11
x1  = collect(range(-1, stop = 1, length = Nl))
# K = 4 grid has 14641
Xg4  = vec(collect(Iterators.product(x1, x1, x1, x1)))

## BUILD CANDIDATE SETS FOR POINT EXCHANGE
X2_cand_set = Matrix{Float64}(undef, length(Xg2), 2)
for i in 1:length(Xg2)
    X2_cand_set[i,:] = [Xg2[i][1], Xg2[i][2]]
end

size(X2_cand_set)
scatter(X2_cand_set[:,1], X2_cand_set[:, 2], markersize = 2, legend = false, title = "K = 2 PEXCH Searches: Candidate Set 31^2 Factorial")

X3_cand_set = Matrix{Float64}(undef, length(Xg3), 3)
for i in 1:length(Xg3)
    X3_cand_set[i,:] = [Xg3[i][1], Xg3[i][2], Xg3[i][3]]
end

X4_cand_set = Matrix{Float64}(undef, length(Xg4), 4)
for i in 1:length(Xg4)
    X4_cand_set[i,:] = [Xg4[i][1], Xg4[i][2], Xg4[i][3], Xg4[i][4]]
end


# number of points
Np2 = size(X2_cand_set)[1]
Np3 = size(X3_cand_set)[1]
Np4 = size(X4_cand_set)[1]


## --------------------------------------------------------------------------------------------

## ============================================================================================
## point exchange algorithms

function genStartingDes(N, cand_set; K = 2)
    # draw a set of random points from candidate set 
    #  to start the optimal design search
    #  NO REP Structure enforced
    Np      = size(cand_set)[1]
    pt_ind  = collect(1:1:Np)
    
    # draw initial design
    X_ind_init  = sample(pt_ind, N; replace = false)
    X_init      = cand_set[ X_ind_init, :]
    F_t         = genModelMat_fac(X_init; N = N, K = K)
    p           = size(F_t)[2]
    inf_mat_t   = transpose(F_t)*F_t
    
    # compute rank of information matrix
    rank_infmat = rank(inf_mat_t)
    while rank_infmat < p
        # if initial design gives singular information matrix
        #  then redraw the starting point
        println("ran the while loop")
        X_ind_init  = sample(pt_ind, N; replace = false)
        X_init      = cand_set[X_ind_init, :]
        F_t         = genModelMat_fac(X_init; N = N, K = K)
        inf_mat_t   = transpose(F_t)*F_t
    end
    return X_init
end

function PEXCH(; N, cand_set, K, objective)
    ## PEXCH - no PE rep structure enforced
    X_init    = genStartingDes(N, cand_set; K = K)
    X_current = deepcopy(X_init)
    I_current = objective(X_current; N = N, K = K)
    Np        = size(cand_set)[1]
    Nex       = N * Np
    crit_arr  = Vector{Float64}(undef, Nex)
    Des_arr   = Array{Any}(undef, Nex)
    k         = 0
    improve = true
    while improve
        I_last = objective(X_current; N = N, K = K)
        k = 0
        for i in 1:Np
            for j in 1:N
                k           += 1
                X_temp      = deepcopy(X_current)
                X_temp[j,:] = cand_set[i,:]
                I_temp      = objective(X_temp; N = N, K = K)
                crit_arr[k] = I_temp
                Des_arr[k]  = X_temp

            end
        end
        best_des  = argmin(crit_arr)
        I_current = crit_arr[best_des]
        X_current = Des_arr[best_des]
        if I_last == I_current 
            improve = false
        end
    end
    return X_current
end

function genStartingDes_Rep(M, cand_set; K = 2, Rvec)
    # generate starting design
    #  WITH PE Rep Structure Enforced
    R       = Diagonal(Rvec)
    Np      = size(cand_set)[1]
    pt_ind  = collect(1:1:Np)
    # draw initial design
    X_ind_init  = sample(pt_ind, M; replace = false)
    X_init      = cand_set[ X_ind_init, :]
    F_t         = genModelMat_fac(X_init; N = M, K = K)
    p           = size(F_t)[2]
    inf_mat_t   = transpose(F_t)*R*F_t
    # compute rank of information matrix
    rank_infmat = rank(inf_mat_t)
    while rank_infmat < p
        println("ran the while loop")
        X_ind_init  = sample(pt_ind, N; replace = false)
        X_init      = cand_set[X_ind_init, :]
        F_t         = genModelMat_fac(X_init; N = N, K = K)
        inf_mat_t   = transpose(F_t)*F_t
    end
    return X_init, X_ind_init
end


function PEXCH_rep(; M, cand_set, K, Rvec, objective)
    # This version of PEXCH will enforce a strict user-specified 
    #  replication structure
    X_init, pt_init    = genStartingDes_Rep(M, cand_set, K = 2, Rvec = Rvec)
    X_current = deepcopy(X_init)
    I_current = objective(X_current; M = M, K = K, Rvec = Rvec)
    Np        = size(cand_set)[1]
    Nex       = M * Np
    crit_arr  = Vector{Float64}(undef, Nex)
    Des_arr   = Array{Any}(undef, Nex)
    k         = 0
    # track pts from grid in design
    pt_arr = Array{Any}(undef, Nex)
    for np in 1:Nex
        pt_arr[np] = deepcopy(pt_init)
    end
    improve = true
    while improve
        I_last = objective(X_current; M = M, K = K, Rvec = Rvec)
        k = 0
        for i in 1:Np
            if i in pt_init
                # do nothing if pt from candidate set is already in design
                #  this is required to ensure M unique points in design.
                for j in 1:M
                    k += 1
                     crit_arr[k] = I_current
                     Des_arr[k]  = X_current 
                     pt_arr[k]   = pt_init 
                end
            else
                for j in 1:M 
                    k            += 1
                    pt_temp      = deepcopy(pt_arr[k])
                    X_temp       = deepcopy(X_current)
                    X_temp[j,:]  = cand_set[i,:]
                    I_temp       = objective(X_temp; M = M, K = K, Rvec = Rvec)
                    crit_arr[k]  = I_temp
                    Des_arr[k]   = X_temp
                    pt_temp[j]   = i
                    pt_arr[k]    = pt_temp
                end         
            end       
        end
        best_des  = argmin(crit_arr)
        I_current = crit_arr[best_des]
        X_current = Des_arr[best_des]
        pt_init   = pt_arr[best_des]
        if I_last == I_current 
            # if full pass through design without improvement, terminate
            improve = false
        else
            # if algorithm didn't converge, then update
            #  the tracking of unique points
            #  to ensure a specific rep structure
            for np in 1:Nex
                pt_arr[np] = deepcopy(pt_init)
            end
        end
    end 
    return X_current 
end
## --------------------------------------------------------------------------------------------

## ============================================================================================
## utility functions and optimal design objective functions
function genModelMat_fac(X; N, K)
        # function to generate model matrix under second order model given K = # exp. factors
        if K == 1
            x1 = X[:, 1]
            F = [fill(1, N) x1 x1.^2]
        elseif K == 2
            x1 = X[:, 1]
            x2 = X[:, 2]
            F = [fill(1, N) x1 x2 (x1 .* x2) x1.^2 x2.^2]
        elseif K == 3
            # min N = 10
            x1 = X[:, 1]
            x2 = X[:, 2]
            x3 = X[:, 3]
            F =  [fill(1, N) x1 x2 x3 (x1 .* x2) (x1 .* x3) (x2 .* x3) x1.^2 x2.^2 x3.^2]
        elseif K == 4
            x1 = X[:, 1]
            x2 = X[:, 2]
            x3 = X[:, 3]
            x4 = X[:, 4]
            F = [fill(1, N) x1 x2 x3 x4 (x1 .* x2) (x1 .* x3) (x1 .* x4) (x2 .* x3) (x2 .* x4) (x3 .* x4)  x1.^2 x2.^2 x3.^2 x4.^2]
        end
    return F
end


function I_criterion(X; N = N, K = 2)
    # I-criterion, not scaled by N (can compare I across N)
    #
    F         = genModelMat_fac(X; N = N, K = K)
    msize     = size(F)
    p         = msize[2]
    # compute model information matrix
    InfMat    = transpose(F)*F
    # volume of design space 2^k
    V = 2^K

    if K == 2
    # region moments matrix for this K = 2
        W = [   4    0    0    0  4/3  4/3 ;
                0  4/3    0    0    0    0 ;
                0    0  4/3    0    0    0 ;
                0    0    0  4/9    0    0 ;
                4/3  0    0    0  4/5  4/9 ;
                4/3  0    0    0  4/9  4/5
        ]
    elseif  K == 3
        # three factor moment matrix for second order model 10 x 10
        W = [   8   0   0   0   0   0   0   8/3 8/3 8/3 ;
                0   8/3 0   0   0   0   0   0   0   0   ;
                0   0   8/3 0   0   0   0   0   0   0   ;
                0   0   0   8/3 0   0   0   0   0   0   ;
                0   0   0   0   8/9 0   0   0   0   0   ;
                0   0   0   0   0   8/9 0   0   0   0   ;
                0   0   0   0   0   0   8/9 0   0   0   ;
                8/3 0   0   0   0   0   0   8/5 8/9 8/9 ;
                8/3 0   0   0   0   0   0   8/9 8/5 8/9 ;
                8/3 0   0   0   0   0   0   8/9 8/9 8/5
        ]
    elseif K == 4
        # region moments matrix for K = 4
        W = [  16    0    0    0    0    0    0    0    0    0    0 16/3 16/3 16/3 16/3 ;
                0 16/3    0    0    0    0    0    0    0    0    0    0    0    0    0 ;
                0    0 16/3    0    0    0    0    0    0    0    0    0    0    0    0 ;
                0    0    0 16/3    0    0    0    0    0    0    0    0    0    0    0 ;
                0    0    0    0 16/3    0    0    0    0    0    0    0    0    0    0 ;
                0    0    0    0    0 16/9    0    0    0    0    0    0    0    0    0 ;
                0    0    0    0    0    0 16/9    0    0    0    0    0    0    0    0 ;
                0    0    0    0    0    0    0 16/9    0    0    0    0    0    0    0 ;
                0    0    0    0    0    0    0    0 16/9    0    0    0    0    0    0 ;
                0    0    0    0    0    0    0    0    0 16/9    0    0    0    0    0 ;
                0    0    0    0    0    0    0    0    0    0 16/9    0    0    0    0 ;
                16/3    0    0    0    0    0    0    0    0    0    0 16/5 16/9 16/9 16/9 ;
                16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/5 16/9 16/9 ;
                16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/9 16/5 16/9 ;
                16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/9 16/9 16/5  ]
    end
    if rank(InfMat) < p
         result = typemax(Float64)
    else
         result =  tr(InfMat \ W) / V
    end

    return result
end

function I_crit_rep(X; M, K, Rvec)
    F           = genModelMat_fac(X; N = M, K = K)
    msize       = size(F)
    p           = msize[2]
    if K == 2
        # region moments matrix for this K = 2
            W = [   4    0    0    0  4/3  4/3 ;
                    0  4/3    0    0    0    0 ;
                    0    0  4/3    0    0    0 ;
                    0    0    0  4/9    0    0 ;
                    4/3  0    0    0  4/5  4/9 ;
                    4/3  0    0    0  4/9  4/5
            ]
        elseif  K == 3
            # three factor moment matrix for second order model 10 x 10
            W = [   8   0   0   0   0   0   0   8/3 8/3 8/3 ;
                    0   8/3 0   0   0   0   0   0   0   0   ;
                    0   0   8/3 0   0   0   0   0   0   0   ;
                    0   0   0   8/3 0   0   0   0   0   0   ;
                    0   0   0   0   8/9 0   0   0   0   0   ;
                    0   0   0   0   0   8/9 0   0   0   0   ;
                    0   0   0   0   0   0   8/9 0   0   0   ;
                    8/3 0   0   0   0   0   0   8/5 8/9 8/9 ;
                    8/3 0   0   0   0   0   0   8/9 8/5 8/9 ;
                    8/3 0   0   0   0   0   0   8/9 8/9 8/5
            ]
        elseif K == 4
            # region moments matrix for K = 4
            W = [  16    0    0    0    0    0    0    0    0    0    0 16/3 16/3 16/3 16/3 ;
                    0 16/3    0    0    0    0    0    0    0    0    0    0    0    0    0 ;
                    0    0 16/3    0    0    0    0    0    0    0    0    0    0    0    0 ;
                    0    0    0 16/3    0    0    0    0    0    0    0    0    0    0    0 ;
                    0    0    0    0 16/3    0    0    0    0    0    0    0    0    0    0 ;
                    0    0    0    0    0 16/9    0    0    0    0    0    0    0    0    0 ;
                    0    0    0    0    0    0 16/9    0    0    0    0    0    0    0    0 ;
                    0    0    0    0    0    0    0 16/9    0    0    0    0    0    0    0 ;
                    0    0    0    0    0    0    0    0 16/9    0    0    0    0    0    0 ;
                    0    0    0    0    0    0    0    0    0 16/9    0    0    0    0    0 ;
                    0    0    0    0    0    0    0    0    0    0 16/9    0    0    0    0 ;
                    16/3    0    0    0    0    0    0    0    0    0    0 16/5 16/9 16/9 16/9 ;
                    16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/5 16/9 16/9 ;
                    16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/9 16/5 16/9 ;
                    16/3    0    0    0    0    0    0    0    0    0    0 16/9 16/9 16/9 16/5  ]
        end
    ## create the replication matrix
    R           = Diagonal(Rvec)
    InfMat      = transpose(F)*R*F
    V           = 2^K
    if rank(InfMat) < p
        result = typemax(Float64)
    else
        result =  tr(InfMat \ W) / V
    end
    return result
end



### -----------------------------------------------------------------------------------------------

## ================================================================================================
# Run design searches and benchmark time

##    ============================   K = 2 CASES ==================================================
## Paper Table 3, Scenario 1: -------------------------------------------------
#   no constraints on PE reps - should get FC-CCD with 4 center reps
Nrun      = 100
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH(N = 12, cand_set = X2_cand_set, K = 2, objective = I_criterion)
        score = I_criterion(X_out; N = 12, K = 2)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)
scatter(X_best[:,1], X_best[:,2], legend = false, xlims = (-1.1,1.1), ylim = (-1.1,1.1),
 title = "K = 2, N = 12, I-optimal Design by PEXCH")

# 10 runs find design no prob
#142.324531 seconds (240.63 M allocations: 47.013 GiB, 3.32% gc time, 0.70% compilation time)

#= 12×2 Matrix{Float64}:
  1.0   0.0
  1.0  -1.0
 -1.0  -1.0
  0.0  -1.0
  0.0   0.0   (CP)
  0.0   1.0
  0.0   0.0   (CP)
  0.0   0.0   (CP)
  1.0   1.0
  0.0   0.0   (CP)
 -1.0   0.0
 -1.0   1.0

 # crit score matches PSO design in paper
0.30277777777777776 =#

Ds1 = DataFrame(K = 2, N = 12, M = 12, crit = "I", crit_score = crit_arr, design = Des_arr)
CSV.write("1.1_K2_Icrit_Scen1_PEXCH.csv", Ds1)



## Paper Table 3, Scenario 2: -------------------------------------------------
#   M = 11 unique points
#   11th point rep'd twice

Nrun      = 100
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs2      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 11, cand_set = X2_cand_set, K = 2, Rvec = Rvs2, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 11, K = 2, Rvec = Rvs2)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)
scatter(X_best[:,1], X_best[:,2], legend = false, xlims = (-1.1,1.1), ylim = (-1.1,1.1),
 title = "K = 2, N = 12, I-optimal Design by PEXCH")


 Ds2 = DataFrame(K = 2, N = 12, M = 11, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.2_K2_Icrit_Scen2_PEXCH.csv", Ds2)
 


## Paper Table 3, Scenario 3: -------------------------------------------------
#   M = 10 unique points
#   10th point rep'd thrice

Nrun      = 100
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs3      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 10, cand_set = X2_cand_set, K = 2, Rvec = Rvs3, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 10, K = 2, Rvec = Rvs3)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)
scatter(X_best[:,1], X_best[:,2], legend = false, xlims = (-1.1,1.1), ylim = (-1.1,1.1),
 title = "K = 2, N = 12, I-optimal Design by PEXCH")


 Ds3 = DataFrame(K = 2, N = 12, M = 10, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.3_K2_Icrit_Scen3_PEXCH.csv", Ds3)
 


## Paper Table 3, Scenario 4: -------------------------------------------------
#   M = 10 unique points
#   9th and 10th point each rep'd 2 times

Nrun      = 100
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs4     = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 10, cand_set = X2_cand_set, K = 2, Rvec = Rvs4, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 10, K = 2, Rvec = Rvs4)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end



# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)
scatter(X_best[:,1], X_best[:,2], legend = false, xlims = (-1.1,1.1), ylim = (-1.1,1.1),
 title = "K = 2, N = 12, I-optimal Design by PEXCH")


 Ds4 = DataFrame(K = 2, N = 12, M = 10, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.4_K2_Icrit_Scen4_PEXCH.csv", Ds4)


## Paper Table 3, Scenario 16: -------------------------------------------------
#   M = 7 unique points
#   design pts 3 through 7 each rep'ed twice

Nrun      = 100
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs16     = [1, 1, 2, 2, 2, 2, 2]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 7, cand_set = X2_cand_set, K = 2, Rvec = Rvs16, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 7, K = 2, Rvec = Rvs16)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end



# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)
scatter(X_best[:,1], X_best[:,2], legend = false, xlims = (-1.1,1.1), ylim = (-1.1,1.1),
 title = "K = 2, N = 12, I-optimal Design by PEXCH")


 Ds5 = DataFrame(K = 2, N = 12, M = 7, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.5_K2_Icrit_Scen16_PEXCH.csv", Ds5)





##    ============================   K = 3 CASES ==================================================
## Paper Table 4, Scenario 1: -------------------------------------------------
#   no constraints on PE reps - should get FC-CCD with 4 center reps
Nrun      = 50
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH(N = 16, cand_set = X3_cand_set, K = 3, objective = I_criterion)
        score = I_criterion(X_out; N = 16, K = 3)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)

Ds6 = DataFrame(K = 3, N = 16, M = 16, crit = "I", crit_score = crit_arr, design = Des_arr)
CSV.write("1.6_K3_Icrit_Scen1_PEXCH.csv", Ds6)


## Paper Table 4, Scenario 3: -------------------------------------------------
#   M = 14, last point rep'd 3 times
Nrun      = 50
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs3      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 14, cand_set = X3_cand_set, K = 3, Rvec = Rvs3, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 14, K = 3, Rvec = Rvs3)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end



# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)


 Ds7= DataFrame(K = 3, N = 16, M = 14, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.7_K3_Icrit_Scen3_PEXCH.csv", Ds7)


## Paper Table 4, Scenario 4: -------------------------------------------------
#   M = 14, last 2 pts each rep'd 2 times
Nrun      = 50
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs4      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 14, cand_set = X3_cand_set, K = 3, Rvec = Rvs4, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 14, K = 3, Rvec = Rvs4)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end



# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)


 Ds8 = DataFrame(K = 3, N = 16, M = 14, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("1.8_K3_Icrit_Scen4_PEXCH.csv", Ds8)



 ##    ============================   K = 4 CASES ==================================================
## Paper Table 5, Scenario 1: -------------------------------------------------
#   no constraints on PE reps - should get FC-CCD with 4 center reps
Nrun      = 20
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH(N = 20, cand_set = X4_cand_set, K = 4, objective = I_criterion)
        score = I_criterion(X_out; N = 20, K = 4)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)

Ds9 = DataFrame(K = 4, N = 20, M = 20, crit = "I", crit_score = crit_arr, design = Des_arr)
CSV.write("1.9_K4_Icrit_Scen1_PEXCH.csv", Ds9)


## Paper Table 5, Scenario 2: -------------------------------------------------
#   M = 19, last 2 pts each rep'd 2 times
Nrun      = 20
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs2      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 19, cand_set = X4_cand_set, K = 4, Rvec = Rvs2, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 19, K = 4, Rvec = Rvs2)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)


 Ds10 = DataFrame(K = 4, N = 20, M = 19, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("10.0_K4_Icrit_Scen2_PEXCH.csv", Ds10)

 
 ## Paper Table 5, Scenario 4: -------------------------------------------------
#   M = 18, last 3 pts each rep'd 2 times
Nrun      = 20
crit_arr  = Vector{Float64}(undef, Nrun)
Des_arr   = Array{Any}(undef, Nrun)
Rvs4      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]

@time begin
    for i in 1:Nrun
        println(i)
        X_out = PEXCH_rep(; M = 18, cand_set = X4_cand_set, K = 4, Rvec = Rvs4, objective = I_crit_rep)
        score = I_crit_rep(X_out; M = 18, K = 4, Rvec = Rvs4)
        crit_arr[i] = score
        Des_arr[i]  = X_out
    end
end

# check what PEXCH found in Nrun random starts
X_best = Des_arr[argmin(crit_arr)]
minimum(crit_arr)


 Ds11 = DataFrame(K = 4, N = 20, M = 18, crit = "I", crit_score = crit_arr, design = Des_arr)
 CSV.write("10.1_K4_Icrit_Scen4_PEXCH.csv", Ds11)


