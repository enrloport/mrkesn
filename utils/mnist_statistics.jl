include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb

# using Pkg
# Pkg.add("Noise")
# using Noise


# Random.seed!(42)

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
# tr_x, tr_y = FashionMNIST(split=:train)[:]
# te_x, te_y = FashionMNIST(split=:test)[:]

trl, tel = 6000,1000

# function resize_map_mnist(train_x, sz, trl)
#     trx = mapslices(x-> imresize(x, sz), train_x[:,:,1:trl] ,dims=(1,2))
#     trx[:] = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, trx)
#     return trx
# end
# function map_resize_mnist(train_x, sz, trl)
#     trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
#     trx = mapslices(x-> imresize(x, sz), trx[:,:,1:trl] ,dims=(1,2))
#     return trx
# end
# px      = rand([20])
# sz      = (px,px)

# rem  = resize_map_mnist(test_x, sz, tel)
# mr  = map_resize_mnist(test_x, sz, tel)


xmpl = 1
or  = test_x[:,:,xmpl]'


Images.Gray.(or)

poses_train = Dict(
    c => [i for i in 1:length(train_y) if train_y[i] == c]
    for c in 0:9
)
poses_test = Dict(
    c => [i for i in 1:length(test_y) if test_y[i] == c]
    for c in 0:9
)

# m0 = Array(mean(train_x[:,:,poses_train[0] ], dims=3 ))[:,:,1]


########################################


#### MEAN ####
means = Dict(
    c => Array(mean(train_x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
    for c in 0:9
)

bm0 = means[0]

b_means = Dict(
    c => map( x -> x>0.1 ? 1 : 0 , means[c])
    for c in 0:9
)
all_mean = hcat( [means[c]' for c in keys(means) ]... )
all_bmean= hcat( [b_means[c]' for c in keys(b_means) ]... )
Images.Gray.(
    vcat(
        all_mean, all_bmean
    )
)

#### MEDIAN ####
medians = Dict(
    c => Array(median(train_x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
    for c in 0:9
)
all_median = hcat( [medians[c]' for c in keys(medians) ]... )
Images.Gray.(all_median)

#### VARIANCE ####
vars = Dict(
    c => Array(var(train_x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
    for c in 0:9
)
all_vars = hcat( [vars[c]' for c in keys(vars) ]... )
Images.Gray.(all_vars)

#### STANDARD DEVIATION ####
stds = Dict(
    c => Array(std(train_x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
    for c in 0:9
)
all_stds = hcat( [stds[c]' for c in keys(stds) ]... )
Images.Gray.(all_stds)

#### SUM ####
aux = Dict(
    c => Array(sum(train_x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
    for c in 0:9
)
sums = Dict(
    # c => aux[c] ./ norm(aux[c]) # normalized
    c => aux[c] ./ (maximum(aux[c])) # equalized
    for c in 0:9
)

all_sums = hcat( [sums[c]' for c in keys(sums) ]... )
Images.Gray.(all_sums)


####### ALL ########
Images.Gray.(
    vcat(all_mean,all_median,all_vars,all_stds,all_sums)
)

xmpl = 1
x,y = train_x[:,:,xmpl]', train_y[xmpl]

Images.Gray.(
    hcat(
        vcat(
             x .* means[y]'
            # ,x .* b_means[y]'
            ,x .* medians[y]'
            ,x .* vars[y]'
            ,x .* stds[y]'
            ,x .* sums[y]'
        )
        ,vcat(
             x .*   means[0]'
            # ,x .* b_means[0]'
            ,x .* medians[0]'
            ,x .*    vars[0]'
            ,x .*    stds[0]'
            ,x .*    sums[0]'
        )
        # ,vcat(
        #      means[y]'
        #     ,b_means[y]'
        #     ,medians[y]'
        #     ,vars[y]'
        #     ,stds[y]'
        #     ,sums[y]'
        # )
        # ,vcat(
        #        means[9]'
        #     ,b_means[9]'
        #     ,medians[9]'
        #     ,   vars[9]'
        #     ,   stds[9]'
        #     ,   sums[9]'
        # )
    )
)