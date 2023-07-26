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

function resize_map_mnist(train_x, sz, trl)
    trx = mapslices(x-> imresize(x, sz), train_x[:,:,1:trl] ,dims=(1,2))
    trx[:] = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, trx)
    return trx
end

function map_resize_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(x-> imresize(x, sz), trx[:,:,1:trl] ,dims=(1,2))
    return trx
end

px      = 14
sz      = (px,px)

rsm  = resize_map_mnist(test_x, sz, tel)
mrs  = map_resize_mnist(test_x, sz, tel)


xmpl = 1
or  = test_x[:,:,xmpl]'
rs  = rsm[:,:,xmpl]'
mr  = mrs[:,:,xmpl]'

all = zeros(28, 28*3)
all[1:28,1:28]= or
all[1:px, 28+1:28+px]= rs
all[1:px, 28*2+1:28*2+px]= mr

Images.Gray.(all)



Images.Gray.(or)


clean_or = or[ vec(mapslices(col -> any(col .!= 0), or, dims = 2)), vec(mapslices(col -> any(col .!= 0), or, dims = 1))]
Images.Gray.(clean_or)



function clean_resize_map_mnist(train_x, sz, trl)
    trx = mapslices(
            x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
        )
    trx[:] = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, trx)
    return trx
end

function clean_map_resize_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(
        x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
    )
    return trx
end


crsm  = clean_resize_map_mnist(test_x, sz, tel)
cmrs  = clean_map_resize_mnist(test_x, sz, tel)

xmpl  = 3
or    = test_x[:,:,xmpl]'
crsmx = crsm[:,:,xmpl]'
cmrsx = cmrs[:,:,xmpl]'

Images.Gray.( hcat(or, vcat(crsmx,cmrsx)) )

alot = [hcat(test_x[:,:,xmpl]', vcat(crsm[:,:,xmpl]',cmrs[:,:,xmpl]')) for xmpl in 11:20]

Images.Gray.(vcat(alot...))