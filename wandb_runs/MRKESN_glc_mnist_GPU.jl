include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb


# Random.seed!(42)

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
#train_x, train_y = FashionMNIST(split=:train)[:]
#test_x, test_y = FashionMNIST(split=:test)[:]

repit =1#500
_params = Dict{Symbol,Any}(
     :gpu           => false
    ,:wb            => false
    ,:wb_logger_name=> "MRKESN_glc_mnist_GPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1]
    ,:test_length   => size(test_y)[1]
    ,:train_f       => __do_train_MrKESN_mnist!
    ,:test_f        => __do_test_MrKESN_mnist!
    ,:B => 0.007
    ,:K => 1.5
)
const _B = _params[:B]
const _K = _params[:K]

#Generalised Logistic Curve
# A: Upper asymptote
# K: Lower asymptote
# C: Typically takes a value of 1. Otherwise, the upper asymptote is A + (K-A) / (C^(1/v))
# B: Growth rate
# v > 0: affects near which asymptote maximum growth occurs
# Q: is related to the value glc(0)
function glc(x; A=-_K, K=_K, C=1.0, B=_B, v=1.0, Q=1.0)
    return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/v) )
end


#for i in -10:10
#    v = i
#    println(v," ", glc(v))
#end

# b = 0.1
# max = 4
# x = [x for x in -1000:10:1000]
# y = [glc(i/50;B=b) for i in x ]
# ys = [ [glc(i/50;B=b/10) for i in x ] for b in 1:max]
# plot(x,ys; title="B influence in glc", labels= [b/10 for b in 1:max]' )

function do_batch(_params_esn, _params,sd)
    sz       = _params[:image_size]
    # im_sz    = sz[1]*sz[2]
    im_sz    = sz[1]*sz[2]*11
    nodes    = _params_esn[:nodes] #im_sz
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    sgmds    = _params_esn[:sgmds]
    densities= _params_esn[:density]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]
    glcs     = _params_esn[:glcs]
    esns = [
        KESN( 
             R      = _params[:gpu] ? CuArray(new_R(nodes[i], density=densities[i], rho=rhos[i])) : new_R(nodes[i], density=densities[i], rho=rhos[i])
            ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], im_sz )) : rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], im_sz )
            ,R_scaling = r_scales[i]
            ,alpha  = alphas[i]
            ,rho    = rhos[i]
            ,sigma  = sigmas[i]
            #,sgmd   = _x -> sgmds[i](_x; B =_params_esn[:bs][i] )
            ,sgmd = glcs[i]
        ) for i in 1:_params[:num_esns]
    ]

    tms = @elapsed begin
        mrE = MrESN(
            esns=esns
            ,beta=_params[:beta] 
            ,train_function = _params[:train_f]
            ,test_function = _params[:test_f]
            ,constant_terms = _params[:num_esns]
            ,constant_value = 1.0
            )
        tm_train = @elapsed begin
            mrE.train_function(mrE,_params)
        end
        println("TRAIN FINISHED, ", tm_train)
        tm_test = @elapsed begin
            mrE.test_function(mrE,_params)
        end
        println("TEST FINISHED, ", tm_test)
    end
 
    to_log = Dict(
        "Total time" => tms
        ,"Train time"=> tm_train
        ,"Test time" => tm_test
       , "Error"     => mrE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
    end
    return mrE
end


function transform_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(x-> imresize(x, sz), trx[:,:,1:trl] ,dims=(1,2))
    return trx
end

# r1      = 0
px      = 14 # rand([14,20,25,28])
sz      = (px,px)
train_x = transform_mnist(train_x, sz, _params[:train_length] )
test_x  = transform_mnist(test_x, sz, _params[:test_length])



function knowledge_mean(x, y)
    #### Classes by position ####
    poses_train = Dict(
        c => [i for i in 1:length(y) if y[i] == c]
        for c in 0:9
    )
    #### MEAN ####
    means = Dict(
        c => Array(mean(x[:,:,poses_train[c] ], dims=3 ))[:,:,1]
        for c in 0:9
    )
    means = Dict(
        c => means[c] ./ maximum(means[c])
        for c in keys(means)
    )
    return means
end


means = knowledge_mean(train_x,train_y)
all_mean = hcat( [means[c]' for c in keys(means) ]... )
# Images.Gray.(all_mean)

function img_to_knowledge(example, knowledge_dict=means)
    return vcat(
        example,[example .* knowledge_dict[k] for k in keys(knowledge_dict) ]...
    )
end

# ki = img_to_knowledge(train_x[:,:,1])
# Images.Gray.(ki')





for _ in 1:repit
    sd = rand(1:10000)
    Random.seed!(sd)
    _params[:num_esns] = 1 # rand([10,15,20,25])
    # _params[:num_hadamard] = 0 # rand([1,2])
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
        ,:density  => rand(Uniform(0.01,0.7),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => [10000 for _ in 1:_params[:num_esns] ] # rand([500, px*px ,1000],_params[:num_esns])
        ,:sgmds    => rand([glc],_params[:num_esns])
        # ,:sgmds    => [ ln for _ in 1:_params[:num_esns] ]
        ,:glcs     => [glc for _ in 1:_params[:num_esns]]
    )
    _params[:initial_transient] = rand([1,2,3])
    _params[:image_size]   = sz
    _params[:train_data]   = train_x
    _params[:test_data]    = test_x
    _params[:train_labels] = train_y
    _params[:test_labels]  = test_y
    _params[:i2k]          = img_to_knowledge
    par = Dict(
        "Reservoirs" => _params[:num_esns]
        # ,"Hadamard reservoirs" => _params[:num_hadamard]
        , "Total nodes"        => sum(_params_esn[:nodes])
        , "Train length"       => _params[:train_length]
        , "Test length"        => _params[:test_length]
        , "Resized"            => _params[:image_size][1]
        , "Nodes per reservoir"=> _params_esn[:nodes]
        , "Initial transient"  => _params[:initial_transient]
        , "seed"               => sd
        , "sgmds"              => _params_esn[:sgmds]
        # , "bs"                 => _params_esn[:bs]
        , "alphas"             => _params_esn[:alpha]
        , "densities"          => _params_esn[:density]
        , "rhos"               => _params_esn[:rho]
        , "sigmas"             => _params_esn[:sigma]
        , "R_scalings"         => _params_esn[:R_scaling]
        , "glcs"               => _params_esn[:glcs]
        , "Constant terms"     => _params[:num_esns]
        , "B"                  => _params[:B]
        , "K"                  => _params[:K]
        )
    if _params[:wb]
        using Logging
        using Wandb
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    else
        display(par)
    end
    par = Dict(""=>0)
    GC.gc()

    tm = @elapsed begin
        r1 = do_batch(_params_esn,_params, sd)
    end
    if _params[:wb]
        close(_params[:lg])
    end
    println("Error: ", r1.error )
    if _params[:gpu]
        println("Time GPU: ", tm )
    else
        println("Time CPU: ", tm )
    end
end


# EOF