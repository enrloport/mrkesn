function __fill_X_MrKESN_mnist!(mrE, args::Dict )
    
    function f(u)
        if args[:gpu]
            return CuArray(reshape(u, :, 1))
        else
            return reshape(u, :, 1)
        end
    end


    for t in 1:args[:train_length]
        ku = args[:i2k](args[:train_data][:,:,t])
        for i in 1:length(mrE.esns)
            for _ in 1:args[:initial_transient]
                update(mrE.esns[i], ku, f )
            end
            update(mrE.esns[i], ku, f )
        end
        aux = vcat(f(ku),[es.x for es in mrE.esns]...,f([mrE.constant_value for _ in 1:mrE.constant_terms ]))
        mrE.X[:,t] = aux
    end
end


function __make_Rout_MrKESN_mnist!(mrE,args)
    X             = mrE.X
    classes       = args[:classes]
    classes_Yt    = Dict( c => zeros(args[:train_length]) for c in classes )  # New dataset for each class

    for t in 1:args[:train_length]
        lt = args[:train_labels][t]
        for c in classes
            y = lt == c ? 1.0 : 0.0
            classes_Yt[c][t] = y
        end
    end
    if args[:gpu]
        classes_Yt = Dict( k => CuArray(classes_Yt[k]) for k in keys(classes_Yt) )
    end

    cudamatrix = args[:gpu] ? CuArray : Matrix
    mrE.classes_Routs = Dict( c => cudamatrix(transpose((X*transpose(X) + mrE.beta*I) \ (X*classes_Yt[c]))) for c in classes )
end


function __do_train_MrKESN_mnist!(mrE, args)
    num   = args[:train_length]
    mrE.X = zeros( sum([esn.R_size for esn in mrE.esns]) + args[:image_size][1]*args[:image_size][2]*11 + mrE.constant_terms  , num)
    for i in 1:length(mrE.esns)
        mrE.esns[i].x = zeros( mrE.esns[i].R_size, 1)
    end

    if args[:gpu]
        mrE.X = CuArray(mrE.X)
        for i in 1:length(mrE.esns)
            mrE.esns[i].x = CuArray(mrE.esns[i].x)
        end
    end

    __fill_X_MrKESN_mnist!(mrE,args)
    __make_Rout_MrKESN_mnist!(mrE,args)
end