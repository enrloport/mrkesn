# Function to test an already trained MrESN struct
function __do_test_MrKESN_mnist!(mrE, args::Dict)
    test_length = args[:test_length]

    classes_Y    = Array{Tuple{Float64,Int,Int}}[]
    wrong_class  = []
    mrE.Y        = []

    # if args[:gpu]
    #     # mrE.Y = CuArray(mrE.Y)
    #     f = (u) -> CuArray(reshape(u, :, 1))
    # else
    #     f = (u) -> reshape(u, :, 1)
    # end

    function f(u)
        if args[:gpu]
            return CuArray(reshape(u, :, 1))
        else
            return reshape(u, :, 1)
        end
    end

    for t in 1:test_length
        ku = args[:i2k](args[:train_data][:,:,t])
        for i in 1:length(mrE.esns)
            for _ in 1:args[:initial_transient]
                update(mrE.esns[i], ku, f )
            end
            update(mrE.esns[i], ku, f )
        end
        x = vcat(f(ku),[es.x for es in mrE.esns]..., f([mrE.constant_value for _ in 1:mrE.constant_terms ]) )

        pairs  = []
        for c in args[:classes]
            yc = Array(mrE.classes_Routs[c] * x)[1]
            push!(pairs, (yc, c, args[:test_labels][t]))
        end
        pairs_sorted  = reverse(sort(pairs))
        
        if pairs_sorted[1][2] != pairs_sorted[1][3]
            push!(wrong_class, (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
        end

        push!(mrE.Y,[pairs_sorted[1][2] ;])
        push!(classes_Y, pairs )
    end

    mrE.wrong_class= wrong_class
    mrE.classes_Y  = classes_Y
    mrE.Y_target   = args[:test_labels]
    mrE.error      = length(wrong_class) / length(classes_Y)

end