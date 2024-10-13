module PostTensorBoard

using TensorBoardLogger
using Plots
using Printf
using Dates: format as dformat
using DataFrames
using ValueHistories
using FileIO
using VideoIO
using ImageIO
import ProtoBuf

export output_with_keys
export post_tb_data
export domain2mp4
export get_imgs

struct DFImage
    subpath::String
end
Base.isless(::DFImage, ::DFImage) = false

struct DFTypst{T}
    names::Vector{String}
end

function TensorBoardLogger.deserialize_tensor_summary(::TensorBoardLogger.tensorboard.var"Summary.Value")
    return nothing 
end

"""
    _get_run_hparams(path::String)
get the hyperparameters of a run from tensorboard log file located at `path`.
"""
function _get_run_hparams(path::String)
    file = first(readdir(path))
    f = open(joinpath(path, file), "r")
    s = TensorBoardLogger.read_event(f)
    SESSION_START_INFO_TAG = "_hparams_/session_start_info"
  
    while s.what.name != :summary || s.what.value.value[1].tag != SESSION_START_INFO_TAG
        s = TensorBoardLogger.read_event(f)
    end
  
    hparams_metadata_encoded_bytes = s.what.value.value[1].metadata.plugin_data.content
    d = TensorBoardLogger.ProtoDecoder(IOBuffer(deepcopy(hparams_metadata_encoded_bytes)))
    decoded_content = ProtoBuf.Codecs.decode(d, TensorBoardLogger.HP.HParamsPluginData)
    decoded_session_info = decoded_content.data.value.hparams
    hparams = Dict{String, Union{String, Bool, Real}}()
    for (k, v) in decoded_session_info
      push!(hparams, k => v.kind.value)
    end
    return hparams
end

"""
    get_tb_hparams(tb_path::String)
get the hyperparameters of all runs from tensorboard log files located at `tb_path`.
"""
function get_tb_hparams(tb_path::String)
    runs = parse_runs(tb_path)
    dfs = DataFrame()
    for run in runs 
        hp = _get_run_hparams(joinpath(tb_path, run))
        hp["run_i"] = run
        append!(dfs, DataFrame(hp))
    end

    return dfs[!, Cols(Not("run_i"), "run_i")]
end

"""
    get_run_data(path::String; kwargs...)
get the raw data of a run in the form of `Dict` from tensorboard log file located at `path`.

# Keyword Arguments
- ignore_tags::Vector{Symbol}= [Symbol("host/base")]: tags to ignore
- kwargs: other keyword arguments to pass to `TensorBoardLogger.map_summaries`
"""
function get_run_data(path::String; ignore_tags::Vector{Symbol}= [Symbol("host/base")], kwargs...)
    tb = TBReader(path)
    hist = MVHistory()

    TensorBoardLogger.map_summaries(tb; kwargs...) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    s = hist.storage
    return s
end

"""
    parse_runs(tb_path::String, [runs::Union{Vector{String}, Vector{Int}, Nothing}])
get the valid runs located at `tb_path` in the form of `Vector{String}` like `["run_1", "run_2", ...]`.
"""
function parse_runs(tb_path::String, runs::Union{Vector{String}, Vector{Int}, Nothing}= nothing)
    if isnothing(runs)
        runs = readdir(tb_path)
    elseif runs isa Vector{Int}
        runs = map(i -> "run_$(i)", runs)
    end

    valid_runs = filter(x -> isdir(joinpath(tb_path, x)), runs)

    if length(valid_runs) != length(runs)
        @warn "some files in the subdir are wrong, please check!"
    end

    return valid_runs
end

"""
    domain2mp4(tb_path::String, [runs]; kwargs...)
convert the domain images of one, several or all runs to mp4 video which will be stored in `tb_path`.

# Keyword Arguments
- fr::Int= 5: frame rate of the video
"""
function domain2mp4(tb_path, runs= nothing; fr::Int= 5, tag= "domain/χ")
    runs = parse_runs(tb_path, runs)
    
    encoder_options = (color_range=2, crf=0, preset="veryslow")

    for run in runs
        run_path = joinpath(tb_path, run)
        img_list = get_imgs(run_path, :, tag)

        file = run_path*".mp4"
        infile = run_path*"_.mp4"
        first_img = first(img_list)
        sz = map(x -> x ÷ 2 * 2, size(first_img))
        open_video_out(infile, eltype(first_img), sz, framerate=fr, encoder_options=encoder_options) do writer
            for img in img_list
                write(writer, img)
            end
        end
        cmd = `ffmpeg -i $(infile) -r 24 $(file) -y`
        readchomp(cmd)
        rm(infile)
    end
    return nothing
end

function get_imgs(run_path, Ids, tag= "domain/χ")
    df_img = get_run_data(run_path; tags=tag)
    return df_img[Symbol(tag)].values[Ids]
end

Plots.plot!(::Nothing, args...; kwargs...) = nothing
Plots.png(::Nothing, args...; kwargs...) = nothing


"""
    post_tb_data(trajectory_tags::Matrix{String}, scalar_tags::Vector{String}, data_path::String; image_tag::String = "domain/χ")
post-process the trajectory data with tag `trajectory_tags` and scalar data with tag `scalar_tags` in `data_path`.

# Keyword Arguments
- image_tag::String= "domain/χ": the tag of the domain images
"""
function post_tb_data(trajectory_tags::Matrix{String}, scalar_tags::Vector{String}, data_path::String; image_tag::String = "domain/χ", typst_img_path = "images", regen= true)
    tb_path = joinpath(data_path, "tb")
    @assert isdir(tb_path) "tb path does not exist."
    hp = get_tb_hparams(tb_path)

    post_path = joinpath(data_path, "post")
    mkpath(post_path)
    @info "all post-processed data will be stored in $post_path"

    img_path = joinpath(post_path, "images")
    mkpath(img_path)
    @info "processing images... all images will be stored in $img_path"
    
    df_results = DataFrame()
    _all = nrow(hp)
    num_done = 0
    for I = 1:_all
        # try
            run_i = hp[I, :run_i]
            scalars = get_run_data(joinpath(tb_path, run_i), tags=scalar_tags)
            trajectories = get_run_data(joinpath(tb_path, run_i), tags=trajectory_tags)
            images = get_run_data(joinpath(tb_path, run_i), tags=image_tag)

            img_init_chi = "init_chi_$(run_i).png"
            img_trajectory = "trajectory_$(run_i).png"
            img_chis = map(1:length(trajectory_tags)) do j
                return "chi_$(run_i)_tag_$(j).png"
            end

            _path_img_init_chi = joinpath(img_path, img_init_chi)
            regen && FileIO.save(_path_img_init_chi, first( images[Symbol(image_tag)].values ))
            
            _path_img_trajectory = joinpath(img_path, img_trajectory)
            fig_trajectory = regen ? plot(title= "Objective Functional", xlabel= "iteration", ylabel= "value") : nothing
            
            _d_tr = Pair{String, Union{DFImage, Float64, Float32}}[]
            for j = eachindex(trajectory_tags)
                img_chi_path = joinpath(img_path, img_chis[j])
                tag = trajectory_tags[j]
                h = trajectories[Symbol(tag)]
                val, step = findmin(h.values)
                _min_val = abs(val) < 1e-1 ? val : round(val; digits= 2)

                J = min(step + 10, length(h.iterations))
                plot!(fig_trajectory, h.iterations[1:J], h.values[1:J], label= tag, linewidth= 3)
                
                plot!(fig_trajectory, [step], [val], seriestype=:scatter, label= @sprintf("%.1f", _min_val))
                regen && save(img_chi_path, images[Symbol(image_tag)].values[step])
                
                out_tag = split(tag, "/")[end]
                push!(_d_tr, 
                    out_tag => _min_val, 
                    "chi_"*out_tag => DFImage(joinpath(typst_img_path, img_chis[j]))
                )
            end
            png(fig_trajectory, _path_img_trajectory)
            
            _d_scalar = Pair{String, String}[]
            for j = eachindex(scalar_tags)
                tag = scalar_tags[j]
                if haskey(scalars, Symbol(tag))
                    h = scalars[Symbol(tag)]
                    _end_val = @sprintf("%.2e", h.values[end])
                else
                    _end_val = "missing"
                end
                out_tag = split(tag, "/")[end]
                push!(_d_scalar, out_tag => _end_val)
            end
            append!(df_results, DataFrame(
                _d_tr...,
                "trajectory" => DFImage(joinpath(typst_img_path, img_trajectory)),
                _d_scalar...,
                "run_i" => run_i,
            ))

            num_done += 1
            @info "$run_i ($num_done // $_all) completed."
        # catch; 
        # end
    end
    return hp, df_results
end

@inline _get_typst_hp(key, hp) = DFTypst{:HP}([x for x in names(hp) if (x != key && x !="run_i")])
@inline _get_typst_data(key, name4df_results) = DFTypst{:DATA}(vcat(key, name4df_results))

"""
    _parse_item(::Val{F}, x::Union{Float64, DFImage, Any}) where F
parse the item `x` to string.
"""
function _parse_item(::Val{F}, x::AbstractFloat) where F 
    str = abs(x) < 0.1 ? @sprintf("%.2e", x) : @sprintf("%.2f", x)
    if F 
        str = "#box(stroke: green, inset: 4pt)[$str]"
    end
    return str
end
_parse_item(::Val{F}, x::DFImage) where F = "#image(\"$(x.subpath)\")"
_parse_item(::Val{F}, x) where F = string(x)

"""
    parse_to_typ(df::AbstractDataFrame, dftyp::DFTypst{Union{:HP, :DATA}}, grp_id)
parse the hyperparameters (`:HP`) or data (`:DATA`) of `df` which is `grp_id`th subgroup to typst string.
"""
function parse_to_typ(df::AbstractDataFrame, dftyp::DFTypst{:HP}, grp_id)
    dfnames = dftyp.names
    n = length(dfnames)
    title = "th"*join(["[$item]" for item in dfnames], "")
    data = "tr"*join(["[$(df[1, nm])]" for nm in dfnames ], "")
    typ_str = """
    = Case $grp_id
    == Config
    #easytable({
    let th = th.with(trans: emph)
    let tr = tr.with(
        cell_style: (x: none, y: none)
        => (fill: if calc.even(y) {
            luma(95%)
        } else {
            none
        })
    )
    cstyle(..(center + horizon,)*$n)
    cwidth(..(1fr,)*$n)
    $title
    $data
    })
    """
    return typ_str
end
function parse_to_typ(df::AbstractDataFrame, dftyp::DFTypst{:DATA})
    dfnames = dftyp.names
    n = length(dfnames)

    title = "th"*join(["[$item]" for item in dfnames], "")
    argmin_id = map(argmin, eachcol(df[!, dfnames]))
  
    data = [
       "tr"*join(["[$(_parse_item(Val(i == argmin_id[j]), df[i, nm]))]" for (j, nm) in enumerate(dfnames)], "") for i = 1:nrow(df)
    ]
  
    data_str = """
    == Results
    #easytable({
    let th = th.with(trans: emph)
    let tr = tr.with(
        cell_style: (x: none, y: none)
        => (fill: if calc.even(y) {
            luma(95%)
        } else {
            none
        })
    )
    cstyle(..(center + horizon,)*$n)
    cwidth(..(1fr,)*$n)
    $title
    $(join(data, "\n"))
    })
    """
    return data_str
end

"""
    output_with_keys(path, keys; kwargs...)
output the typst file of the post-processed data in `path`.

# Keyword Arguments
- tags::Vector{String}= ["energy/E" "energy/Jt"]: the tags of the data
"""
function output_with_keys(path_list, keys, tags; ex_tags= String[], regen= true)
    main_path = first(path_list)
    hp, df_results = post_tb_data(tags, ex_tags, main_path; regen= regen)
    
    for i = 2:length(path_list)
        path = path_list[i]
        typst_img_path = "images_$i"
        _hp, _df_results = post_tb_data(tags, ex_tags, path; typst_img_path= typst_img_path, regen= regen)
        
        @assert names(_hp) == names(hp) "hyperparameters does not have the same items."
        _hp = combine(_hp, Not("run_i"), "run_i" => (x -> "$(i)_" .* x) => "run_i")
        _df_results = combine(_df_results, Not("run_i"), "run_i" => (x -> "$(i)_" .* x) => "run_i")
        
        src = joinpath(path, "post", "images")
        dest = joinpath(main_path, "post", typst_img_path)
        cp(src, dest; follow_symlinks= true, force= true)
        append!(df_results, _df_results)
        append!(hp, _hp)
    end
    post_path = joinpath(main_path, "post")
    ns = names(df_results)

    typfile = joinpath(post_path, "main.typ")
    if length(keys) > 1
        out_prefix = joinpath(post_path, "pdf")
        mkpath(out_prefix)
    else
        out_prefix = post_path
    end
    if length(path_list) > 1 
        ts = map(path_list) do p 
            DateTime(replace(splitpath(p)[end], '_' => ':'))
        end
        tmin, tmax = extrema(ts)
        file = dformat(tmin, "mm-dd") * "TO" * dformat(tmax, "mm-dd")
    else
        file = ""
    end    
    
    for key in keys 
        df_typst_hp = _get_typst_hp(key, hp)
        df_typst_data = _get_typst_data(key, ns)
        df = rightjoin(hp, df_results; on= "run_i")

        grp_results = groupby(df, Not(ns, key))

        outfile = joinpath(out_prefix, file * key)
        open(typfile, "w") do io 
            print(io, """
        #import "@preview/easytable:0.1.0": easytable, elem
        #import elem: *
        """)

            for i = 1:length(grp_results)
                print(io, parse_to_typ(grp_results[i], df_typst_hp, i))
                print(io, parse_to_typ(grp_results[i], df_typst_data))
            end
        end

        run(`typst compile $typfile $outfile.pdf`)
        @info "compiling $outfile.pdf ..."
    end

    rm(typfile)
    return nothing
end




end
