"""
    run_with_configs(f, vec_configs, comments; keyargs...)

# Arguments
- `f::Function` function to accept single config, must has the signature `f(config, vtk_file_prefix, vtk_file_pvd, tb_lg, i)`
    - `config::Dict{String, Any}` single config generated from `vec_configs`
    - `vtk_file_prefix` prefix to store `vtu` file when storing data as `pvd` format
    - `vtk_file_pvd` pvd file handler, usage `vtk_file_pvd[iter] = createpvd(trian, vtk_file_prefix * String(iter),...)`
    - `tb_lg` tensorboard logger, redirect to the documents of TensorBoardLogger.jl for more details
    - `i::Int` index of config in the generated list with vec_configs
- `vec_configs::Vector{<:Pair{String, Any}}` 
- `comments` some helpful comments to you, which will be stored in the toml file

# Keyword Arguments
- `prefix` global path, default "./"
- `max_it` maximal iteration for iterations

# Directories will be created under `prefix/`
    - data
        - yyyy-mm-ddTHH_MM_SS
            - errors: some output logs if errors occured 
            - tb: path of tensorboard logs
            - vtk: pvd/vtu files 
            - config.toml: save all the configs in `toml` format
            - tensorboard.sh: a script to start tensorboard
            - vec_configs.jl: save the `vec_configs` code to be reused directly

"""
function run_with_configs(f, vec_configs, comments; prefix=".", max_it= 100)
    base_config, appended_config_arr = parse_vec_configs(vec_configs)

    data_path = joinpath(prefix, "data")
    make_path(data_path, 0o751)

    path = joinpath(data_path, Dates.format(now(), "yyyy-mm-ddTHH_MM_SS"))
    make_path(path, 0o751)

    open(joinpath(path, "vec_configs.jl"), "w") do io
        println(io, vec_configs)
        flush(io)
    end
    
    dict_info = Dict(
        "LOGNAME" => ENV["LOGNAME"],
        "PID" => getpid(),
        "NPROCS" => nprocs(),
        "UID" => parse(Int, readchomp(`id -u`)),
        "GID" => parse(Int, readchomp(`id -g`)),
        "COMMENTS" => comments,
    )
    haskey(ENV, "HOSTNAME") && push!(dict_info, "HOSTNAME" => ENV["HOSTNAME"])
    try 
        push!(dict_info, "COMMIT" => readchomp(`git rev-parse --short HEAD`))
    catch; end

    open(joinpath(path, "config.toml"), "w") do io
        dict_vec_config = Dict(vec_configs)
        out = Dict(
            "info" => dict_info,
            "vec_configs" => dict_vec_config,
        )
        TOML.print(io, out) do el 
            if el isa Symbol
                return String(el)
            end
            return el
        end
        flush(io)
    end

    tb_path_prefix = joinpath(path, "tb")
    make_path(tb_path_prefix, 0o753)
    sh_file_name = joinpath(path, "tensorboard.sh")
    touch(sh_file_name)
    chmod(sh_file_name, 0o755)
    open(sh_file_name, "w") do io
        println(io, "#!/bin/bash")
        println(io, "tensorboard --logdir=./tb --port=\$1 --samples_per_plugin=images=$(max_it+1)")
        flush(io)
    end

    vtk_prefix = joinpath(path, "vtk")
    make_path(vtk_prefix, 0o753)

    err_prefix = joinpath(path, "errors")
    make_path(err_prefix, 0o753)

    function singlerun(config, i, multi_config)
        tb_file_prefix = joinpath(tb_path_prefix, "run_$i")
        tb_lg = TBLogger(tb_file_prefix, tb_overwrite, min_level=Logging.Info)

        if !isempty(multi_config)
            write_hparams!(tb_lg, multi_config, ["energy/E"])
        end

        local_dict_info = Dict(
            "LOGNAME" => ENV["LOGNAME"],
            "PID" => getpid(),
            "UID" => parse(Int, readchomp(`id -u`)),
            "GID" => parse(Int, readchomp(`id -g`)),
            "WORKER" => myid(),
        )   
        haskey(ENV, "HOSTNAME") && push!(local_dict_info, "HOSTNAME" => ENV["HOSTNAME"])
        with_logger(tb_lg) do 
            @info "host" base=TBText(local_dict_info) log_step_increment=0
        end

        vtk_file_prefix = joinpath(vtk_prefix, "run_$(i)_")
        vtk_file_pvd = createpvd(vtk_file_prefix)
        try
            f(config, vtk_file_prefix, vtk_file_pvd, tb_lg, i)    
        catch e
            open(joinpath(err_prefix, "run_$i.err"), "w") do io
                println(io, current_exceptions())
                flush(io)
            end
        finally
            close(tb_lg)
            savepvd(vtk_file_pvd)
            nothing
        end
    end

    if isempty(appended_config_arr)
        singlerun(base_config, 1, [])
    else
        pmap(eachindex(appended_config_arr)) do i
            multi_config = Dict{String, Any}(appended_config_arr[i])
            config = merge(base_config, multi_config)
            singlerun(config, i, multi_config)
        end
    end
    return nothing
end

function parse_vec_configs(vec_configs::Vector)
    multi_config_pairs = filter(((_, v),) -> isa(v, Vector), vec_configs) 
    single_config_pairs = filter(((_, v),) -> !isa(v, Vector), vec_configs)
        
    base_config = Dict(single_config_pairs)
    broadcasted_config_pairs = map(((k, v),) -> broadcast(Pair, k, v), multi_config_pairs)
    appended_config_arr = Iterators.product(broadcasted_config_pairs...) |> collect

    return base_config, appended_config_arr
end

# make TBLogger to be able to log the a dictionary as table.
function TensorBoardLogger.markdown_repr(d::Dict{String})
    pairs = collect(d)
    headers = join(["<th>$(first(x))</th>" for x in pairs], "")
    vals = join(["<td>$(last(x))</td>" for x in pairs], "")
    txt = """
    <table>
        <thead>
            <tr>
                $headers
            </tr>
        </thead>
        <tbody>
            <tr>
                $vals
            </tr>
        </tbody>
    </table>
    """
    return txt
end

"""
    make_path(path::AbstractString, mode::UInt16)
make a directory with `path` and set the mode of the directory to `mode`.
"""
function make_path(path::AbstractString, mode::UInt16)
    mkpath(path)
    chmod(path, mode)
end