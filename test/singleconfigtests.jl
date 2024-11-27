
# using Distributed

using PostTensorBoard
using Gridap
using Gridap.CellData

function f(config, vtk_file_prefix, vtk_file_pvd, tb_lg, i)
    model = CartesianDiscreteModel((0, 1, 0, 1), (10, 10))
    trian = Triangulation(model)

    @info "$i th is running"
    for it in 1:5
        uh = CellField(rand(num_cells(trian)), trian)
        vtk_file_pvd[it] = createvtk(trian, vtk_file_prefix * string(it), cellfields= ["uh" => uh])
    end
    l2_norm = 0.1 # some integration
    h1_norm = 0.2
    with_logger(tb_lg) do 
        @info "norm" L2= l2_norm log_step_increment=0
        @info "norm" H1= h1_norm log_step_increment=1
    end
end

vec_configs = [
]

comments = "some comments"

run_with_configs(f, vec_configs, comments, max_it= 100)

# if you want to view the tensorboard log data,
# then cd to data/yyyy-mm-ddTHH_MM_SS, 
# and run code `./tensorboard.sh port` in some shell, 
# where `port` is some port number, such as 6008, 6009