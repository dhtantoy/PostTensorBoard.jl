using PostTensorBoard

# All output data will be stored at the original path.

# ---- output pdf ----
path = "/export/home/tandh/JlProjects/HeatFlowTopOpt/data/2024-11-09T12_17_47"
tags = ["energy/J" "energy/Jt" "energy/Ju"]
key = ["ks"]
ex_tags = String[]
output_with_keys([path], key, tags; ex_tags= ex_tags)

# ---- output mp4 ----
tb_path = joinpath(path, "tb")
runs = [1]
domain2mp4(tb_path, runs)

# ---- output domain ----
run_path = "/export/home/tandh/JlProjects/HeatFlowTopOpt/data/2024-11-09T12_17_47/tb/run_7" 
r = get_mat(run_path, 3)
