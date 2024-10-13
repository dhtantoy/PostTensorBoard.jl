using PostTensorBoard

# ---- output pdf ----
path = "/export/home/tandh/JlProjects/HeatFlowTopOpt/data/2024-09-24T17_52_08"
tags = ["energy/E" "energy/Jt"]
key = ["InitType"]
ex_tags = String[]
output_with_keys([path], key, tags; ex_tags= ex_tags)

# ---- output mp4 ----
tb_path = joinpath(path, "tb")
runs = [1]
domain2mp4(tb_path, runs)

# ---- output domain ----
run_path = "/export/home/tandh/JlProjects/HeatFlowTopOpt/data/2024-10-12T01_12_01/tb/run_7" 
r = get_imgs(run_path, 3)
