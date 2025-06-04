import os
import shutil

dirs = []

def gen_file_with_ratio(ratio):
    print("generate input file for ratio ", ratio)
    out_dir = "hfmix" + ratio
    os.makedirs(out_dir, exist_ok=True)
    dirs.append(out_dir)
    with open("in.lj_longstep_basic", "r") as f:
        lines = f.readlines()
    out_path = os.path.join(out_dir, "in.lj_longstep") 
    with open(out_path, "w") as f:
        for line in lines:
            if line.startswith("fhcut"):
                f.write(line.replace("0.6", ratio))
            else:
                f.write(line)
    script_path = os.path.join(out_dir, "run.sh") 
    shutil.copy("run.sh", script_path)

for i in range(1, 10):
    gen_file_with_ratio("0." + str(i))

# with open("run_all.sh", "w") as f:
#     f.write("#!/bin/bash\n")
#     for cur_dir in dirs:
#         f.write("cd {}\n".format(cur_dir))
#         f.write("./run.sh\n")
#         f.write("cd ..\n")
