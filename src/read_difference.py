import numpy as np

with open("kl_out.csv", "r") as f:
    lines = f.readlines()
    full = [float(line.split(",")[-1].strip()) for line in lines[1:6]]
    extracted = [float(line.split(",")[-1].strip()) for line in lines[6:11]]
    names = [line.split(",")[2].strip() for line in lines[1:6]]

nparray_full = np.array(full)
nparray_extracted = np.array(extracted)

for i, name in enumerate(names):
    print(f"{name}: {nparray_extracted[i] / nparray_full[i] }")