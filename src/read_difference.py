import numpy as np
import pandas as pd

with open("kl_out.csv", "r") as f:
    lines = f.readlines()
    full = [float(line.split(",")[-1].strip()) for line in lines[1:6]]
    extracted = [float(line.split(",")[-1].strip()) for line in lines[6:11]]
    between = [float(line.split(",")[-1].strip()) for line in lines[11:16]]
    names = [line.split(",")[2].strip() for line in lines[1:6]]

nparray_full = np.array(full)
nparray_extracted = np.array(extracted)
nparray_between = np.array(between)

df = pd.DataFrame({
    "name": names,
    "full": nparray_full,
    "extracted": nparray_extracted,
    "between": nparray_between,
    "extracted/full": nparray_extracted / nparray_full
})

# for i, name in enumerate(names):
#     print(f"{name}: {nparray_extracted[i] / nparray_full[i] }")
print(df)