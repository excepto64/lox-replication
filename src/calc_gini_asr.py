import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_gini = pd.read_csv("gini_attack.csv").set_index("option")
df_asr = pd.read_csv("asr_attack.csv").set_index("option")

print(df_gini)

df_joined = df_gini.join(df_asr, "option", "outer")
df_joined["del_ASR"] = df_joined["after_mean"] - df_joined["before_mean"]
df_joined["del_ASR_std"] = np.sqrt(df_joined["after_std"]**2 + df_joined["before_std"]**2)

print(df_joined)

df_plot = df_joined.reset_index()

series_list = sorted(df_plot["series"].unique())
option_list = sorted(df_plot["option"].unique())

colors = ["tab:blue", "tab:red", "gold"]
color_map = dict(zip(series_list, colors))

markers = ["o", "s", "o", "s", "v", "P", "X", "*"]
marker_map = dict(zip(option_list, markers))

for prefix in ["dpo", "sft"]:
    df_subset = df_plot[df_plot["option"].str.startswith(prefix)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for (series, option), group in df_subset.groupby(["series", "option"]):
        ax.errorbar(
            group["value"],
            group["del_ASR"],
            xerr=group["std"],
            yerr=group["del_ASR_std"],
            fmt=marker_map[option],
            color=color_map[series],
            label=f"{series} / {option}",
            capsize=3,
            markersize=8,
            linestyle="none",
        )

    ax.set_xlabel("Gini value")
    ax.set_ylabel("del_ASR")
    ax.set_title(f"Gini vs del_ASR ({prefix.upper()})")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"gini_vs_del_asr_{prefix}.pdf")