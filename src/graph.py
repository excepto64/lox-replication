import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--n-main", type=int, default=2048) # Number of main singular values to consider for extrapolation.
parser.add_argument("--n-sec", type=int, default=0) # Number of extra singular values to consider for extrapolation.

args = parser.parse_args()
model_local = args.model.split('/')[-1]

def find_cum(coeff):
    total = torch.sum(coeff**2)
    cum = torch.cumsum(coeff**2, dim = 0) / total
    return cum

def plot_cum(svd_coeffs):
    n_main = 0
    cum_main = torch.zeros(args.n_main)
    
    plt.figure(0, figsize=(10, 6))

    if args.n_sec > 0:
        n_sec = 0
        cum_sec = torch.zeros(args.n_sec)

    for coeff in svd_coeffs:
        cum = find_cum(coeff)
        if coeff.size()[0] == args.n_main:
            n_main += 1
            cum_main += cum
        elif coeff.size()[0] == args.n_sec and args.n_sec > 0:
            n_sec += 1
            cum_sec += cum

    cum_main /= n_main
    torch.save(cum_main, f"cum_main_{model_local}.pt")
    print(torch.where(cum_main > 0.8)[0][0], cum_main[10])

    if args.n_sec > 0:
        cum_sec /= n_sec
        torch.save(cum_sec, f"cum_sec_{model_local}.pt")
        print(torch.where(cum_sec > 0.8)[0][0], cum_sec[10])
        plt.plot(cum_sec.numpy(), label = "Secondary")

    plt.plot(cum_main.numpy(), label = "Main")
    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.savefig(f"cumulative_proportion_{model_local}.pdf")

def plot_svd(svd_coeffs):
    n_main = 0
    sum_main = torch.zeros(args.n_main)

    plt.figure(1, figsize=(10, 6))
    
    if args.n_sec > 0:
        n_sec = 0
        sum_sec = torch.zeros(args.n_sec)

    for coeff in svd_coeffs:
        if coeff.size()[0] == args.n_main:
            n_main += 1
            sum_main += coeff
        elif coeff.size()[0] == args.n_sec and args.n_sec > 0:
            n_sec += 1
            sum_sec += coeff

    sum_main /= n_main
    torch.save(sum_main, f"sum_main_{model_local}.pt")

    if args.n_sec > 0:
        sum_sec /= n_sec
        torch.save(sum_sec, f"sum_sec_{model_local}.pt")
        plt.plot(sum_sec.numpy(), label = "Secondary", color = "blue", linewidth = 2, marker = "o", markersize = 4)

    plt.plot(sum_main.numpy(), label = "Main", color = "orange", linewidth = 2, marker = "o", markersize = 4)
    plt.xscale("log")
    plt.xlabel("Singular Values")
    plt.ylabel("Average Singular Value")
    plt.legend()
    plt.savefig(f"average_singular_value_{model_local}.pdf")

def main():
    svd_coeffs = torch.load(f"SVD_coeffs_{model_local}.pt", weights_only = True)
    
    plot_cum(svd_coeffs)

    plot_svd(svd_coeffs)

if __name__ == "__main__":
    main()