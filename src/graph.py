import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n-main", type=int, default=2048) # Number of main singular values to consider for extrapolation.
parser.add_argument("--n-sec", type=int, default=512) # Number of extra singular values to consider for extrapolation.

args = parser.parse_args()
print(args)

def find_cum(coeff):
    total = torch.sum(coeff**2)
    cum = torch.cumsum(coeff**2, dim = 0) / total
    return cum

def main():
    svd_coeffs = torch.load("SVD_coeffs.pt", weights_only = True)
    
    n_main = 0
    cum_main = torch.zeros(args.n_main)
    
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
    torch.save(cum_main, "cum_main.pt")
    print(torch.where(cum_main > 0.8)[0][0], cum_main[10])

    if args.n_sec > 0:
        cum_sec /= n_sec
        torch.save(cum_sec, "cum_sec.pt")
        print(torch.where(cum_sec > 0.8)[0][0], cum_sec[10])
        plt.plot(cum_sec.numpy(), label = "Secondary")

    plt.plot(cum_main.numpy(), label = "Main")
    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    #plt.show()
    plt.savefig("cumulative_proportion.pdf")

if __name__ == "__main__":
    main()