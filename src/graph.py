import torch
import matplotlib.pyplot as plt

def find_cum(coeff):
    total = torch.sum(coeff**2)
    cum = torch.cumsum(coeff**2, dim = 0) / total
    return cum

def main():
    svd_coeffs = torch.load("SVD_coeffs.pt", weights_only = True)

    n_320 = 0
    cum_320 = torch.zeros(320)
    n_960 = 0
    cum_960 = torch.zeros(960)


    for coeff in svd_coeffs:
        cum = find_cum(coeff)
        if coeff.size()[0] == 320:
            n_320 += 1
            cum_320 += cum
        elif coeff.size()[0] == 960:
            n_960 += 1
            cum_960 += cum

    cum_320 /= n_320
    cum_960 /= n_960
    torch.save(cum_320, "cum_320.pt")
    torch.save(cum_960, "cum_960.pt")

    print(torch.where(cum_320 > 0.8)[0][0], cum_320[10])
    print(torch.where(cum_960 > 0.8)[0][0], cum_960[10])

    plt.plot(cum_320.numpy(), label = "320")
    plt.plot(cum_960.numpy(), label = "960")
    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    #plt.show()
    plt.savefig("cumulative_proportion.pdf")

if __name__ == "__main__":
    main()