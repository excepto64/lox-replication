import torch

def find_cum(coeff):
    total = torch.sum(coeff**2)
    cum = 1 - torch.cumsum(coeff**2, dim = 0) / total
    return cum

def main():
    svd_coeffs = torch.load("SVD_coeffs.pt", weights_only = True)

    n_320 = 0
    cum_320 = torch.zeros(321)
    n_960 = 0
    cum_960 = torch.zeros(961)

    for coeff in svd_coeffs:
        cum = find_cum(coeff)
        if coeff.size() == 320:
            n_320 += 1
            cum_320[1:] += cum
        elif coeff.size() == 960:
            n_960 += 1
            cum_960[1:] += cum

    cum_320 /= n_320
    cum_960 /= n_960
    torch.save(cum_320, "cum_320.pt")
    torch.save(cum_960, "cum_960.pt")

if __name__ == "__main__":
    main()