import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasource import download
from datasource.loaders import mnist_loader

from vis import scatter


def P_ji(distance, beta):
    """Calculate the Gaussian distribution of all js surrounding i

    # Arguments
        distance [1D array]: the squared distance between all js and i
        beta [float]: the inverted variance

    # Returns
        [1D array]: pair-wise similarity that has shape n_elements
        [float]: the H(pi) obtained (used for outer process to search for sigma)
    """
    p = torch.exp(-distance * beta)
    total = p.sum() + 1e-8

    h = torch.log(total) + beta * torch.sum(distance * p) / total
    p = p / total

    return p, h.item()


def pca(X, n_dims=2):
    """Perform PCA

    # Arguments
        X [2D array]: the data points, with shape [n_elements, n_features]
        n_dims [int]: the number of dimensions to keep

    # Returns
        [2D array]: the reduced representation
    """
    X = X - torch.mean(X, 0).expand_as(X)
    u, _, _ = torch.svd(X.t())

    return torch.mm(X, u[:, :n_dims])


def similarity(X, perplexity):
    """Get similarity measure given constrained perplexity

    # Arguments
        X [2D array]: the data points, with shape [n_elements, n_features]
        perplexity [float]: the perplexity to determine sigma

    # Returns
        [2D array]: the similarity matrix (asymmetric)
    """
    n, d = X.shape

    sum_X = (X ** 2).sum(dim=1).unsqueeze(-1)
    distance = (-2 * torch.mm(X, X.t()) + sum_X).t() + sum_X
    P = torch.zeros((n, n))
    beta = torch.ones((n, 1))
    logU = torch.log(torch.Tensor([perplexity])).item()

    for i in range(n):

        # search for sigma
        betamin = float("-inf")
        betamax = float("inf")
        Di = distance[i, list(range(i)) + list(range(i + 1, n))]
        currentP, H = P_ji(Di, beta[i])

        # check if beta value yields close enough perplexity
        Hdiff = H - logU
        tries = 0
        while torch.abs(torch.Tensor([Hdiff])).item() > 1e-4 and tries < 50:

            # increase or decrease beta
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == float("inf") or betamax == float("-inf"):
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i]
                if betamin == float("inf") or betamin == float("-inf"):
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            currentP, H = P_ji(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # save the best P
        P[i, list(range(i)) + list(range(i+1, n))] = currentP

    return P


def tsne(X, n_dims=2, perplexity=20.0, label=None):
    """Perform tsne on dataset X

    # Arguments
        X [2D array]: the data points, with shape [n_elements, n_features]
        n_dims [int]: the number of dimensions to keep
        perplexity [float]: the perplexity to determine sigma
        label [1D array]: the label info of shape [n_elements]. If set, will draw plot

    # Returns
        [2D array]: the reduced representation
    """
    X = pca(X, 50)

    P = similarity(X, perplexity)
    P = P + P.t()
    P = P / P.sum(1).unsqueeze(-1)
    P = torch.max(P, torch.ones(P.shape) * 1e-8)

    y = torch.randn(X.shape[0], n_dims)
    y.requires_grad_()

    if label is not None:
        figs = scatter(y.cpu().data.numpy(), label)
        figs[0].savefig('images/before.png')

    criterion = nn.KLDivLoss(reduction='sum')
    optimizer = optim.SGD([y], lr=30, momentum=0.5)

    random_idx = np.random.permutation(X.shape[0])
    for _idx in range(1000):
        optimizer.zero_grad()

        # calculate low-dim piecewise similarity
        sum_y = (y ** 2).sum(1).unsqueeze(-1)
        num = -2. * torch.mm(y, y.t())
        num = 1. / (1. + ((num + sum_y).t() + sum_y))
        num[range(num.shape[0]), range(num.shape[1])] = 0.
        Q = num / num.sum(1).unsqueeze(-1)
        Q = torch.max(Q, torch.ones(Q.shape) * 1e-12)
        Q = torch.log(Q)

        loss = criterion(Q, P)
        loss.backward()
        optimizer.step()

        if _idx % 100 == 0:
            print(f'[{_idx}|1000] {loss.item()},',
                  f'{y.grad.mean().item()}, {y.sum().item()}')

    if label is not None:
        figs = scatter(y.cpu().data.numpy(), label)
        figs[0].savefig('images/after.png')

    return y


if __name__ == "__main__":
    """Run the main program"""
    download("mnist", "datasets/mnist")
    X_train, y_train = mnist_loader("datasets/mnist", train=True)
    X_train = X_train.astype(np.float32) / 255
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train = torch.FloatTensor(X_train)

    n_data = 200
    X_train, y_train = X_train[:n_data], y_train[:n_data]

    similar = pca(X_train)
    temp = scatter(similar.cpu().data.numpy(), y_train)
    temp[0].savefig('images/pca.png')

    similar = tsne(X_train, 2, 5.0, y_train)
    temp = scatter(similar.cpu().data.numpy(), y_train)
