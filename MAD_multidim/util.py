from mpmath import mp
import numpy as np
import torch
from typing import List

mp.dps = 500
def gen_data(mu: float, delta: List[int], n: int, d: int):
    mu = np.full((n, d), mu, dtype=np.float64)
    noise = np.random.normal(loc = 0, scale = 1, size=(n, d))
    X = mu + noise
    labels = np.zeros(n)
    if len(delta) == 1:
        n_anomalies = int(n * 0.05)
        idx = np.random.choice(n, n_anomalies, replace=False)
        X[idx] = X[idx] + delta[0]
        if delta[0] != 0:
            labels[idx] = np.ones(n_anomalies)
    else:
        # In this case, we generate data for source domain.
        # 5% of the data is abnormal.
        # Anomalies are generated by randomly adding deltas to the data.
        n_anomalies = int(n * 0.05)
        idx = np.random.choice(n, n_anomalies, replace=False)
        if 0 in delta: 
            delta.pop(delta.index(0))
        if len(delta) != 0:
            split_points = sorted(np.random.choice(range(1, len(idx)), len(delta) - 1, replace=False))
            segments = np.split(idx, split_points)
            for i, segment in enumerate(segments):
                X[segment] = X[segment] + delta[i]
            labels[idx] = 1
    return X, labels

def intersect(itv1, itv2):
    # print(itv1, itv2)
    itv = [max(itv1[0], itv2[0]), min(itv1[1], itv2[1])]
    if itv[0] > itv[1]:
        return None    
    return itv

def solve_linear_inequality(u, v): #u + vz < 0
    u = float(u)
    v = float(v)
    if (v > -1e-16 and v < 1e-16):
        if (u <= 1e-7):
            return [-np.Inf, np.Inf]
        else:
            print('error', u, v)
            return None
    if (v < 0):
        return [-u/v, np.Inf]
    return [np.NINF, -u/v]

def get_dnn_interval(Xtj, a, b, model):
    layers = []

    for name, param in model.generator.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')

    ptr = 0
    itv = [np.NINF, np.Inf]
    u = a
    v = b
    temp = Xtj
    weight = None
    bias = None
    for name, param in model.generator.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = param.data.cpu().detach().numpy()
            elif ('bias' in name):
                bias = param.data.cpu().detach().numpy().reshape(-1, 1)
                ptr += 1
                temp = weight.dot(temp) + bias
                u = weight.dot(u) + bias
                v = weight.dot(v)

        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            ptr += 1
            Relu_matrix = np.zeros((temp.shape[0], temp.shape[0]))
            sub_itv = [np.NINF, np.inf]
            for i in range(temp.shape[0]):
                if temp[i] > 0:
                    Relu_matrix[i][i] = 1
                    sub_itv = intersect(sub_itv, solve_linear_inequality(-u[i], -v[i]))
                else:
                    sub_itv = intersect(sub_itv, solve_linear_inequality(u[i], v[i]))
            itv = intersect(itv, sub_itv)
            temp = Relu_matrix.dot(temp)
            u = Relu_matrix.dot(u)
            v = Relu_matrix.dot(v)
    return itv, u[:, 0], v[:, 0]

def median(a):
    return np.argsort(a)[len(a) // 2]

def MAD_AD(Xs, Xt, alpha):
    O = []
    X = np.concatenate((Xs, Xt), axis=0)
    for i in range(X.shape[1]):
        median1 = median(X[:, i])
        absolute_deviation = np.abs(X[:, i] - X[median1, i])
        median2 = median(absolute_deviation)
        
        lower = X[median1, i] - alpha*absolute_deviation[median2]
        upper = X[median1, i] + alpha*absolute_deviation[median2]
        for j in range(Xt.shape[0]):
            if j not in O and ((Xt[j, i] < lower) or (Xt[j, i] > upper)):
                O.append(j)
    return np.sort(O)

def get_ad_interval(X, X_hat, ns, nt, O, a, b, model, alpha):
    itv = [np.NINF, np.Inf]
    u = np.zeros((X.shape[0], X_hat.shape[1]))
    v = np.zeros((X.shape[0], X_hat.shape[1]))
    # print(u.shape, v.shape)
    O = []
    d = X.shape[1]
    for i in range(X_hat.shape[0]):
        sub_itv, u[i], v[i] = get_dnn_interval(X[i].reshape(-1, d).T, a[i].reshape(-1, d).T, b[i].reshape(-1, d).T, model)
        itv = intersect(itv, sub_itv)
    # print(u, v)
    # print(itv)
    sub_itv = [np.NINF, np.inf]
    for d in range(X_hat.shape[1]):
        k1 = median(X_hat[:, d])
        for i in range(X_hat.shape[0]):
            if X_hat[i, d] <= X_hat[k1, d]:
                sub_itv = intersect(sub_itv, solve_linear_inequality(u[i, d] - u[k1, d], v[i, d] - v[k1, d]))
            else:
                sub_itv = intersect(sub_itv, solve_linear_inequality(u[k1, d] - u[i, d], v[k1, d] - v[i, d]))

        dev = np.abs(X_hat[:, d] - X_hat[k1, d])
        k2 = median(dev)
        sk2 = np.sign(X_hat[k2, d] - X_hat[k1, d])
        for j in range(X_hat.shape[0]):
            if abs(X_hat[j, d] - X_hat[k1, d]) <= abs(X_hat[k2, d] - X_hat[k1, d]):
                sj = np.sign(X_hat[j, d] - X_hat[k1, d])
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    sj * (u[j, d] - u[k1, d]) - sk2 * (u[k2, d] - u[k1, d]), 
                    sj * (v[j, d] - v[k1, d]) - sk2 * (v[k2, d] - v[k1, d])))
            else:
                sj = np.sign(X_hat[j, d] - X_hat[k1, d])
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    -sj * (u[j, d] - u[k1, d]) + sk2 * (u[k2, d] - u[k1, d]), 
                    -sj * (v[j, d] - v[k1, d]) + sk2 * (v[k2, d] - v[k1, d])))

        upper = X_hat[k1, d] + alpha * dev[k2]
        lower = X_hat[k1, d] - alpha * dev[k2]
        for j in range(ns, X.shape[0]):
            if (X_hat[j, d] < lower):
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    u[j, d] - u[k1, d] + alpha * sk2 * (u[k2, d] - u[k1, d]),
                    v[j, d] - v[k1, d] + alpha * sk2 * (v[k2, d] - v[k1, d])
                ))
            else:
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    -(u[j, d] - u[k1, d] + alpha * sk2 * (u[k2, d] - u[k1, d])),
                    -(v[j, d] - v[k1, d] + alpha * sk2 * (v[k2, d] - v[k1, d]))
                ))
            if (X_hat[j, d] > upper):
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    -(u[j, d] - u[k1, d] - alpha * sk2 * (u[k2, d] - u[k1, d])),
                    -(v[j, d] - v[k1, d] - alpha * sk2 * (v[k2, d] - v[k1, d]))
                ))
            else:
                sub_itv = intersect(sub_itv, solve_linear_inequality(
                    (u[j, d] - u[k1, d] - alpha * sk2 * (u[k2, d] - u[k1, d])),
                    (v[j, d] - v[k1, d] - alpha * sk2 * (v[k2, d] - v[k1, d]))
                ))
    # print(f'mad itv: {sub_itv}')
    # print(f'dnn itv: {itv}')
    itv = intersect(itv, sub_itv)
    return itv

def compute_yz(X, etaj, zk):
    sq_norm = (np.linalg.norm(etaj))**2

    e1 = np.identity(X.shape[0]) - (np.dot(etaj, etaj.T))/sq_norm
    a = np.dot(e1, X)

    b = etaj/sq_norm

    Xz = a + b*zk

    return Xz, a, b

def max_sum(X):
    return 0

def parametric_wdgrl(Xz, a, b, zk, model, ns, nt, alpha):
    Xz = Xz.reshape(ns + nt, Xz.shape[0] // (ns + nt))
    a = a.reshape(ns + nt, a.shape[0] // (ns + nt))
    b = b.reshape(ns + nt, b.shape[0] // (ns + nt))
    Xz = torch.DoubleTensor(Xz)
    Xz_hat = model.extract_feature(Xz.cuda()).cpu().numpy()
    Xzs_hat = Xz_hat[:ns]
    Xzt_hat = Xz_hat[ns:]
    Oz = MAD_AD(Xzs_hat, Xzt_hat, alpha)
    itv = get_ad_interval(Xz, Xz_hat, ns, nt, Oz, a, b, model, alpha)
    if (itv[0] > zk):
        print(f'error{itv[0] - zk}')
    return itv[1] - min(zk, itv[1]), Oz


def run_parametric_wdgrl(X, etaj, n, threshold, model, ns, nt, alpha):
    zk = -threshold

    list_zk = [zk]
    list_Oz = []

    while zk < threshold:
        Xz, a, b = compute_yz(X, etaj, zk)
        skz, Oz = parametric_wdgrl(Xz, a, b, zk, model, ns, nt, alpha)
        zk = zk + skz + 1e-3 
        # zk = min(zk, threshold)
        list_zk.append(zk)
        list_Oz.append(Oz)
        # print(f'intervals: {zk-skz-1e-3} - {zk -1e-3}')
        # print(f'Anomaly index: {Oz}')
        # print('-------------')
    return list_zk, list_Oz
        
def cdf(mu, sigma, list_zk, list_Oz, etajTX, O, constraint):
    numerator = 0
    denominator = 0
    cnt = 0
    for each_interval in range(len(list_zk) - 1):
        al = list_zk[each_interval]
        ar = list_zk[each_interval + 1] - 1e-3

        al = max(al, constraint[0])
        ar = min(ar, constraint[1])
        if (al > ar):
            continue

        # print(f'observed O: {O}')
        # print(f'list_Oz: {list_Oz[each_interval]}')
        if (np.array_equal(O, list_Oz[each_interval]) == False):
            continue
        # print(f'interval: {al} - {ar}')
        cnt += 1
        denominator = denominator + mp.ncdf((ar - mu)/sigma) - mp.ncdf((al - mu)/sigma)
        if etajTX >= ar:
            numerator = numerator + mp.ncdf((ar - mu)/sigma) - mp.ncdf((al - mu)/sigma)
        elif (etajTX >= al) and (etajTX< ar):
            numerator = numerator + mp.ncdf((etajTX - mu)/sigma) - mp.ncdf((al - mu)/sigma)
    # print(f'numerator: {numerator}')
    # print(f'denominator: {denominator}')
    # print('cnt: ', cnt)
        # if numerator == 0:
        #     print('con loi ne')
        #     print(f'etajTx: {etajTX}')
        #     for each_interval in range(len(list_zk) - 1):
        #         al = list_zk[each_interval]
        #         ar = list_zk[each_interval + 1] - 1e-3
        #         print(f'interval: {al} - {ar}')
    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None

def truncated_cdf(etajTy, mu, sigma, left, right):
    numerator = mp.ncdf((etajTy - mu) / sigma) - mp.ncdf((left - mu) / sigma)
    denominator = mp.ncdf((right - mu) / sigma) - mp.ncdf((left - mu) / sigma)
    # print(f'numerator: {numerator}')
    # print(f'denominator: {denominator}')
    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None