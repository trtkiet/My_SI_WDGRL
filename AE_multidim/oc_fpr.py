from util import *
import torch
from multiprocessing import Pool
import os
from model import WDGRL, AutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import time

def run_fpr(self):
    _, wdgrl, ae = self
    # Create a new instance of the WDGRL model (same architecture as before)
    start = time.time()
    # Create a new instance of the WDGRL model (same architecture as before)
    ns, nt, d = 150, 25, 32
    mu_s, mu_t = 0, 2
    delta_s, delta_t = [0, 1, 2, 3, 4], [0]
    xs, ys = gen_data(mu_s, delta_s, ns, d)
    xt, yt = gen_data(mu_t, delta_t, nt, d)

    wdgrl.generator = wdgrl.generator.cuda().double()
    ae.net = ae.net.cuda().double()
    # wdgrl.generator = wdgrl.generator.double()
    # ae.net = ae.net.double()

    xs = torch.DoubleTensor(xs)
    ys = torch.LongTensor(ys)
    xt = torch.DoubleTensor(xt)
    yt = torch.LongTensor(yt)

    xs_hat = wdgrl.extract_feature(xs.cuda())
    xt_hat = wdgrl.extract_feature(xt.cuda())
    x_hat = torch.cat([xs_hat, xt_hat], dim=0)
    x_tilde = ae.forward(x_hat.to(ae.device))
    reconstruction_loss = ae.reconstruction_loss(x_hat)
    reconstruction_loss = [i.item() for i in reconstruction_loss]


    reconstruction_loss = np.asarray(reconstruction_loss)
    xs_hat = np.asarray(xs_hat.cpu())
    xt_hat = np.asarray(xt_hat.cpu())
    x_hat = np.asarray(x_hat.cpu())
    x_tilde = np.asarray(x_tilde.cpu())
    xs = np.asarray(xs.cpu())
    xt = np.asarray(xt.cpu())
    ys = np.asarray(ys.cpu())
    yt = np.asarray(yt.cpu())
    alpha = 0.05
    O = AE_AD(xs_hat, xt_hat, x_tilde, alpha)
    # print(O)
    if (len(O) == 0) or (len(O) == nt):
        return None, None
    true_O = []
    for i in O:
        if yt[i] == 1:
            true_O.append(i)
    yt_hat = np.zeros_like(yt)
    yt_hat[O] = 1
    Oc = list(np.where(yt_hat == 0)[0])
    X = np.vstack((xs.flatten().reshape((ns * d, 1)), xt.flatten().reshape((nt * d, 1))))
    j = np.random.choice(O)
    etj = np.zeros((nt * d, 1)) 
    for i in range(d):
        etj[j * d + i] = 1
    etOc = np.zeros((nt * d, 1))
    for i in Oc:
        for k in range(d):
            etOc[i * d + k] = 1
    s = np.zeros((ns * d + nt * d, 1))
    for i in range(d):
        testj = xt[j, i]
        testOc = (1/len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
        if np.sign(testj - testOc) == -1:
            etj[j * d + i] = -1
            for k in Oc:
                etOc[k * d + i] = -1
    etaj = np.vstack((np.zeros((ns * d, 1)), etj - (1/len(Oc))*etOc))
    etajTx = etaj.T.dot(X)
    
    # print(f'Anomaly indexes: {O}')
    # print(f'etajTX: {etajTx}')
    mu = np.vstack((np.full((ns * d,1), mu_s), np.full((nt * d,1), mu_t)))
    sigma = np.identity(ns * d + nt * d)
    etajTmu = etaj.T.dot(mu)
    etajTsigmaetaj = etaj.T.dot(sigma).dot(etaj)
    b = sigma.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
    a = (np.identity(ns * d + nt * d) - b.dot(etaj.T)).dot(X)
    itv = [-np.inf, np.inf]
    for i in range(d):
        testj = xt[j, i]
        testOc = (1/len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
        if (testj - testOc) < 0:
            itv = intersect(itv, solve_linear_inequality(a[j * d + i + ns * d] - (1/len(Oc))*np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))), b[j * d + i + ns * d] - (1/len(Oc))*np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc)))))
        else:
            itv = intersect(itv, solve_linear_inequality(-a[j * d + i + ns * d] + (1/len(Oc))*np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))), -b[j * d + i + ns * d] + (1/len(Oc))*np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc)))))
    threshold = 20 * np.sqrt(etajTsigmaetaj[0][0])
    threshold = [-threshold, threshold]
    # print(itv)
    # print(etajTsigmaetaj)
    itv = intersect(itv, get_ad_interval(X.reshape(ns+nt, -1), x_hat.reshape(ns+nt, -1), x_tilde.reshape(ns+nt, -1), reconstruction_loss, a.reshape(ns+nt, -1), b.reshape(ns+nt, -1), wdgrl, ae, alpha))
    CDF = truncated_cdf(etajTx[0][0], etajTmu[0][0], np.sqrt(etajTsigmaetaj[0][0]), itv[0], itv[1])
    p_value = 2 * min(CDF, 1 - CDF)
    # print(f'etajTx: {etajTx[0][0]}')
    # print(f'itv: {itv}')
    print(f'p-value: {p_value}')
    stop = time.time()
    print(f'Execution time: {stop - start}')
    return p_value, stop - start

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    d = 32
    generator_hidden_dims = [500, 100, 20]
    critic_hidden_dims = [100]
    wdgrl = WDGRL(input_dim=d, generator_hidden_dims=generator_hidden_dims, critic_hidden_dims=critic_hidden_dims)
    index = None
    with open("model/wdgrl_models.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line[:-1].split("/")
            if words[1] == str(generator_hidden_dims) and words[2] == str(critic_hidden_dims):
                index = i
                break
    if index is None:
        print("Model not found")
        exit()
    check_point = torch.load(f"model/wdgrl_{index}.pth", map_location=wdgrl.device, weights_only=True)
    wdgrl.generator.load_state_dict(check_point['generator_state_dict'])
    wdgrl.critic.load_state_dict(check_point['critic_state_dict'])
    wdgrl.generator = wdgrl.generator.cpu()

    input_dim = generator_hidden_dims[-1]
    encoder_hidden_dims = [16, 8, 4, 2]
    decoder_hidden_dims = [4, 8, 16, input_dim]
    ae = AutoEncoder(input_dim=input_dim, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims)
    index = None
    with open("model/ae_models.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line[:-1].split("/")
            if words[1] == str(input_dim) and words[2] == str(encoder_hidden_dims) and words[3] == str(decoder_hidden_dims):
                index = i
                break
    if index is None:
        print("Model not found")
        exit()
    check_point = torch.load(f"model/ae_{index}.pth", map_location=ae.device, weights_only=True)
    ae.load_state_dict(check_point['state_dict'])
    ae.net = ae.net.cpu()

    max_iter = 1000
    alpha = 0.05
    list_wdgrl = [wdgrl for i in range(max_iter)]
    list_ae = [ae for i in range(max_iter)]
    reject = 0
    detect = 0
    list_p_value = []
    pool = Pool(initializer=np.random.seed)
    list_result = pool.map(run_fpr, zip(range(max_iter), list_wdgrl, list_ae))
    pool.close()
    pool.join()
    total_time = 0
    for p_value, runtime in list_result:
        if p_value is not None:
            detect += 1
            total_time += runtime
            list_p_value.append(p_value)

            if (p_value < alpha):
                reject += 1
    with open(f"results/fpr_oc.txt", "w") as f:
        f.write(str(reject/detect) + '\n')
        f.write(str(kstest(list_p_value, 'uniform')) + '\n')
        f.write(str(total_time) + '\n')
    plt.hist(list_p_value)
    plt.savefig(f'results/fpr_oc.png')
    plt.close()

