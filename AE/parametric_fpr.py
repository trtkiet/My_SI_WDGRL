from util import *
import torch
from multiprocessing import Pool
import os
from model import WDGRL, AutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

def run_fpr(self):
    # np.random.seed(52907)
    _, wdgrl, ae = self
    # Create a new instance of the WDGRL model (same architecture as before)
    ns, nt, d = 150, 25, 1
    mu_s, mu_t = 0, 2
    delta_s, delta_t = [0, 1, 2, 3, 4], [0]
    xs, ys = gen_data(mu_s, delta_s, ns, d)
    xt, yt = gen_data(mu_t, delta_t, nt, d)

    Model.generator = Model.generator.cuda()
    ae.net = ae.net.cuda()

    xs = torch.FloatTensor(xs)
    ys = torch.LongTensor(ys)
    xt = torch.FloatTensor(xt)
    yt = torch.LongTensor(yt)

    xs_hat = Model.extract_feature(xs.cuda())
    xt_hat = Model.extract_feature(xt.cuda())
    x_hat = torch.cat([xs_hat, xt_hat], dim=0)

    xs_hat = xs_hat.cpu()
    xt_hat = xt_hat.cpu()
    x_hat = x_hat.cpu()
    xs = xs.cpu()
    xt = xt.cpu()
    ys = ys.cpu()
    yt = yt.cpu()
    alpha = 0.05
    O = AE_AD(xs_hat, xt_hat, ae, alpha)
    X_tilde = ae.forward(x_hat.to(ae.device))
    reconstruction_loss = ae.reconstruction_loss(x_hat.to(ae.device))
    reconstruction_loss = [i.item() for i in reconstruction_loss]
    # print(O)
    if (len(O) == 0) or (len(O) == nt):
        return None
    yt_hat = torch.zeros_like(yt)
    yt_hat[O] = 1
    Oc = list(np.where(yt_hat == 0)[0])
    X = np.vstack((xs, xt))
    X = torch.FloatTensor(X)
    j = np.random.choice(O)
    etj = np.zeros((nt, 1))
    etj[j][0] = 1
    etOc = np.zeros((nt, 1))
    etOc[Oc] = 1
    etaj = np.vstack((np.zeros((ns, 1)), etj-(1/len(Oc))*etOc))

    etajTx = etaj.T.dot(X)
    
    # print(f'Anomaly indexes: {O}')
    print(f'etajTX: {etajTx}')
    mu = np.vstack((np.full((ns,1), mu_s), np.full((nt,1), mu_t)))
    sigma = np.identity(ns+nt)
    etajTmu = etaj.T.dot(mu)
    etajTsigmaetaj = etaj.T.dot(sigma).dot(etaj)
    b = sigma.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
    a = (np.identity(ns+nt) - b.dot(etaj.T)).dot(X)
    threshold = 20
    list_zk, list_Oz = run_parametric_si(X, etaj, ns+nt, threshold, wdgrl, ae, alpha)
    CDF = cdf(etajTmu[0][0], np.sqrt(etajTsigmaetaj[0][0]), list_zk, list_Oz, etajTx[0][0], O)
    p_value = 2 * min(CDF, 1 - CDF)
    print(f'p-value: {p_value}')
    # print('--------------------------')
    return p_value

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    d = 1
    generator_hidden_dims = [4, 4, 2]
    critic_hidden_dims = [4, 2, 1]
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
    encoder_hidden_dims = [2, 2, 1]
    decoder_hidden_dims = [2, 2]
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

    max_iter = 2000
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

    for p_value in list_result:
        if p_value is not None:
            detect += 1
            list_p_value.append(p_value)

            if (p_value < alpha):
                reject += 1
    with open(f"results/fpr_parametric.txt", "w") as f:
        f.write(str(reject/detect) + '\n')
        f.write(str(kstest(list_p_value, 'uniform')) + '\n')
    plt.hist(list_p_value)
    plt.savefig(f'results/fpr_parametric.png')
    plt.close()

