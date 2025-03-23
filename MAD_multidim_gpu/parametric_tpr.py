from util import *
import torch
from multiprocessing import Pool
import os
from model import WDGRL
import numpy as np

def run_tpr(self):
    _, delta, Model = self
    # Create a new instance of the WDGRL model (same architecture as before)
    ns, nt, d = 150, 25, 16
    mu_s, mu_t = 0, 2
    delta_s, delta_t = [0, 1, 2, 3, 4], [delta]
    xs, ys = gen_data(mu_s, delta_s, ns, d)
    xt, yt = gen_data(mu_t, delta_t, nt, d)

    Model.generator = Model.generator.cuda().double()
    # Model.generator = Model.generator.double()

    xs = torch.DoubleTensor(xs)
    ys = torch.LongTensor(ys)
    xt = torch.DoubleTensor(xt)
    yt = torch.LongTensor(yt)
    xs = xs.to(Model.device)
    xt = xt.to(Model.device)
    xs_hat = Model.extract_feature(xs)
    xt_hat = Model.extract_feature(xt)
    x_hat = torch.cat([xs_hat, xt_hat], dim=0)
    alpha = 15
    O = MAD_AD(xs_hat, xt_hat, alpha)
    # print(len(O))
    if (len(O) == 0) or (len(O) == nt):
        return None
    yt_hat = torch.zeros_like(yt)
    yt_hat[O] = 1
    Oc = list(torch.where(yt_hat == 0)[0])
    X = torch.vstack((xs.flatten().reshape((ns * d, 1)), xt.flatten().reshape((nt * d, 1))))
    true_O = []
    for i in O:
        if yt[i] == 1:
            true_O.append(i)
    if len(true_O) == 0:
        return None
    j = np.random.choice(O)
    etj = torch.zeros((nt * d, 1), device=Model.device, dtype=torch.double) 
    for i in range(d):
        etj[j * d + i] = 1
    etOc = torch.zeros((nt * d, 1), device=Model.device, dtype=torch.double)
    for i in Oc:
        for k in range(d):
            etOc[i * d + k] = 1
    for i in range(d):
        testj = X[j * d + i]
        testOc = (1/len(Oc)) * np.sum(X[Oc[k] * d + i] for k in range(len(Oc)))
        if torch.sign(testj - testOc) == -1:
            etj[j * d + i] = -1
            for k in Oc:
                etOc[k * d + i] = -1
    etaj = torch.vstack((torch.zeros((ns * d, 1), device=Model.device), etj - (1/len(Oc))*etOc))
    etajTx = etaj.T.matmul(X)
    
    print(f'Anomaly indexes: {O}')
    print(f'etajTX: {etajTx}')
    mu = torch.vstack((torch.full((ns * d,1), mu_s, device=Model.device, dtype=torch.double), torch.full((nt * d,1), mu_t, device=Model.device, dtype=torch.double)))
    sigma = torch.eye(ns * d + nt * d, device=Model.device, dtype=torch.double)
    etajTmu = etaj.T.matmul(mu)
    etajTsigmaetaj = etaj.T.matmul(sigma).matmul(etaj)
    b = sigma.matmul(etaj).matmul(torch.linalg.inv(etajTsigmaetaj))
    a = (torch.eye(ns * d + nt * d, device=Model.device, dtype=torch.double) - b.matmul(etaj.T)).matmul(X)
    itv = [-np.inf, np.inf]
    for i in range(d):
        testj = a[j * d + i]
        testOc = (1/len(Oc)) * np.sum(a[Oc[k] * d + i] for k in range(len(Oc)))
        if (testj - testOc) < 0:
            itv = intersect(itv, solve_linear_inequality(a[j * d + i] - (1/len(Oc))*np.sum(a[Oc[k] * d + i] for k in range(len(Oc))), b[j * d + i] - (1/len(Oc))*np.sum(b[Oc[k] * d + i] for k in range(len(Oc)))))
        else:
            itv = intersect(itv, solve_linear_inequality(-a[j * d + i] + (1/len(Oc))*np.sum(a[Oc[k] * d + i] for k in range(len(Oc))), -b[j * d + i] + (1/len(Oc))*np.sum(b[Oc[k] * d + i] for k in range(len(Oc)))))
    threshold = 20 * np.sqrt(etajTsigmaetaj[0][0].detach().cpu().numpy())
    threshold = min(threshold, max(np.abs(itv[0]), np.abs(itv[1])))
    # print(itv)
    # print(etajTsigmaetaj)
    list_zk, list_Oz = run_parametric_wdgrl(X, etaj, ns+nt, threshold, Model, ns, nt, alpha)
    CDF = cdf(etajTmu[0][0], np.sqrt(etajTsigmaetaj[0][0]), list_zk, list_Oz, etajTx[0][0], O, itv)
    p_value = 2 * min(CDF, 1 - CDF)
    print(f'p-value: {p_value}')
    return p_value

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    max_iter = 1
    alpha = 0.05
    list_tpr = []
    d = 16
    generator_hidden_dims = [32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16, 8, 4]
    critic_hidden_dims = [64, 64, 64, 64, 64, 32, 16, 8 , 4, 2, 1]
    Model = WDGRL(input_dim=d, generator_hidden_dims=generator_hidden_dims, critic_hidden_dims=critic_hidden_dims)
    index = None
    with open("model/models.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line[:-1].split("/")
            if words[1] == str(generator_hidden_dims) and words[2] == str(critic_hidden_dims):
                index = i
                break
    if index is None:
        print("Model not found")
        exit()
    check_point = torch.load(f"model/wdgrl_{index}.pth", map_location=Model.device, weights_only=True)
    Model.generator.load_state_dict(check_point['generator_state_dict'])
    Model.critic.load_state_dict(check_point['critic_state_dict'])
    Model.generator = Model.generator.cpu()
    list_model = [Model for _ in range(max_iter)] 
    with open('results/tpr_parametric.txt', 'w') as f:
        f.write('')
    for delta in reversed(range(4, 5)):
        reject = 0
        detect = 0
        list_p_value = []
        pool = Pool(initializer=np.random.seed, processes=1)
        list_result = pool.map(run_tpr, zip(range(max_iter),[delta]*max_iter, list_model))
        pool.close()
        pool.join()

        for p_value in list_result:
            if p_value is not None:
                detect += 1
                list_p_value.append(p_value)

                if (p_value < alpha):
                    reject += 1
        with open("results/tpr_parametric.txt", "a") as f:
            f.write('delta: ' + str(delta) + '\n')
            f.write('tpr:' + str(reject/detect) + '\n')
            f.write(f'reject: {reject}, detect: {detect}\n')
        print(f'delta: {delta}, TPR: {reject/detect}')
    

