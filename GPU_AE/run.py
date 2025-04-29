from util import *
import time
from model import WDGRL, AutoEncoder
def run(self):
    seed, wdgrl, ae, np_wdgrl, np_ae = self
    np.random.seed(seed)
    start = time.time()
    print(f'Starting iteration #{seed}')
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
    # print(np.sqrt(etajTsigmaetaj[0][0]))
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
    threshold[0] = max(threshold[0], itv[0])
    threshold[1] = min(threshold[1], itv[1])
    # print(itv)
    # print(etajTsigmaetaj)
    list_zk, list_Oz = run_parametric_si(a.reshape((-1, d)), b.reshape((-1, d)), threshold, wdgrl, ae, np_wdgrl, np_ae, alpha, ns, nt)
    CDF = cdf(etajTmu[0][0], np.sqrt(etajTsigmaetaj[0][0]), list_zk, list_Oz, etajTx[0][0], O)
    p_value = 2 * min(CDF, 1 - CDF)
    print(f'Ending iteration #{seed}')
    print(f'+ p-value: {p_value}')
    stop = time.time()
    print(f'+ Execution time: {stop - start}')
    return p_value, stop - start
    
if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    d = 32
    generator_hidden_dims = [500, 100, 20]
    critic_hidden_dims = [100]
    wdgrl = WDGRL(input_dim=d, generator_hidden_dims=generator_hidden_dims, critic_hidden_dims=critic_hidden_dims)
    index = None
    # with open("/kaggle/input/mydata/model/wdgrl_models.txt", "r") as f:
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
    # check_point = torch.load(f"/kaggle/input/mydata/model/wdgrl_{index}.pth", map_location=wdgrl.device, weights_only=True)
    check_point = torch.load(f"model/wdgrl_{index}.pth", map_location=wdgrl.device, weights_only=True)
    wdgrl.generator.load_state_dict(check_point['generator_state_dict'])
    wdgrl.critic.load_state_dict(check_point['critic_state_dict'])
    wdgrl.generator = wdgrl.generator.cpu()

    input_dim = generator_hidden_dims[-1]
    encoder_hidden_dims = [16, 8, 4, 2]
    decoder_hidden_dims = [4, 8, 16, input_dim]
    ae = AutoEncoder(input_dim=input_dim, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims)
    index = None
    # with open("/kaggle/input/mydata/model/ae_models.txt", "r") as f:
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
    # check_point = torch.load(f"/kaggle/input/mydata/model/ae_{index}.pth", map_location=ae.device, weights_only=True)
    check_point = torch.load(f"model/ae_{index}.pth", map_location=ae.device, weights_only=True)
    ae.load_state_dict(check_point['state_dict'])
    ae.net = ae.net.cpu()
    # with open("/kaggle/working/result/p_values.txt", "w") as f:
    with open("result/p_values.txt", "w") as f:
        f.write("")
    with open("result/execution_times.txt", "w") as f:
    # with open("/kaggle/working/result/execution_times.txt", "w") as f:
        f.write("")
    for i in range(20): 
        p_value, execution_time = run((i, wdgrl, ae, wdgrl.generator, ae))
        if p_value is None:
            continue
        # with open("/kaggle/working/result/p_values.txt", "a") as f:
        with open("result/p_values.txt", "a") as f:
            f.write(f"{p_value}\n")
        # with open("/kaggle/working/result/execution_times.txt", "a") as f:
        with open("result/execution_times.txt", "a") as f:
            f.write(f"{execution_time}\n")
        