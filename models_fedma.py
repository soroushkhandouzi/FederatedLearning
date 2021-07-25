import numpy as np

def pdm_prepare_weights(net):
    weights = []

    layer_i = 1
    statedict = net.state_dict()
    print(statedict.keys())
    net_weights = []
    while True:

        if ('fc%d.weight' % layer_i) not in statedict.keys():
            break

        layer_weight = statedict['fc%d.weight' % layer_i].numpy().T
        layer_bias = statedict['fc%d.bias' % layer_i].numpy()

        net_weights.extend([layer_weight, layer_bias])
        layer_i += 1

    weights.append(net_weights)

    return weights

def compute_pdm_matching_multilayer(models, train_dl, test_dl, cls_freqs, n_classes, sigma0=None, it=0, sigma=None, gamma=None):
    batch_weights = pdm_prepare_weights(models)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gammas = [1.0, 1e-3, 50.0] if gamma is None else [gamma]
    sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
    sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]

    for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
        print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)

        hungarian_weights = pdm_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it, gamma_layers=gamma
        )

        train_acc, test_acc, _, _ = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes)

        key = (sigma0, sigma, gamma)
        res[key] = {}
        res[key]['shapes'] = list(map(lambda x: x.shape, hungarian_weights))
        res[key]['train_accuracy'] = train_acc
        res[key]['test_accuracy'] = test_acc

        print('Sigma0: %s. Sigma: %s. Shapes: %s, Accuracy: %f' % (
        str(sigma0), str(sigma), str(res[key]['shapes']), test_acc))

        if train_acc > best_train_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_weights = hungarian_weights
            best_sigma = sigma
            best_gamma = gamma
            best_sigma0 = sigma0

    print('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

    return (best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res)
def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs
def layer_group_descent(batch_weights, batch_frequencies, sigma_layers, sigma0_layers, gamma_layers, it,
                        assignments_old=None):

    n_layers = int(len(batch_weights[0]) / 2)
    J = len(batch_weights)
    D = batch_weights[0][0].shape[0]
    K = batch_weights[0][-1].shape[0]

    if assignments_old is None:
        assignments_old = (n_layers - 1) * [None]
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    if batch_frequencies is None:
        last_layer_const = [np.ones(K) for _ in range(J)]
    else:
        last_layer_const = []
        total_freq = sum(batch_frequencies)
        for f in batch_frequencies:
            last_layer_const.append(f / total_freq)

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None
    assignment_all = []

    ## Group descent for layer
    for c in range(1, n_layers)[::-1]:
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]
        if c == (n_layers - 1) and n_layers > 2:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1), batch_weights[j][c * 2])) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        elif c > 1:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in
                               range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][0].T, batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array(
                D * [1 / sigma0] + [1 / sigma0_bias] + (weights_bias[0].shape[1] - 1 - D) * [1 / sigma0])
            mean_prior = np.array(D * [mu0] + [mu0_bias] + (weights_bias[0].shape[1] - 1 - D) * [mu0])
            if n_layers == 2:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in
                    range(J)]
            else:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + (weights_bias[j].shape[1] - 1 - D) * [1 / sigma]) for
                    j in range(J)]

        assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it,
                                                                      assignment=assignments_old[c - 1])
        L_next = global_weights_c.shape[0]
        assignment_all = [assignment_c] + assignment_all

        if c == (n_layers - 1) and n_layers > 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:], softmax_bias]
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:], softmax_inv_sigma]
        elif c > 1:
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:]] + global_inv_sigmas_out
        else:
            if n_layers == 2:
                softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
                global_weights_out = [softmax_bias]
                global_inv_sigmas_out = [softmax_inv_sigma]
            global_weights_out = [global_weights_c[:, :D].T, global_weights_c[:, D],
                                  global_weights_c[:, (D + 1):]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, :D].T, global_sigmas_c[:, D],
                                     global_sigmas_c[:, (D + 1):]] + global_inv_sigmas_out

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out, assignment_all

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        print("dara ha",dataidx)
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logging.debug('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(train, test, n_nets, alpha=0.5):

    n_train = train
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, n_nets)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    traindata_cls_counts = record_net_data_stats(train, net_dataidx_map)

    return traindata_cls_counts