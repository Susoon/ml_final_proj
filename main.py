"""
Experiments on linear/non-linear operators learning tasks, results as mean +- standard variance over 10 runs.
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from OrdinalEntropy import ordinal_entropy
import scipy.io as scio
from models import MLP
import pandas as pd
import time
from fourier import fourier

def ordinal(m, lr, epochs, dataset_train, dataset_test, Lambda_d, description):
    model = MLP(m).cuda()

    d = np.load(dataset_train)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load(dataset_test)
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()

    l_train = []
    l_test = []
    l_time = []
    l_train_time = []
    l_test_time = []

    for times in range(10):   # run 10 times
        print('----------------------------{0}th run---------------------------------'.format(times))
        begin = time.time()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _mse_train = 9999
        _mse_test = 9999
        train_time = 0
        test_time = 0
        test_count = 0
        for epoch in range(epochs):
            start_time = time.time()
            X_train_small = X_train[1000 * times:1000 * (times+1)]
            y_train_small = y_train[1000 * times:1000 * (times+1)]
            model.train()
            optimizer.zero_grad()
            pred, feature = model(X_train_small)
            loss = mse_loss(pred, y_train_small)
            if oe:
                loss_oe = ordinal_entropy(feature, y_train_small) * Lambda_d
            else:
                loss_oe = loss * 0
            loss_all = loss + loss_oe
            loss_all.backward()

            optimizer.step()
            train_time += time.time() - start_time
            if epoch % 1000 ==0:
                start_time = time.time()
                model.eval()
                pred, feature = model(X_test)
                loss_test = mse_loss(pred, y_test)
                """
                print('{0}, Epoch: [{1}]\t'
                      'Loss_train: [{loss:.2e}]\t'
                      'Loss_test: [{loss_test:.2e}]\t'
                      'Loss_entropy: [{loss_e:.2e}]\t'
                      .format(description, epoch, loss=loss.data, loss_test=loss_test.data, loss_e=loss_oe.data))
                """

                if loss_test < _mse_test:
                    _mse_test = loss_test
                    _mse_train = loss
                    """
                    print('best model, Loss_test: [{loss_test:.2e}]'.format(
                        loss_test=_mse_test.data))
                    """
                test_time += time.time() - start_time
                test_count += 1

        l_test.append(_mse_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        end = time.time()
        l_time.append(end-begin)
        train_time = train_time / epochs
        test_time = test_time / test_count
        l_train_time.append(train_time)
        l_test_time.append(test_time)
        print(end-begin)

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    l_time = np.array(l_time)
    l_train_time = np.array(l_train_time)
    l_test_time = np.array(l_test_time)
    train_dict = {}
    train_dict['train_mse'] = l_train
    train_dict['test_mse'] = l_test
    train_dict['Elpased_Time'] = l_time
    train_dict['Train_Time'] = l_train_time
    train_dict['Test_Time'] = l_test_time

    if Linear:
        path = './Linear.mat'
    else:
        path = './nonlinear.mat'
    #scio.savemat(path, train_dict)
    if Linear:
        path = './Linear.csv'
    else:
        path = './nonlinear.csv'

    # Convert the dictionary to a DataFrame and save as CSV
    df = pd.DataFrame.from_dict(train_dict)
    df.to_csv(path, index=False)
    print('Mean: \t')
    print(np.mean(l_test))
    print('Std: \t')
    print(np.std(l_test))



def main(Linear=True, oe=True):
    if Linear:
        m = 100
        lr = 1e-3
        epochs = 50000
        dataset_train = "train.npz"
        dataset_test = "test.npz"
        Lambda_d = 1e-3
        description = 'linear'
    else:
        m = 240
        lr = 1e-3
        epochs = 100000
        dataset_train = "train_sde.npz"
        dataset_test = "test_sde.npz"
        Lambda_d = 1e-3
        description = 'nonlinear'

    ordinal(m, lr, epochs, dataset_train, dataset_test, Lambda_d, description)
    fourier(m, lr, epochs, dataset_train, dataset_test, Lambda_d, description)
    
if __name__ == "__main__":
    Linear = False  # choose the Linear/nonlinear task, i.e. True=Linear, False=nonlinear
    oe = True  # using the ordinal entropy or not

    main(Linear, oe)
