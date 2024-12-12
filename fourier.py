import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time

def fourier_feature_mapping(x, B):
    device = x.device
    B = B.to(device)
    x_proj = 2 * np.pi * x @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatureNetwork(nn.Module):
    def __init__(self, input_dim, mapping_dim, hidden_dim=128):
        super(FourierFeatureNetwork, self).__init__()
        self.B = nn.Parameter(torch.randn(mapping_dim, input_dim) * 10.0)
        self.net = nn.Sequential(
            nn.Linear(2 * mapping_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x_mapped = fourier_feature_mapping(x, self.B)
        return self.net(x_mapped)

def fourier(m, lr, epochs, dataset_train, dataset_test, Lambda_d, description):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping_dim = m
    #mapping_dim = 512
    hidden_dim = m
    #hidden_dim = 256
    runs = 10
    batch_size = 1000

    d = np.load(dataset_train)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load(dataset_test)
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    input_dim = X_train.shape[1]

    model = FourierFeatureNetwork(input_dim, mapping_dim, hidden_dim).to(device)
    mse_loss = nn.MSELoss().to(device)

    results = []

    for run in range(runs):
        print(f"----------------------------{run + 1}th run---------------------------------")
        start_time = time.time()

        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        best_train_loss = float('inf')
        best_test_loss = float('inf')

        X_train_small = X_train[batch_size * run: batch_size * (run + 1)]
        y_train_small = y_train[batch_size * run: batch_size * (run + 1)]

        train_time = 0
        test_time = 0
        test_count = 0
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_small)
            loss = mse_loss(pred, y_train_small)
            loss.backward()
            optimizer.step()
            train_time += time.time() - epoch_start_time

            if epoch % 1000 == 0:
                epoch_start_time = time.time()
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test)
                    test_loss = mse_loss(test_pred, y_test)

                #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}")

                if test_loss.item() < best_test_loss:
                    best_test_loss = test_loss.item()
                    best_train_loss = loss.item()
                test_time += time.time() - epoch_start_time
                test_count += 1

        elapsed_time = time.time() - start_time
        train_time = train_time / epochs
        test_time = test_time / test_count
        results.append({
            "Best_Train_Loss": best_train_loss,
            "Best_Test_Loss": best_test_loss,
            "Elapsed_Time": elapsed_time,
            "Train_Time": train_time,
            "Test_Time": test_time
        })
        print(f"Run {run + 1} completed in {elapsed_time:.2f} seconds. Best Train Loss: {best_train_loss:.4e} Best Test Loss: {best_test_loss:.4e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(description + "_FFN.csv", index=False)
    print("Results saved to 'fourier_feature_results.csv'")

