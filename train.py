import numpy as np
import argparse
import os
import json
from datetime import datetime
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sgd import SGD_Vanilla
from svrg import SVRG, SVRG_Snapshot
from storc import STORC
from utils import (
    MNIST_dataset,
    MNIST_logistic,
    MNIST_nn_one_layer,
    AverageCalculator,
    accuracy,
    plot_train_stats,
    plot_comparison,
)

parser = argparse.ArgumentParser(description="Train classifiers via SGD, SVRG, and STORC on MNIST dataset.")
parser.add_argument('--optimizer', type=str, default="STORC", help="Optimizer to use (SGD, SVRG, STORC)")
parser.add_argument('--nn_model', type=str, default="MNIST_logistic", help="Neural network model to use")
parser.add_argument('--n_iter', type=int, default=30, help="Number of training iterations")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--store_stats_interval', type=int, default=10, help="How often to store training statistics")
parser.add_argument('--beta', type=float, default=0.1, help="Beta parameter for STORC optimizer")

OUTPUT_DIR = "outputs"
BATCH_SIZE_LARGE = 256  # for validation and snapshots
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def train_SGD_one_iter(model, optimizer, train_loader, loss_fn):
    model.train()
    loss_calculator = AverageCalculator()
    acc_calculator = AverageCalculator()

    for images, labels in tqdm(train_loader, desc="Training SGD", leave=False):
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0], -1)

        yhat = model(images)
        loss_iter = loss_fn(yhat, labels)

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        acc_iter = accuracy(yhat, labels)
        loss_calculator.update(loss_iter.item())
        acc_calculator.update(acc_iter)

    return loss_calculator.avg, acc_calculator.avg

def train_STORC_one_iter(model, optimizer, train_loader, loss_fn):
    model.train()
    loss_calculator = AverageCalculator()
    acc_calculator = AverageCalculator()

    for images, labels in tqdm(train_loader, desc="Training STORC", leave=False):
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0], -1)

        yhat = model(images)
        loss_iter = loss_fn(yhat, labels)

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        acc_iter = accuracy(yhat, labels)
        loss_calculator.update(loss_iter.item())
        acc_calculator.update(acc_iter)

    return loss_calculator.avg, acc_calculator.avg

def train_SVRG_one_iter(model_k, model_snapshot, optimizer_inner, optimizer_snapshot, train_loader, snapshot_loader, loss_fn):
    model_k.train()
    model_snapshot.train()
    loss_calculator = AverageCalculator()
    acc_calculator = AverageCalculator()

    # Calculate snapshot gradient
    optimizer_snapshot.zero_grad()
    for images, labels in tqdm(snapshot_loader, desc="Calculating Snapshot Gradient", leave=False):
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0], -1)
        yhat = model_snapshot(images)
        snapshot_loss = loss_fn(yhat, labels) / len(snapshot_loader)
        snapshot_loss.backward()

    # Update snapshot gradient
    mu = optimizer_snapshot.get_param_groups()
    optimizer_inner.set_mu(mu)

    for images, labels in tqdm(train_loader, desc="Training SVRG", leave=False):
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0], -1)
        yhat = model_k(images)
        loss_iter = loss_fn(yhat, labels)

        optimizer_inner.zero_grad()
        loss_iter.backward()
        optimizer_inner.step(optimizer_snapshot.get_param_groups())

        acc_iter = accuracy(yhat, labels)
        loss_calculator.update(loss_iter.item())
        acc_calculator.update(acc_iter)

    return loss_calculator.avg, acc_calculator.avg

def validate(model, val_loader, loss_fn):
    model.eval()
    loss_calculator = AverageCalculator()
    acc_calculator = AverageCalculator()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], -1)

            yhat = model(images)
            loss_iter = loss_fn(yhat, labels)
            acc_iter = accuracy(yhat, labels)

            loss_calculator.update(loss_iter.item())
            acc_calculator.update(acc_iter)

    return loss_calculator.avg, acc_calculator.avg

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    # Load MNIST dataset
    train_set, val_set = MNIST_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    snapshot_loader = DataLoader(train_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)

    # Initialize model
    NN_model = MNIST_logistic if args.nn_model == "MNIST_logistic" else MNIST_nn_one_layer
    model_sgd = NN_model().to(device)
    model_svrg = NN_model().to(device)
    model_storc = NN_model().to(device)
    model_snapshot = NN_model().to(device)

    # Initialize optimizers
    optimizer_sgd = SGD_Vanilla(model_sgd.parameters(), lr=args.lr)
    optimizer_svrg = SVRG(model_svrg.parameters(), lr=args.lr)
    optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())
    optimizer_storc = STORC(model_storc.parameters(), lr=args.lr, beta=args.beta)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Training stats
    train_stats_sgd = ([], [], [], [])
    train_stats_svrg = ([], [], [], [])
    train_stats_storc = ([], [], [], [])

    for iteration in range(args.n_iter):
        t0 = time.time()

        # Train SGD
        train_loss_sgd, train_acc_sgd = train_SGD_one_iter(model_sgd, optimizer_sgd, train_loader, nn.NLLLoss())
        val_loss_sgd, val_acc_sgd = validate(model_sgd, val_loader, nn.NLLLoss())

        # Train SVRG
        train_loss_svrg, train_acc_svrg = train_SVRG_one_iter(
            model_svrg, model_snapshot, optimizer_svrg, optimizer_snapshot, train_loader, snapshot_loader, nn.NLLLoss()
        )
        val_loss_svrg, val_acc_svrg = validate(model_svrg, val_loader, nn.NLLLoss())

        # Train STORC
        train_loss_storc, train_acc_storc = train_STORC_one_iter(model_storc, optimizer_storc, train_loader, nn.NLLLoss())
        val_loss_storc, val_acc_storc = validate(model_storc, val_loader, nn.NLLLoss())

        # Log results
        train_stats_sgd[0].append(train_loss_sgd)
        train_stats_sgd[1].append(val_loss_sgd)
        train_stats_sgd[2].append(train_acc_sgd)
        train_stats_sgd[3].append(val_acc_sgd)

        train_stats_svrg[0].append(train_loss_svrg)
        train_stats_svrg[1].append(val_loss_svrg)
        train_stats_svrg[2].append(train_acc_svrg)
        train_stats_svrg[3].append(val_acc_svrg)

        train_stats_storc[0].append(train_loss_storc)
        train_stats_storc[1].append(val_loss_storc)
        train_stats_storc[2].append(train_acc_storc)
        train_stats_storc[3].append(val_acc_storc)

        print(f"Iteration {iteration} - SGD: Train Loss={train_loss_sgd:.4f}, Val Loss={val_loss_sgd:.4f}")
        print(f"Iteration {iteration} - SVRG: Train Loss={train_loss_svrg:.4f}, Val Loss={val_loss_svrg:.4f}")
        print(f"Iteration {iteration} - STORC: Train Loss={train_loss_storc:.4f}, Val Loss={val_loss_storc:.4f}")

        if (iteration + 1) % args.store_stats_interval == 0:
            np.savez(
                os.path.join(OUTPUT_DIR, 'train_stats.npz'),
                train_loss_sgd=np.array(train_stats_sgd[0]),
                val_loss_sgd=np.array(train_stats_sgd[1]),
                train_loss_svrg=np.array(train_stats_svrg[0]),
                val_loss_svrg=np.array(train_stats_svrg[1]),
                train_loss_storc=np.array(train_stats_storc[0]),
                val_loss_storc=np.array(train_stats_storc[1])
            )
            plot_comparison(train_stats_sgd, train_stats_svrg, train_stats_storc, OUTPUT_DIR)
