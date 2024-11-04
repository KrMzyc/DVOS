# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
Modified for PyTorch conversion by ChatGPT at 10/15/2024
"""

# Import Libraries:
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.data_left import VisualOdometryDataLoaderLeft
from utils.data_right import VisualOdometryDataLoaderRight
from RNN.deepvo_net import DeepVONet
from FlowNet.flownet import *


# Custom loss function
def custom_loss(y_pred, y_true, k, criterion):
    mse_position = criterion(y_pred[:, :3], y_true[:, :3])
    mse_orientation = criterion(y_pred[:, 3:], y_true[:, 3:])
    return mse_position + k * mse_orientation


def loss_fn(model, x, y, k, criterion):
    y_pred = model(x)  # Forward pass
    return custom_loss(y_pred, y, k, criterion)


def train_one_epoch(model, train_loader_left, train_loader_right, flownet_left, flownet_right, optimizer, criterion, config, device):
    model.train()
    epoch_loss = 0
    for (batch_x_left, batch_y_left), (batch_x_right, batch_y_right) in zip(train_loader_left, train_loader_right):
        
        optimizer.zero_grad()

        # Move data to the appropriate device (GPU/CPU)
        batch_x_left, batch_y_left, batch_x_right = batch_x_left.to(device).float(), batch_y_left.to(device).float(), batch_x_right.to(device).float()

        # Pass data through flownets and concatenate outputs
        x_left = flownet_left(batch_x_left)
        x_right = flownet_right(batch_x_right)
        x = torch.cat((x_left, x_right), dim=-1)

        # Calculate loss and perform backpropagation
        loss_value = loss_fn(model, x, batch_y_left, config['k'], criterion)
        loss_value.backward()
        optimizer.step()

        epoch_loss += loss_value.item()

    return epoch_loss / len(train_loader_left)


def validate_one_epoch(model, val_loader_left, val_loader_right, flownet_left, flownet_right, criterion, config, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (batch_x_left, batch_y_left), (batch_x_right, batch_y_right) in zip(val_loader_left, val_loader_right):
            batch_x_left, batch_y_left, batch_x_right = batch_x_left.to(device).float(), batch_y_left.to(device).float(), batch_x_right.to(device).float()

            x_left = flownet_left(batch_x_left)
            x_right = flownet_right(batch_x_right)
            x = torch.cat((x_left, x_right), dim=-1)

            loss_value = loss_fn(model, x, batch_y_left, config['k'], criterion)
            val_loss += loss_value.item()

    return val_loss / len(val_loader_left)


def train(flownet_left, flownet_right, model, config):
    print('Load Data...')
    print('=' * 50)
    
    train_loader_left = torch.utils.data.DataLoader(
        VisualOdometryDataLoaderLeft(config['datapath'], 192, 640), batch_size=config['bsize'], shuffle=True)
    val_loader_left = torch.utils.data.DataLoader(
        VisualOdometryDataLoaderLeft(config['datapath'], 192, 640, val=True), batch_size=config['bsize'], shuffle=False)
    train_loader_right = torch.utils.data.DataLoader(
        VisualOdometryDataLoaderRight(config['datapath'], 192, 640), batch_size=config['bsize'], shuffle=True)
    val_loader_right = torch.utils.data.DataLoader(
        VisualOdometryDataLoaderRight(config['datapath'], 192, 640, val=True), batch_size=config['bsize'], shuffle=False)

    criterion = nn.MSELoss()
    optimizer = None

    if config['train'] == 'deepvo':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['train'] == 'magicvo':
        optimizer = optim.Adagrad(model.parameters(), lr=config['lr'])
    elif config['train'] == 'poseconvgru' or config['train'] == 'trmnet':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    flownet_left, flownet_right = flownet_left.to(device), flownet_right.to(device)

    train_loss_results = []
    val_loss_results = []

    print(f"Training {config['train'].upper()} model...")
    print('=' * 50)

    for epoch in range(config['train_iter']):
        print(f"Epoch {epoch + 1}/{config['train_iter']}")

        train_loss = train_one_epoch(model, train_loader_left, train_loader_right, flownet_left, flownet_right, optimizer, criterion, config, device)
        val_loss = validate_one_epoch(model, val_loader_left, val_loader_right, flownet_left, flownet_right, criterion, config, device)

        train_loss_results.append(train_loss)
        val_loss_results.append(val_loss)

        print(f"Training Loss: {train_loss:.10f} \t Validation Loss: {val_loss:.10f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{config['checkpoint_path']}/{config['train']}_checkpoint_{epoch + 1}.pth")
            print(f"Saved checkpoint at epoch {epoch + 1}")

    # Plot loss curves
    print('Plotting loss curves...')
    fig, ax = plt.subplots()
    ax.plot(train_loss_results, label='Training Loss', color='blue', linestyle='--')
    ax.plot(val_loss_results, label='Validation Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Metrics - {config["train"].upper()}')
    ax.grid()
    ax.legend()

    plt.savefig(f"loss_{config['train']}.png")
    plt.show()


def main():
    # Configurations and model setup
    config = {
        'mode': 'train',
        'datapath': '/home/senselab/KITTI',
        'bsize': 8,
        'lr': 0.001,
        'train_iter': 140,
        'checkpoint_path': '/home/senselab/StereoVIO_torch/checkpoints',
        'k': 100,
        'train': 'deepvo'
    }

    # Define models for training
    flownet_left = FlowNetS(height=192, width=640).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    flownet_right = FlowNetS(height=192, width=640).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if config['train'] == 'deepvo':
        model = DeepVONet()
    

    train(flownet_left, flownet_right, model, config)


if __name__ == "__main__":
    main()
