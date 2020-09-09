from model import DeepPix
from dataset import Casia_surf
from loss import CumulativeLoss
from torch.utils.data import DataLoader
import torch.optim as optim 
import numpy as np
import torch


def train(dataloader_train, model, epochs, loss_fn, optimizer, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_epoch_arr = []
    min_loss = 1000 
    n_iters = np.ceil(len(dataloader_train) / batch_size)

    model.train()

    for epoch in range(epochs):
        for i, data in enumerate(dataloader_train, 0):
            inputs, labels = data["images"], data["labels"]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            aux_op, fin_op = model(inputs)
            aux_op = aux_op.view(aux_op.shape[0], -1)
            labels = labels.view(batch_size, 1)
            loss = loss_fn(aux_op, fin_op, labels)                          
            loss.backward()
            optimizer.step()

            if i % 100 == 0: 
                print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

            del inputs, labels, aux_op, fin_op
            torch.cuda.empty_cache()

        loss_epoch_arr.append(loss.item())  


def evaluate(dataloader, model):
    model.eval()
    total, correct  = 0, 0 
    for data in dataloader: 
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    data_dir = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data"
    train_csv = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\train_folds.csv"
    
    dataset = Casia_surf(data_dir, train_csv, (224, 224))
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    net = DeepPix().to(device)
    optimizer = optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-5)
    epoch = 1
    loss_fn = CumulativeLoss(beta=0.5)

    train(train_dataloader, net, 1, loss_fn, optimizer, batch_size)

    