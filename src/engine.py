import torch
import config 
from tqdm import tqdm
from loss import CumulativeLoss

def train_fn(model, data_loader, optimizer, loss_fn, epoch):
    model.train()
    fin_loss = 0
    #tk0 = tqdm(data_loader, total=len(data_loader))
    for batch_id, data in enumerate(data_loader):
        inputs, labels = data["images"], data["labels"]
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        aux_op, fin_op = model(inputs)
        aux_op = aux_op.view(aux_op.shape[0], -1)
        labels = labels.view(labels.shape[0], 1)
        loss = loss_fn(labels, fin_op, aux_op)                          
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_id * len(inputs), len(data_loader.dataset),
                100. * batch_id / len(data_loader), loss.data.item()))
    
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, loss_fn, threshold=0.5):
    model.eval()
    test_loss = 0 
    correct = 0

    with torch.no_grad():
        for batch_id, data in enumerate(data_loader): 
            batch_loss = 0
            inputs, labels = data["images"], data["labels"]
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            _, outputs = model(inputs)
            labels = labels.view(labels.shape[0], 1)
            batch_loss = loss_fn(labels, outputs).item()
            test_loss += batch_loss

            pred = torch.round(outputs)
            correct += (pred == labels.data).sum().item()

            # print('[{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
            #     batch_id * len(inputs), len(data_loader.dataset),
            #     100. * batch_id / len(data_loader), batch_loss))
               
    test_loss /= len(data_loader.dataset)
    test_accuracy = 100.0 * correct / len(data_loader.dataset)

    return test_loss, test_accuracy





