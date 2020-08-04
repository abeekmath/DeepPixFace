import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader

import config 
import dataset 
from engine import train_fn, eval_fn
from model import DeepPix
from loss import CumulativeLoss


def run_training(): 
    train_dataset = dataset.Casia_surf(
        data_dir=config.DATA_DIR,
        csv_file=config.TRAIN_FILE, 
        resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE, 
        num_workers= config.NUM_WORKERS,
        shuffle=True,
        pin_memory=config.PIN_MEMORY, 
    )
    test_dataset = dataset.Casia_surf(
        data_dir=config.DATA_DIR,
        csv_file=config.TRAIN_FILE, 
        resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size = config.EVAL_BATCH_SIZE, 
        num_workers= config.NUM_WORKERS,
        shuffle=False,
        pin_memory=config.PIN_MEMORY, 
    )

    model = DeepPix()
    model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LR, 
                           weight_decay=config.WEIGHT_DECAY)
    loss_fn = CumulativeLoss(0.5)
    for epoch in range(config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, loss_fn, epoch)
        eval_loss, eval_accuracy = eval_fn(model, test_loader, loss_fn)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={eval_loss}, Test Accuracy={eval_accuracy}"
        )
    
    print("Done Training")

        
if __name__ == "__main__":
    run_training()

    
