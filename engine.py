from tqdm import tqdm
import torch

from utils import AvgMeter, get_lr
from config import CFG

def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
        
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        

        preds = model(x, y_input)
        loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_meter.update(loss.item(), x.size(0))
        
        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
        if logger is not None:
            logger.log({"train_step_loss": loss_meter.avg, 'lr': lr})
    
    return loss_meter.avg

def valid_epoch(model, valid_loader, criterion):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    with torch.no_grad():
        for x, y in tqdm_object:
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds = model(x, y_input)
            loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))


            loss_meter.update(loss.item(), x.size(0))
    
    return loss_meter.avg


def train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler,
               step,
               logger):
    
    best_loss = float('inf')
    
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")
        if logger is not None:
            logger.log({"Epoch": epoch + 1})
        
        train_loss = train_epoch(model, train_loader, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, logger=logger)
        
        valid_loss = valid_epoch(model, valid_loader, criterion)
        print(f"Valid loss: {valid_loss:.3f}")
        
        if step == 'epoch':
            pass
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_valid_loss.pth')
            print("Saved Best Model")
        
        if logger is not None:
            logger.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss
            })
            logger.save('best_valid_loss.pth')