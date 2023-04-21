from tqdm import tqdm
import torch
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.util import AvgMeter, get_lr


def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None, args=None):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for img, seq, imgfile, _ in tqdm_object:
        img, seq = img.to(args.device, non_blocking=True), seq.to(args.device, non_blocking=True)

        padding = torch.ones(seq.size(0), 4).fill_(args.pad_idx).long().to(seq.device)
        seq = torch.cat([seq, padding], dim=1)
        
        seq_input = seq[:, :-5]
        seq_expected = seq[:, 5:]
        
        preds = model(img, seq_input)

        loss = criterion(preds.reshape(-1, preds.shape[-1]), seq_expected.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_meter.update(loss.item(), img.size(0))
        
        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
        if logger is not None:
            logger.log({"train_step_loss": loss_meter.avg, 'lr': lr})
    
    return loss_meter.avg


def valid_epoch(model, valid_loader, criterion, args):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    with torch.no_grad():
        for img, seq, imgfile, _ in tqdm_object:
            img, seq = img.to(args.device, non_blocking=True), seq.to(args.device, non_blocking=True)

            seq_input = seq[:, :-1]
            seq_expected = seq[:, 1:]

            preds = model(img, seq_input)
            loss = criterion(preds.reshape(-1, preds.shape[-1]), seq_expected.reshape(-1))

            loss_meter.update(loss.item(), img.size(0))
    
    return loss_meter.avg


def train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler,
               step,
               logger,
               args):
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        if logger is not None:
            logger.log({"Epoch": epoch + 1})
        
        train_loss = train_epoch(model, train_loader, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, logger=logger, args=args)
        
        print(f'train_loss: {train_loss:.5f}')
        
        if step == 'epoch':
            pass
        
        torch.save(model.state_dict(), os.path.join(args.save_folder, 'final.pth'))

        if epoch % args.val_frequency:
            valid_loss = valid_epoch(model, valid_loader, criterion, args=args)
            print(f"Valid loss: {valid_loss:.5f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.save_folder, 'best_valid_loss.pth'))
                print("Saved Best Model")
            
            if logger is not None:
                logger.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                })
                logger.save('best_valid_loss.pth')