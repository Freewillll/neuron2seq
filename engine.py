from tqdm import tqdm
import torch
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.util import AvgMeter, get_lr, save_image_debug


def train_epoch(model, tokenizer, train_loader, optimizer, lr_scheduler, criterion, epoch, logger=None, args=None, debug=False):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    saved = 0
    
    for img, seq, img_files, _ in tqdm_object:
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

        if debug:
            if saved < args.num_debug_save:
                saved += 1
                save_image_debug(tokenizer, img_files, img, seq, preds[:, :-3], epoch, 'train', 0, args)

    return loss_meter.avg


def valid_epoch(model, tokenizer, valid_loader, criterion, epoch, args, debug):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    saved = 0
    
    with torch.no_grad():
        for img, seq, img_files, _ in tqdm_object:
            img, seq = img.to(args.device, non_blocking=True), seq.to(args.device, non_blocking=True)

            padding = torch.ones(seq.size(0), 4).fill_(args.pad_idx).long().to(seq.device)
            seq = torch.cat([seq, padding], dim=1)
            
            seq_input = seq[:, :-5]
            seq_expected = seq[:, 5:]

            preds = model(img, seq_input)
            
            loss = criterion(preds.reshape(-1, preds.shape[-1]), seq_expected.reshape(-1))

            loss_meter.update(loss.item(), img.size(0))

            tqdm_object.set_postfix(val_loss=loss_meter.avg)

            if debug:
                if saved < args.num_debug_save:
                    saved += 1
                    save_image_debug(tokenizer, img_files, img, seq, preds[:, :-3], epoch, 'val', 0, args)
    
    return loss_meter.avg


def train_eval(model, 
               tokenizer,
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

        debug = False
        if (epoch % args.debug_frequency == 0) and (epoch != 0):
            debug = True
        
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, epoch, logger=logger, args=args, debug=debug)
        
        print(f'train_loss: {train_loss:.5f}')
        
        if step == 'epoch':
            pass
        
        torch.save(model.state_dict(), os.path.join(args.save_folder, 'final.pth'))

        if epoch % args.val_frequency == 0 and epoch != 0:
            valid_loss = valid_epoch(model, tokenizer, valid_loader, criterion, epoch, args=args, debug=debug)
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