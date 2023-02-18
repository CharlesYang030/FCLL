'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import torch
import torch.nn as nn
import numpy as np
from load_dataset import get_dataloader
from tqdm import tqdm

def convert_models_to_fp32(model):
    for p in model.parameters():
        if not p.data == None:
            p.data = p.data.float()
        if not p.grad == None:
            p.grad.data = p.grad.data.float()

def train_fn(epoch, args, train_metadata, model, optimizer, mode):
    model.train()
    device = args.device
    train_dataloader = get_dataloader(args, train_metadata, mode=mode)
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=150)

    # record status
    LOSS = []
    LR = []

    for idx,batch in loop:
        ambiguous_word,amb_trans,language,gloss,image,image_idx,gloss_idx = batch
        ambiguous_word,amb_trans,language, gloss = list(ambiguous_word),list(amb_trans), list(language), list(gloss)
        image,image_idx,gloss_idx = image.to(device),image_idx.to(device),gloss_idx.to(device)

        # 设置alpha值
        if epoch > 0:
            alpha = args.alpha
        else:
            alpha = args.alpha * min(1, idx / len(train_dataloader))

        # start training
        optimizer.zero_grad()
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                loss_SIC, loss_ISC, loss_itm, loss_TTC = model(ambiguous_word,amb_trans,language,gloss,image,image_idx,gloss_idx,alpha=alpha)
                loss = loss_SIC + loss_ISC + loss_itm + loss_TTC
        # backward, clip grad and step
        convert_models_to_fp32(model)
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # record
        loss_np = loss.detach().cpu().numpy()
        LOSS.append(loss_np)
        LR.append(optimizer.param_groups[0]["lr"])

        # update the loop message
        loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}] Training [{idx + 1}/{len(loop)}]')
        loop.set_postfix(loss=np.mean(LOSS))

    status = {
        'loss': np.mean(LOSS),
        'lr': LR[-1]
    }

    return status

