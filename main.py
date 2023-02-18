'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import argparse
import os
from load_dataset import load_data
from utils import seed_everything,cosine_lr_schedule,init_logger,update_logger,summarize_logger,output_prediction,integrate_test_status
from FCLL_model import FCLL
import torch
from torch.optim import AdamW
from train import train_fn
from evaluation import evaluation_fn

def set_config():
    parser = argparse.ArgumentParser(description="FCLL for Visual WSD")
    parser.add_argument('--run_name', type=str, required=False, default="FCLLv1")
    parser.add_argument('--data_dir', type=str, required=False, default=r"D:\工作\Datasets\V-WSD\vwsd")
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--train_batch_size', type=int, required=False, default=2)
    parser.add_argument('--eval_batch_size', type=int, required=False, default=16)
    parser.add_argument('--num_workers', type=int, required=False, default=4)
    parser.add_argument('--epochs', type=int, required=False, default=10, help="epochs to train")
    parser.add_argument('--alpha', type=float, required=False, default=0.4)

    parser.add_argument('--save_dir', type=str, required=False, default="./save_model",help="path to save weights")
    parser.add_argument('--log_dir', type=str, required=False, default="./log",help="path to save log")
    parser.add_argument('--result_dir', type=str, required=False, default="./result", help="path to save results")

    parser.add_argument('--mixed_precision', action='store_true', default=True, help="use mixed precision or not")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="clip the gradients or not")
    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate")
    parser.add_argument('--min_lr', type=float, required=False, default=0.0, help="minimum learning rate")
    parser.add_argument('--weight_decay', type=float, required=False, default=0.05, help="weight_decay")
    parser.add_argument('--patience', type=int, required=False, default=3, help="patience for rlp")

    parser.add_argument('--use_checkpoint', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluate or not")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_config()
    seed_everything(args.seed)

    # 1. get data
    kb_metadata,eval_data,en_data,fa_data,it_data = load_data(args)

    # 2. set the model
    model = FCLL(args,kb_metadata)
    if args.use_checkpoint:
        model.load_state_dict(torch.load(os.path.join(args.save_dir,'checkpoint2.pt')))
        print('>>>>>>Using the model checkpoint :\n')
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # init logger before training
    log_path = init_logger(args)
    # set the recorder for saving the best model
    recorder = {
        'best_epoch': 0,
        'best_en_acc': 0,
        'best_en_mrr': 0,
        'best_fa_acc': 0,
        'best_fa_mrr': 0,
        'best_it_acc': 0,
        'best_it_mrr': 0,
        'best_total_acc': 0,
        'best_total_mrr': 0
    }

    # 3.train and evaluation
    for epoch in range(args.epochs):
        print('======'*8, f'Epoch {str(epoch+1)}', '======'*8)
        cosine_lr_schedule(optimizer, epoch, args.patience, args.lr, args.min_lr)
        if not args.evaluate:
            train_status = train_fn(epoch,args,kb_metadata,model,optimizer,mode='train')
            eval_status,best_imgs,SORT10,GOLD = evaluation_fn(epoch, args, eval_data, model, mode='eval')
        en_status, en_best_imgs, en_SORT10,en_GOLD = evaluation_fn(epoch, args, en_data, model, mode='test_en')
        fa_status, fa_best_imgs, fa_SORT10,fa_GOLD = evaluation_fn(epoch, args, fa_data, model, mode='test_fa')
        it_status, it_best_imgs, it_SORT10,it_GOLD = evaluation_fn(epoch, args, it_data, model, mode='test_it')

        # calculate total metrics
        total_acc,total_mrr = integrate_test_status([en_SORT10,fa_SORT10,it_SORT10],[en_GOLD,fa_GOLD,it_GOLD])

        # output prediction.txt
        output_prediction(epoch, en_status, en_best_imgs, en_SORT10, language='en')
        output_prediction(epoch, fa_status, fa_best_imgs, fa_SORT10, language='fa')
        output_prediction(epoch, it_status, it_best_imgs, it_SORT10, language='it')

        # update logger
        update_logger(log_path,epoch,train_status,eval_status,[en_status,fa_status,it_status],total_acc,total_mrr)

        # save model
        if total_acc > recorder['best_total_acc']:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pt'))
            recorder['best_epoch'] = epoch + 1
            recorder['best_en_acc'] = en_status['acc']
            recorder['best_en_mrr'] = en_status['mrr']
            recorder['best_fa_acc'] = fa_status['acc']
            recorder['best_fa_mrr'] = fa_status['mrr']
            recorder['best_it_acc'] = it_status['acc']
            recorder['best_it_mrr'] = it_status['mrr']
            recorder['best_total_acc'] = total_acc
            recorder['best_total_mrr'] = total_mrr
        elif total_acc == recorder['best_total_acc'] and total_mrr > recorder['best_total_mrr']:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pt'))
            recorder['best_epoch'] = epoch + 1
            recorder['best_en_acc'] = en_status['acc']
            recorder['best_en_mrr'] = en_status['mrr']
            recorder['best_fa_acc'] = fa_status['acc']
            recorder['best_fa_mrr'] = fa_status['mrr']
            recorder['best_it_acc'] = it_status['acc']
            recorder['best_it_mrr'] = it_status['mrr']
            recorder['best_total_acc'] = total_acc
            recorder['best_total_mrr'] = total_mrr

    # 5.print the best result and write into logger
    best_recorder = f'At the end, the best evaluating description: {recorder}'
    print('\n'+best_recorder)
    summarize_logger(log_path, best_recorder)



