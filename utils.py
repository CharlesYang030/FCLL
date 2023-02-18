'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import random
import os
from datetime import datetime
import numpy as np
import torch
import math

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_logger(args):
    dirs = [args.log_dir,args.save_dir,args.result_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    log_path = os.path.join(args.log_dir,str(datetime.now())[:-7].replace(' ', '_').replace(':', '.') + f'{args.run_name}_log.txt')
    logger = open(log_path, 'w',encoding='utf-8')
    logger.write(str(args) + '\n\n')
    logger.close()
    return log_path

def update_logger(log_path,epoch,train_status,eval_status,test_status,total_acc,total_mrr):
    logger = open(log_path, 'a+',encoding='utf-8')
    logger.write('>>>>>> (' + str(datetime.now())[:-7] + ') Epoch ' + str(epoch + 1) + ':\n')
    logger.write('Train Loss: ' + str(train_status['loss']) + '\n')
    logger.write('Learning rate: ' + str(train_status['lr']) + '\n')
    logger.write('Evaluation Acc: ' + str(eval_status['acc']) + '    Mrr: ' + str(eval_status['mrr']) + '\n')
    state_name = ['"test_en"','"test_fa"','"test_it"']
    for idx,state in enumerate(test_status):
        logger.write(f'{state_name[idx]} Acc: ' + str(state['acc']) + '    Mrr: ' + str(state['mrr']) + '\n')
    logger.write('*** Total Acc: ' + str(total_acc) + '    Mrr: ' + str(total_mrr) + '\n\n')
    logger.close()

def summarize_logger(log_path,best_recorder):
    logger = open(log_path, 'a+')
    logger.write('\n**********' + str(best_recorder) + '\n')
    logger.close()

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cal_metrics(SORT10,GOLD,total):
    gold_imgs = []
    for golds in GOLD:
        for g in golds:
            gold_imgs.append(g)

    acc = 0
    mrr = 0
    count = 0
    for prediction in SORT10:
        batch = len(prediction)
        for i in range(batch):
            glod = gold_imgs[count + i]
            pred_best = prediction[i][0]
            # acc
            if glod == pred_best:
                acc += 1
            # mrr
            for j in range(len(prediction[i])):
                if glod == prediction[i][j]:
                    # 注意j的取值从0开始
                    mrr += 1/(j+1)
                    break
        count += batch
    acc = acc / total * 100.
    mrr = mrr / total * 100.
    return acc,mrr

def integrate_test_status(SORT10S,GOLDS):
    total_gold_imgs = []
    for GOLD in GOLDS:
        for golds in GOLD:
            for g in golds:
                total_gold_imgs.append(g)
    total_length = len(total_gold_imgs)

    total_sort10 = []
    for SORT10 in SORT10S:
        for sort in SORT10:
            for s10 in sort:
                total_sort10.append(s10)

    # calculate acc and mrr
    acc = 0
    mrr = 0
    for idx,sort10 in enumerate(total_sort10):
        pred_best = sort10[0]
        gold = total_gold_imgs[idx]
        # acc
        if gold == pred_best:
            acc += 1
        # mrr
        for j in range(len(sort10)):
            if gold == sort10[j]:
                # 注意j的取值从0开始
                mrr += 1 / (j + 1)
                break

    acc = acc / total_length * 100.
    mrr = mrr / total_length * 100.

    print('*** The total test Acc: ',acc,'  The total test MRR: ',mrr)
    return acc,mrr

def output_prediction(epoch,status,best_imgs,predictions,language):
    dir = f'./result/Epoch{epoch+1}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # 打印预测结果
    out_path = os.path.join(dir,f'Epoch{epoch+1}_{language}_result.txt')
    outfile = open(out_path, 'w', encoding='utf-8')
    outfile.write(f'{language} Accuracy = ' + str(status['acc']) + ' MRR = ' + str(status['mrr']) + '\n')
    for idx, pre in enumerate(best_imgs):
        outfile.write(str(idx + 1) + ' ' + pre + '\n')
    outfile.close()

    prediction_path = os.path.join(dir,f'Epoch{epoch+1}_{language}_prediction.txt')
    prefile = open(prediction_path,'w',encoding='utf-8')
    for k,prediction in enumerate(predictions):
        batch = len(prediction)
        for i in range(batch):
            for j in range(len(prediction[i])):
                prefile.write(prediction[i][j])
                if j < len(prediction[i]) -1:
                    prefile.write('\t')
            if not (k == len(predictions) -1 and i == batch -1):
                prefile.write('\n')
    prefile.close()


def _convert_image_to_rgb(image):
    return image.convert("RGB")

_transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
