'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import cal_metrics
from load_dataset import get_dataloader
from tqdm import tqdm

@torch.no_grad()
def evaluation_fn(epoch, args, eval_data, model,mode):
    model.eval()
    eval_dataloader = get_dataloader(args, eval_data, mode=mode)
    loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=150)

    count = 0
    acc_sum = 0
    GOLD = []
    best_imgs = []
    SORT10 = []

    for idx,batch in loop:
        amb,phrase,language,img_names,candidate_images,gold_img = batch
        GOLD.append(gold_img)

        # start evaluating
        with torch.no_grad():
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    pred_imgs, sort_ten = eval_model(model,amb,phrase,language,img_names,candidate_images)

        ### record
        best_imgs += pred_imgs
        SORT10.append(sort_ten)

        # calculate the accuracy
        for i, pred in enumerate(pred_imgs):
            if pred == gold_img[i]:
                acc_sum += 1

        count += len(phrase)
        now_accuracy = (acc_sum / count) * 100.

        # update the loop message
        loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}] Evaluating [{idx + 1}/{len(loop)}] Acc:{now_accuracy:.4f} ')

    test_mode = ['test_en','test_fa','test_it']
    # calculate metrics
    acc, mrr = cal_metrics(SORT10, GOLD, count)
    if mode in test_mode:
        print(f'"{mode}" Evaluating Accuracy = ', acc, ' MRR = ', mrr)
    else:
        print('Evaluating Accuracy = ', acc, ' MRR = ', mrr)

    status = {
        'acc': acc,
        'mrr': mrr
    }

    return status,best_imgs,SORT10,GOLD

@torch.no_grad()
def eval_model(model,amb,phrase,language,img_names,candidate_images):
    this_bs = len(phrase)  # 记录当前batch的长度
    pred_imgs = []
    sort_ten = []
    for i in range(this_bs):
        # 得到每个歧义短语对应的10张候选图片的特征，以及每个歧义短语的特征（为了对齐，复制10遍）
        candidate_img_feats, candidate_img_atts = model.visual_encoder(candidate_images[i].to(model.device))
        candidate_img_feats = F.normalize(model.vision_proj(candidate_img_feats), dim=-1)
        candidate_img_atts = model.att_images_align(candidate_img_atts)

        # 自动补充上下文
        sentence = sense_complement(model,amb[i],phrase[i],language[i])

        sentence_feats, sentence_atts = model.text_encoder(model.clip_tokenize(sentence, truncate=True).to(model.device))
        sentence_feats = F.normalize(model.text_proj(sentence_feats), dim=-1)
        sentence_atts = sentence_atts.repeat(10, 1, 1)

        # 得到多模态特征以及通过多模态transformer
        multimodal_fea = torch.cat([sentence_atts,candidate_img_atts], dim=1)
        multimodal_fea = model.multi_transformer(multimodal_fea)
        multimodal_fea = model.ln(multimodal_fea.mean(1))

        # 计算logits
        itm_logits = model.itm_head(multimodal_fea)  # 通过itm_head判断每张候选图片与当前歧义短语是否匹配片
        sim_logits = sentence_feats @ candidate_img_feats.T / model.logit_scale  # 计算每个歧义短语和10张候选图片的相似度矩阵，与itm_logits相加，选出最佳图片
        final_logits = sim_logits + itm_logits[:, 1].T

        # 确定预测图片
        logits_numpy = final_logits.softmax(1).detach().cpu().numpy()
        max_index = np.argmax(logits_numpy)
        pred = img_names[i][max_index]
        pred_imgs.append(pred)

        # 记录结果
        _, idx_topk = torch.topk(final_logits, k=10, dim=-1)
        result = []
        for j in idx_topk[0]:
            j = int(j)
            result.append(img_names[i][j])
        sort_ten.append(result)

    return pred_imgs, sort_ten

def sense_complement(model,amb,phrase,language):
    kb_record = model.metadata[model.metadata['amb'] == amb]
    kb_record = kb_record[kb_record['language'] == language].reset_index(drop=True)

    if len(kb_record) != 0:
        rel_glo_cls = []
        rel_gloss = []
        for i in range(len(kb_record)):
            ### 记录batch的所有gloss和gloss的类别
            gloss = kb_record.loc[i]['gloss']
            gloss_idx = kb_record.loc[i]['gloss_idx']
            if gloss_idx not in rel_glo_cls:
                rel_gloss.append(gloss)
                rel_glo_cls.append(gloss_idx)

        phrase_feats, _ = model.sense_encoder(model.clip_tokenize(phrase, truncate=True).to(model.device))
        phrase_feats = F.normalize(model.text_proj(phrase_feats), dim=-1)
        sense_feats, _ = model.sense_encoder(model.clip_tokenize(rel_gloss, truncate=True).to(model.device))
        sense_feats = F.normalize(model.text_proj(sense_feats), dim=-1)
        sim = phrase_feats @ sense_feats.T / model.logit_scale

        # 确定sense
        logits_numpy = sim.softmax(1).detach().cpu().numpy()
        max_index = np.argmax(logits_numpy)
        sense = rel_gloss[max_index]

        sentence = 'A photo of ' + phrase + ', ' + sense
    else:
        sentence = 'A photo of ' + phrase

    return sentence