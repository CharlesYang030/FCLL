'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
from utils import _transform
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from CLIP import clip
from PIL import Image
import copy
from transformers import logging
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class visual_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(visual_encoder_withparas, self).__init__()
        self.visual_encoder = clip_model.visual
        self.dtype = torch.float16

    def forward(self, image):
        img_feat,img_atts = self.visual_encoder(image.type(self.dtype))
        return img_feat.float(),img_atts.float()

class text_encoder_withparas(nn.Module):
    def __init__(self,clip_model):
        super(text_encoder_withparas, self).__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float16

    def forward(self,text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        att_texts_status = x

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x.float(),att_texts_status.float()


class FCLL(nn.Module):
    def __init__(self,args,metadata):
        super(FCLL, self).__init__()
        clip_model, _ = clip.load("ViT-B/32", device=args.device,jit=False)
        self.args = args
        self.metadata = metadata
        self.device = args.device
        self.clip_model = clip_model
        self.clip_tokenize = clip.tokenize

        self.width = 512
        self.embed_dim = 128
        self.dtype = torch.float16
        self.logit_scale = nn.Parameter(0.7 * torch.ones([]))

        # create normal encoders
        self.visual_encoder = visual_encoder_withparas(clip_model)
        self.vision_proj = nn.Linear(self.width, self.embed_dim)
        self.text_encoder = text_encoder_withparas(clip_model)
        self.text_proj = nn.Linear(self.width, self.embed_dim)

        # sense encoder
        self.sense_encoder = text_encoder_withparas(clip_model)

        # create momentum encoders
        self.visual_encoder_m = visual_encoder_withparas(clip_model)
        self.vision_proj_m = nn.Linear(512, self.embed_dim)
        self.text_encoder_m = text_encoder_withparas(clip_model)
        self.text_proj_m = nn.Linear(512, self.embed_dim)
        self.momentum = 0.995

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        self.agI_img_queue_size = 80000
        self.aiG_text_queue_size = 20000
        # create (amb+gloss)-images queue (agI)
        self.register_buffer("agI_image_queue", torch.randn(self.embed_dim, self.agI_img_queue_size))
        self.register_buffer("agI_idx_queue", torch.full((1, self.agI_img_queue_size), -100))  # gold images idx
        self.register_buffer("agI_ptr_queue", torch.zeros(1, dtype=torch.long))
        # create (amb+image)-glosses queue (aiG)
        self.register_buffer("aiG_gloss_queue", torch.randn(self.embed_dim, self.aiG_text_queue_size))
        self.register_buffer("aiG_idx_queue", torch.full((1, self.aiG_text_queue_size), -100))  # gold images idx
        self.register_buffer("aiG_ptr_queue", torch.zeros(1, dtype=torch.long))

        # transformer for multimodal feature
        self.transformer = Multimodal_transformers()
        self.att_images_align = nn.Linear(768, 512, bias=False)
        self.multimodal_align = nn.Linear(512, 128, bias=False)
        self.multimodal_length = 50 + 77
        self.transformer_width = 512
        self.ln = LayerNorm(self.transformer_width)

        # itm for Image-Text Matching
        self.itm_head = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.Linear(128, 2)
        )

    def forward(self,ambiguous_word,amb_trans,language,gloss,image,image_idx,gloss_idx,alpha):
        this_bs = len(ambiguous_word)  # 记录当前batch的长度
        rel_images,rel_img_cls,rel_gloss,rel_glo_cls,record_t2t = self.get_kb(ambiguous_word,language) #从kb中寻找相关知识
        ambiguous_word = self.amb_translation(ambiguous_word,amb_trans,language)  #将非英语的amb翻译为英文
        with torch.no_grad():
            self.logit_scale.clamp_(0.001,0.9)

        ###============== Step 1 : (amb+gloss)-images Momentum Contrastive Learning ===================###
        sentences = self.get_sentences(ambiguous_word,gloss)
        # 1.sentences(amb+gloss) feats
        sentences_feats, sentences_atts = self.text_encoder(self.clip_tokenize(sentences, truncate=True).to(self.device))
        sentences_feats = F.normalize(self.text_proj(sentences_feats), dim=-1)

        # 2. use momentum update (amb+gloss)-images queue
        gold_idx = gloss_idx.view(-1,1)
        agI_idx_all = torch.cat([rel_img_cls.view(-1, 1).t(), self.agI_idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(gold_idx, agI_idx_all).float()
        sim_targets_agI = pos_idx / pos_idx.sum(1, keepdim=True)
        with torch.no_grad():
            self._momentum_update()
            # use momentum encoder
            sentences_feats_m, _ = self.text_encoder_m(self.clip_tokenize(sentences, truncate=True).to(self.device))
            sentences_feats_m = F.normalize(self.text_proj_m(sentences_feats_m), dim=-1)

            agI_image_feats_m, agI_image_atts_m = self.visual_encoder_m(rel_images)
            agI_image_feats_m = F.normalize(self.vision_proj_m(agI_image_feats_m), dim=-1)
            agI_image_feats_all = torch.cat([agI_image_feats_m.t(), self.agI_image_queue.clone().detach()], dim=1)

            # 计算sentences_feats_m与image_feats_all的相似度，并得到agI_sim_ag2i_targets
            sim_ag2i_m = sentences_feats_m @ agI_image_feats_all / self.logit_scale
            sim_ag2i_targets = alpha * F.softmax(sim_ag2i_m, dim=1) + (1 - alpha) * sim_targets_agI

        sim_ag2i = sentences_feats @ agI_image_feats_all / self.logit_scale
        loss_SIC = -torch.sum(F.log_softmax(sim_ag2i, dim=1) * sim_ag2i_targets, dim=1).mean()
        self._dequeue_and_enqueue_agI(agI_image_feats_m, rel_img_cls)

        ###============== Step 2 : (amb+image)-glosses Momentum Contrastive Learning ===================###
        # 1.get amb_feats, then integrate amb_feats and image_feats by transformers
        amb_feats, amb_atts = self.text_encoder(self.clip_tokenize(ambiguous_word, truncate=True).to(self.device))
        _, image_atts = self.visual_encoder(image)
        image_atts = self.att_images_align(image_atts)

        multimodal_fea = torch.cat([amb_atts, image_atts], dim=1)
        multimodal_fea = self.multi_transformer(multimodal_fea)
        multimodal_fea = self.multimodal_align(self.ln(multimodal_fea.mean(1)))

        # 2. use momentum update (amb+image)-glosses queue
        aiG_idx_all = torch.cat([rel_glo_cls.view(-1, 1).t(), self.aiG_idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(gold_idx, aiG_idx_all).float()
        sim_targets_aiG = pos_idx / pos_idx.sum(1, keepdim=True)
        with torch.no_grad():
            # use momentum encoder
            amb_feats_m, amb_atts_m = self.text_encoder_m(
                self.clip_tokenize(ambiguous_word, truncate=True).to(self.device))

            aiG_image_feats_m, aiG_image_atts_m = self.visual_encoder_m(image)
            aiG_image_atts_m = self.att_images_align(aiG_image_atts_m)
            multimodal_fea_m = torch.cat([amb_atts_m, aiG_image_atts_m], dim=1)
            multimodal_fea_m = self.multi_transformer(multimodal_fea_m)
            multimodal_fea_m = self.multimodal_align(self.ln(multimodal_fea_m.mean(1)))

            aiG_gloss_feats_m, aiG_gloss_atts_m = self.text_encoder_m(
                self.clip_tokenize(rel_gloss, truncate=True).to(self.device))
            aiG_gloss_feats_m = F.normalize(self.text_proj_m(aiG_gloss_feats_m), dim=-1)
            aiG_gloss_feats_all = torch.cat([aiG_gloss_feats_m.t(), self.aiG_gloss_queue.clone().detach()], dim=1)

            # 计算multimodal_fea_m与aiG_gloss_feats_all的相似度，并得到aiG_sim_ai2g_targets
            sim_ai2g_m = multimodal_fea_m @ aiG_gloss_feats_all / self.logit_scale
            sim_ai2g_targets = alpha * F.softmax(sim_ai2g_m, dim=1) + (1 - alpha) * sim_targets_aiG

        sim_ai2G = multimodal_fea @ aiG_gloss_feats_all / self.logit_scale
        loss_ISC = -torch.sum(F.log_softmax(sim_ai2G, dim=1) * sim_ai2g_targets, dim=1).mean()
        self._dequeue_and_enqueue_aiG(aiG_gloss_feats_m, rel_glo_cls)

        ###============== Step 3 : Image-text Matching Predictor ===================###
        itm_labels = torch.eq(gold_idx, rel_img_cls.view(-1, 1).t()).long()
        _, itm_image_atts = self.visual_encoder(rel_images)
        itm_image_atts = self.att_images_align(itm_image_atts)

        loss_itm = 0
        for i in range(this_bs):
            # 得到每一组 （amb+gloss）的atts，为了对齐，复制十遍
            sentence_atts_itm = sentences_atts[i].unsqueeze(0).repeat(rel_images.size(0), 1, 1)

            # 得到多模态特征以及通过多模态transformer
            multimodal_fea_itm = torch.cat([sentence_atts_itm, itm_image_atts], dim=1)
            multimodal_fea_itm = self.multi_transformer(multimodal_fea_itm)
            multimodal_fea_itm = self.ln(multimodal_fea_itm.mean(1))

            # 通过itm_head判断每张候选图片与当前歧义短语是否匹配
            itm_logits = self.itm_head(multimodal_fea_itm)
            loss_itm += F.cross_entropy(itm_logits, itm_labels[i])

        ###============== Step 4 : Sense Auto-complementing Module ===================###
        concepts = self.get_concept(record_t2t)
        concepts_feats, _ = self.sense_encoder(self.clip_tokenize(concepts, truncate=True).to(self.device))
        concepts_feats = F.normalize(self.text_proj(concepts_feats), dim=-1)
        ttc_logits = concepts_feats @ aiG_gloss_feats_m.T / self.logit_scale
        labels = torch.arange(len(concepts),dtype=torch.int64,device=self.device)
        loss_TTC = F.cross_entropy(ttc_logits, labels)

        return loss_SIC,loss_ISC,loss_itm,loss_TTC

    def get_kb(self,ambiguous_word,language):
        ### 找出该batch中所有知识
        kb_record = self.metadata[self.metadata['amb'] == ambiguous_word[0]]
        kb_record = kb_record[kb_record['language'] == language[0]].reset_index(drop=True)
        for idx,amb in enumerate(ambiguous_word):
            if idx != 0:
                temp = self.metadata[self.metadata['amb'] == amb]
                temp = temp[temp['language'] == language[idx]]
                kb_record = pd.concat([kb_record,temp]).reset_index(drop=True)

        ### 记录图片和gloss
        rel_images = []
        rel_img_cls = []
        rel_gloss = []
        rel_glo_cls = []
        record_t2t = []
        for i in range(len(kb_record)):
            ### 记录batch的所有关联图片和图片的类别
            image_path = kb_record.loc[i]['image_path']
            image = Image.open(image_path)
            image_vec = _transform(image)
            rel_images.append(image_vec.reshape(1, 3, 224, 224).to(self.device))
            rel_img_cls.append(kb_record.loc[i]['gloss_idx'])

            ### 记录batch的所有gloss和gloss的类别
            gloss = kb_record.loc[i]['gloss']
            gloss_idx = kb_record.loc[i]['gloss_idx']
            if gloss_idx not in rel_glo_cls:
                rel_gloss.append(gloss)
                rel_glo_cls.append(gloss_idx)
                record_t2t.append(kb_record.loc[i]['amb'] + '###' + gloss)

        rel_images = torch.vstack(rel_images)
        rel_img_cls = torch.from_numpy(np.array(rel_img_cls)).to(self.device)
        rel_glo_cls = torch.from_numpy(np.array(rel_glo_cls)).to(self.device)
        return rel_images,rel_img_cls,rel_gloss,rel_glo_cls,record_t2t

    def amb_translation(self,ambiguous_word,amb_trans,language):
        temp = []
        for idx,lang in enumerate(language):
            if lang != 'en':
                temp.append(amb_trans[idx])
            else:
                temp.append(ambiguous_word[idx])
        return temp

    def get_sentences(self,ambiguous_word,gloss):
        sentences = []
        for i in range(len(ambiguous_word)):
            sent = 'A photo of ' + ambiguous_word[i] + ', ' + gloss[i]
            sentences.append(sent)

        return sentences

    def get_concept(self,record_t2t):
        en_stopwords = set(stopwords.words('english'))
        punct = [',', '.', '!', '?', '(', ')', ';', ':', '-', '/', '\\']
        concepts = []
        for cont in record_t2t:
            amb = cont.split('###')[0]
            gloss = cont.split('###')[-1]
            tokens = word_tokenize(gloss)
            gloss = [word for word in tokens if word not in en_stopwords and word not in punct]
            rand = np.random.randint(low=0,high=len(gloss),dtype=int)
            word = gloss[rand]
            concepts.append(amb+' '+word)
        return concepts

    def multi_transformer(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln(x).type(self.dtype)

        return x

    @torch.no_grad()
    def copy_params(self):
        for idx,model_pair in enumerate(self.model_pairs):
            if idx == 0:  # idx == 0 表示VisualTransformer
                for param, param_m in zip(model_pair[0].visual_encoder.parameters(), model_pair[1].visual_encoder.parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient
            else:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for idx, model_pair in enumerate(self.model_pairs):
            if idx == 0: # idx == 0 表示VisualTransformer
                for param, param_m in zip(model_pair[0].visual_encoder.parameters(), model_pair[1].visual_encoder.parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            else:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue_agI(self, image_feat, idxs):
        size = idxs.size(0)
        # gather keys before updating queue
        ptr = int(self.agI_ptr_queue)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + size) <= self.agI_img_queue_size:
            self.agI_image_queue[:, ptr:ptr + size] = image_feat.T
            self.agI_idx_queue[:, ptr:ptr + size] = idxs.T
        else:
            gap = self.agI_img_queue_size - ptr
            self.agI_image_queue[:, ptr:self.agI_img_queue_size] = image_feat[0:gap,:].T
            self.agI_idx_queue[:, ptr:self.agI_img_queue_size] = idxs[0:gap].T

            self.agI_image_queue[:, 0:size-gap] = image_feat[gap:gap+size, :].T
            self.agI_idx_queue[:, 0:size-gap] = idxs[gap:gap+size].T

        ptr = (ptr + size) % self.agI_img_queue_size # move pointer

        self.agI_ptr_queue[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_aiG(self, gloss_feat, idxs):
        size = idxs.size(0)
        # gather keys before updating queue
        ptr = int(self.aiG_ptr_queue)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + size) <= self.aiG_text_queue_size:
            self.aiG_gloss_queue[:, ptr:ptr + size] = gloss_feat.T
            self.aiG_idx_queue[:, ptr:ptr + size] = idxs.T
        else:
            gap = self.aiG_text_queue_size - ptr
            self.aiG_gloss_queue[:, ptr:self.aiG_text_queue_size] = gloss_feat[0:gap, :].T
            self.aiG_idx_queue[:, ptr:self.aiG_text_queue_size] = idxs[0:gap].T

            self.aiG_gloss_queue[:, 0:size - gap] = gloss_feat[gap:gap+size, :].T
            self.aiG_idx_queue[:, 0:size - gap] = idxs[gap:gap+size].T

        ptr = (ptr + size) % self.aiG_text_queue_size  # move pointer

        self.aiG_ptr_queue[0] = ptr



#########################   transformers part   #########################
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.d_model =d_model
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Multimodal_transformers(nn.Module):
    def __init__(self):
        super(Multimodal_transformers, self).__init__()
        transformer_width = 512
        transformer_layers = 12
        transformer_heads = 8
        self.multimodal_length = 50+77
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.multimodal_length, self.multimodal_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self,joint_fea):
        joint_att = self.transformer(joint_fea)
        return joint_att

