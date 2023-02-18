'''
 # Copyright
 # 2023/2/18
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from utils import _transform
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")

def load_data(args):
    # built the relation of idx and image_name (To avoid unknown bugs due to different image formats)
    train_image_dir = os.path.join(args.data_dir, 'kb.data', 'kbimages')
    imagelist = os.listdir(train_image_dir)
    image_map = {}
    for image in imagelist:
        idx = int(image.split('.')[1])
        image_map[idx] = image

    # load train metadata           ( ***Note that 'gloss' in codes denotes 'sense' in paper.)
    train_metadata = json.load(open(os.path.join(args.data_dir, 'kb.data', 'kb_metadata.json')))
    train_df = pd.DataFrame(train_metadata)
    train_df['gloss'] = train_df['gloss'].str.lower()
    train_df['image_path'] = train_df['image_idx'].apply(lambda x: os.path.join(args.data_dir, 'kb.data', 'kbimages', image_map.get(x)))

    # load official  train.data and gold img
    eval_data_path = os.path.join(args.data_dir, 'official.traindata', 'train.data.v1.txt')
    eval_gold_path = os.path.join(args.data_dir,'official.traindata','train.gold.v1.txt')
    eval_data_f = open(eval_data_path, 'r', encoding='utf-8')
    eval_content = eval_data_f.readlines()
    eval_gold_f = open(eval_gold_path, 'r', encoding='utf-8')
    eval_gold = eval_gold_f.readlines()
    eval_gold_imgs = []
    for gt in eval_gold:
        eval_gold_imgs.append(gt.replace('\n', ''))
    # integrate eval data
    eval_imgs_path = os.path.join(args.data_dir, 'official.traindata', 'train_images_v1')
    eval_data = integrate_evaldata(eval_content,eval_gold_imgs,eval_imgs_path)

    # load test data
    en_data_path = os.path.join(args.data_dir, 'official.testdata', 'en.test.data.txt')
    en_data_f = open(en_data_path, 'r', encoding='utf-8')
    en_content = en_data_f.readlines()
    fa_data_path = os.path.join(args.data_dir, 'official.testdata', 'fa.test.data.txt')
    fa_data_f = open(fa_data_path, 'r', encoding='utf-8')
    fa_content = fa_data_f.readlines()
    it_data_path = os.path.join(args.data_dir, 'official.testdata', 'it.test.data.txt')
    it_data_f = open(it_data_path, 'r', encoding='utf-8')
    it_content = it_data_f.readlines()

    # integrate test data
    test_imgs_path = os.path.join(args.data_dir, 'official.testdata', 'test_images')
    en_data,fa_data,it_data = integrate_testdata(args.data_dir,en_content,fa_content,it_content, test_imgs_path)

    return train_df,eval_data,en_data,fa_data,it_data

def integrate_evaldata(content,gold_imgs,eval_imgs_path):
    data = []
    for i in range(len(content)):
        c = content[i].replace('\n','').split('\t')
        temp = {}
        temp['amb'] = c[0]
        temp['phrase'] = c[1]

        candidate_imgs = c[2:]
        img_paths = []
        for cand in candidate_imgs:
            img_paths.append(os.path.join(eval_imgs_path,cand))
        temp['candidate_imgs'] = candidate_imgs
        temp['img_paths'] = img_paths
        temp['gold_img'] = gold_imgs[i]
        data.append(temp)
    data = pd.DataFrame(data)
    return data

def integrate_testdata(data_dir,en_content,fa_content,it_content, test_imgs_path):
    gold_root = os.path.join(data_dir,'official.testdata', 'test.data.v1.1.gold')
    en_gold = open(os.path.join(gold_root,'en.test.gold.v1.1.txt'),'r',encoding='utf-8').readlines()
    fa_gold = open(os.path.join(gold_root, 'fa.test.gold.v1.1.txt'), 'r', encoding='utf-8').readlines()
    it_gold = open(os.path.join(gold_root, 'it.test.gold.v1.1.txt'), 'r', encoding='utf-8').readlines()

    fa_translation = open(os.path.join(data_dir, 'fa_translation.txt'), 'r', encoding='utf-8').readlines()
    it_translation = open(os.path.join(data_dir, 'it_translation.txt'), 'r', encoding='utf-8').readlines()


    en_data = []
    for i in range(len(en_content)):
        cont = en_content[i].replace('\n', '').split('\t')
        temp = {}
        temp['amb'] = cont[0]
        temp['phrase'] = cont[1]

        candidate_imgs = cont[2:]
        img_paths = []
        for cand in candidate_imgs:
            img_paths.append(os.path.join(test_imgs_path, cand))
        temp['candidate_imgs'] = candidate_imgs
        temp['img_paths'] = img_paths

        temp['gold_img'] = en_gold[i].replace('\n','')
        temp['language'] = 'en'
        en_data.append(temp)
    en_data = pd.DataFrame(en_data)

    fa_data = []
    for i in range(len(fa_content)):
        cont = fa_content[i].replace('\n', '').split('\t')
        trans = fa_translation[i].lower().replace('\n', '').split('\t')
        temp = {}
        temp['amb'] = cont[0]
        temp['amb_trans'] = trans[1]
        temp['phrase'] = cont[1]
        temp['phrase_trans'] = trans[3]

        candidate_imgs = cont[2:]
        img_paths = []
        for cand in candidate_imgs:
            img_paths.append(os.path.join(test_imgs_path, cand))
        temp['candidate_imgs'] = candidate_imgs
        temp['img_paths'] = img_paths

        temp['gold_img'] = fa_gold[i].replace('\n','')
        temp['language'] = 'fa'
        fa_data.append(temp)
    fa_data = pd.DataFrame(fa_data)

    it_data = []
    for i in range(len(it_content)):
        cont = it_content[i].replace('\n', '').split('\t')
        trans = it_translation[i].lower().replace('\n', '').split('\t')
        temp = {}
        temp['amb'] = cont[0]
        temp['amb_trans'] = trans[1]
        temp['phrase'] = cont[1]
        temp['phrase_trans'] = trans[3]

        candidate_imgs = cont[2:]
        img_paths = []
        for cand in candidate_imgs:
            img_paths.append(os.path.join(test_imgs_path, cand))
        temp['candidate_imgs'] = candidate_imgs
        temp['img_paths'] = img_paths

        temp['gold_img'] = it_gold[i].replace('\n','')
        temp['language'] = 'it'
        it_data.append(temp)
    it_data = pd.DataFrame(it_data)

    return en_data,fa_data,it_data

def get_img_vec(image_path):
    image = Image.open(image_path).convert("RGB")
    image_vec = _transform(image)
    return image_vec

class train_dataset(Dataset):
    def __init__(self, train_metadata,args, mode):
        super(train_dataset, self).__init__()
        self.train_metadata = train_metadata
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.train_metadata)

    def __getitem__(self, item):
        ambiguous_word = self.train_metadata.loc[item]['amb']
        amb_trans = self.train_metadata.loc[item]['amb_trans'].lower()
        language = self.train_metadata.loc[item]['language']
        gloss = self.train_metadata.loc[item]['gloss']
        image_path = self.train_metadata.loc[item]['image_path']
        image = get_img_vec(image_path)
        image_idx = self.train_metadata.loc[item]['image_idx']
        gloss_idx = self.train_metadata.loc[item]['gloss_idx']

        return ambiguous_word,amb_trans,language,gloss,image,image_idx,gloss_idx

class eval_dataset(Dataset):
    def __init__(self, eval_data, args, mode):
        super(eval_dataset, self).__init__()
        self.eval_data = eval_data
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, item):
        amb = self.eval_data.loc[item]['amb']
        phrase = self.eval_data.loc[item]['phrase']
        language = 'en'
        img_names = self.eval_data.loc[item]['candidate_imgs']
        img_paths = self.eval_data.loc[item]['img_paths']
        candidate_images = torch.vstack([get_img_vec(path).unsqueeze(0) for path in img_paths])
        gold_img = self.eval_data.loc[item]['gold_img']

        return amb,phrase,language,img_names,candidate_images,gold_img

class test_dataset(Dataset):
    def __init__(self, test_data, args, mode):
        super(test_dataset, self).__init__()
        self.test_data = test_data
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, item):
        amb = self.test_data.loc[item]['amb']
        language = self.test_data.loc[item]['language']
        if language == 'en':
            phrase = self.test_data.loc[item]['phrase']
        else:
            phrase = self.test_data.loc[item]['phrase_trans'].lower()
        img_names = self.test_data.loc[item]['candidate_imgs']
        img_paths = self.test_data.loc[item]['img_paths']
        candidate_images = torch.vstack([get_img_vec(path).unsqueeze(0) for path in img_paths])
        gold_img = self.test_data.loc[item]['gold_img']

        return amb,phrase,language,img_names,candidate_images,gold_img

def collate_fn(batch):
    amb,phrase,language ,img_names,candidate_images,gold_img = zip(*batch)
    amb,phrase,language, img_names,candidate_images,gold_img = list(amb),list(phrase),list(language),list(img_names),list(candidate_images),list(gold_img)
    return amb,phrase,language,img_names,candidate_images,gold_img

def get_dataloader(args,data,mode):
    if mode =='train':
        mydataset = train_dataset(data, args=args, mode=mode)
        data_loader = DataLoader(mydataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
    elif mode == 'eval':
        mydataset = eval_dataset(data, args=args, mode=mode)
        data_loader = DataLoader(mydataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=collate_fn)
    elif mode == 'test_en' or mode == 'test_fa' or mode == 'test_it':
        mydataset = test_dataset(data, args=args, mode=mode)
        data_loader = DataLoader(mydataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers, collate_fn=collate_fn)
    return data_loader