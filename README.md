## FCLL: A Fine-grained Contrastive Language-Image Learning Model for Cross-language Visual Word Sense Disambiguation
FCLL is a reliable, verifiable and extensible model that has a powerful synthesis of reasoning from text to image and from image to text on fine-grained image-text knowledge. The results on the benchmark datasets of SemEval-2023 Task 1 show that FCLL performs well in the three tracks of English, Farsi and Italian, with an average hit rate at 1 of 72.56% and an average mean reciprocal rank of 82.22%, ranking first in the overall evaluation.

#### Approach:
 ![image](https://github.com/CharlesYang030/FCLL/blob/main/FCLL.png)

#### Announcement: Visual Word Sense Disambiguation (Visual WSD) is proposed by [SemEval-2023 Task 1](https://raganato.github.io/vwsd/) for the first time. Thanks to Raganato *et al.* for leading us to recognize this multimodal-multilingual field.

---

### Environment
Our code has been implemented on Pytorch 1.8.1. To reproduce our experiments, please run: <pre/>pip install -r requirements.txt</pre> 

### Usage
#### 1.Download the datasets: 
Please click on the following links to download the official training/test set and our V-WSD KB, and then create a new `. /data` folder in the project directory.

Dataset | Num. atw | Language of atw | Num. phrase | Language of phrase | Num. image | Correspondence | Size | Link
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
Official training set | 12869 | EN | 12869 | EN | 12999 | 1-1-1 | 16.8GB | [Download](https://1drv.ms/u/s!AgvzREJAm7GyhEH4UfA4QFhhCM7E)
Official test set | 968 | EN, FA, IT | 968 | EN, FA, IT | 8100 | 1-1-1 | 10.4GB | [Download](https://1drv.ms/u/s!AgvzREJAm7GyhEBWWGyB5DkfT-fS)
V-WSD KB | 12956 | EN, FA, IT | 20904 | EN | 97267 | 1-n-n | 114GB | [Download]()

#### 2.Translate the non-English texts:
In the official test set, Non-English ambiguous target words and phrases should be translated into English text, stored in `fa_translation.txt` and `it_translation.txt` separately, as the following format ('\t' is uesd as the delimiter):

```
(an instance in Farsi)
برنج‎	brass	فلز برنج	brass
(an instance in Italian)
gomma	eraser	gomma per smacchiare	eraser for stain removal
```

Note that after downloading and translating, please place the above files as follows:<br>
```.
(the folder tree)
|—— FCLL
|    |—— data
|         |—— kb.data
|              |—— ...
|         |—— official.traindata
|              |—— ...
|         |—— official.testdata
|              |—— ...
|         |—— fa_translation.txt
|         |—— it_translation.txt
|    |—— CLIP
|    |—— ...
```

#### 3.To train from the scratch, please run:
```.
python main.py --train_batch_size 2 --num_workers 4
```
In training, the checkpoint of the best model will be saved into `./save_model`, the log of the training process will be saved into `./log`, and the outputs of each epoch will be saved into `./result`.

#### 4.To evaluate using the best checkpoint, please run:
```.
python main.py --eval_batch_size 16 --use_checkpoint --evaluate 
```
---

### Acknowledgement
FCLL is inspired by [CLIP](https://github.com/openai/CLIP) and [MoCo](https://github.com/facebookresearch/moco), simultaneously relies on resources from [BLIP](https://github.com/salesforce/BLIP) and [BabelNet](https://babelnet.org/). The original authors and their open-sourcing are appreciated.
