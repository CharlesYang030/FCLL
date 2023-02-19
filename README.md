## FCLL
### FCLL: A **F**ine-grained **C**ontrastive **L**anguage-Image **L**earning Model for Cross-language Visual Word Sense Disambiguation

### Announcement: Visual Word Sense Disambiguation (Visual WSD) is proposed by [SemEval-2023 Task 1](https://raganato.github.io/vwsd/) for the first time. Thanks to Raganato *et al.* for leading us to recognize this multimodal-multilingual field.

---

### Environment
Our code has been implemented on Pytorch 1.8.1. To reproduce our experiments, please run: <pre/>pip install -r requirements.txt</pre> 

### Usage
#### 1.Download the datasets: 
Please click on the following links to download our V-WSD KB and the official training/test set, and then create a new `. /data` folder in the project directory.

Dataset | Num. atw | Language of atw | Num. phrase | Language of phrase | Num. image | Correspondence | Size | Link
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
V-WSD KB | 12956 | EN, FA, IT | 20904 | EN | 97267 | 1-n-n | 114GB | [Download]()
Official training set | 12869 | EN | 12869 | EN | 12999 | 1-1-1 | 16.8GB | [Download](https://1drv.ms/u/s!AgvzREJAm7GyhEH4UfA4QFhhCM7E)
Official test set | 968 | EN, FA, IT | 968 | EN, FA, IT | 8100 | 1-1-1 | 10.4GB | [Download](https://1drv.ms/u/s!AgvzREJAm7GyhEBWWGyB5DkfT-fS)

#### 2.Translate the non-English texts:
In the official test set, Non-English ambiguous target words and phrases should be translated into English text, stored in `fa_translation.txt` and `it_translation.txt` separately, as the following format:

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

#### 4.To evaluating, please run:
```.
python main.py --eval_batch_size 16 --use_checkpoint --evaluate 
```
---

### Acknowledgement
FCLL is inspired by CLIP and MoCo, simultaneously relies on resources from BLIP, BabelNet. The original authors and their open-sourcing is appreciated.
