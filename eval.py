"""
Run evaluation with saved models.
"""
import random
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
from data.loader import DataLoader,map_to_ids,get_long_tensor
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,default="saved_models/01",help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model_aug.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test_rev_coref', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
print(torch.cuda.device_count())
args.cuda=True
with open('err.json','r') as f:
    data=json.load(f)
args.cpu=False
torch.manual_seed(args.seed)
random.seed(1234)
# if args.cpu:
#     args.cuda = False
# elif args.cuda:
torch.cuda.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"
print(torch.cuda.device_count())
# load opt
args.model_dir='saved_models/02'
#model_file=args.model_dir+'/best_model_aug.pt'
# print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
# trainer = GCNTrainer(opt)
# trainer.load(model_file)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
entitie_pairs = constant.REL_ENTITIES
predictions = []
key=[]
# for entitie_pair in entitie_pairs:
#     subj=entitie_pair[0]
#     obj=entitie_pair[-1]
with torch.cuda.device(1):
    #load vocab
    print(torch.cuda.current_device())
    vocab_file = args.model_dir + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    labels = constant.LABEL_TO_ID.keys()
    lbstokens = []
    for lbs in labels:
        lb = []
        if lbs == 'no_relation':
            subj = ['<UNK>']
            rels = '<UNK>'
        else:
            subj, rel = lbs.split(":")
            subj = (['SUBJ-PERSON'] if subj == 'per' else ['SUBJ-ORGANIZATION'])
            rels = rel.split("_")
        lb += map_to_ids(subj, vocab.word2id)
        lb += map_to_ids(rels, vocab.word2id)
        lbstokens.append(lb)
    lbstokens = get_long_tensor(lbstokens, len(lbstokens))
    subj="PERSON"
    obj="NUMBER"
#     #
#     # assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
#     #
#     # # load data
#
#     # print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
#     #batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True,corefresolve=False)
#     #train_batch=DataLoader(data_file,opt['batch_size'],opt,vocab,evaluation=True,corefresolve=False,is_aug=True)
#     # train_aug_batch=DataLoader(data_file,opt['batch_size'],opt,vocab,evaluation=True,corefresolve=False)
#     # helper.print_config(opt)

#     SUBJ_LIST=list(constant.SUBJ_NER_TO_ID.keys())
#     OBJ_LIST=list(constant.OBJ_NER_TO_ID.keys())




    #train_batch.setisEval(True)
    # probs={}
    # for i,batch in enumerate(train_batch):
    #     preds, prob, _,sample = trainer.predict(batch)
    #     probs=dict(probs,**prob)
    # # probs = sorted(probs.items(),key=lambda item:max(item[1]))
    # train_batch.LabeledAugData(probs)
    # train_batch.setisEval(False)
    # for i, b in enumerate(batch_iter):
    #     preds, prob, _,sample = trainer.predict(b)
    #     predictions += preds
    #     all_probs += prob
    #     samples = dict(samples, **sample)
    # SUBJ_LIST=list(constant.SUBJ_NER_TO_ID.keys())
    # OBJ_LIST=list(constant.OBJ_NER_TO_ID.keys())
    # SUBJ_LIST.remove('<PAD>')
    # SUBJ_LIST.remove('<UNK>')
    # OBJ_LIST.remove('<PAD>')
    # OBJ_LIST.remove('<UNK>')

    # print(torch.cuda.device_count())
    # with torch.cuda.device(1):
    #     for subj in SUBJ_LIST:
    #         for obj in OBJ_LIST:
    # print("eval samples of subj:"+subj+" obj:"+obj)
    # args.model_dir = 'saved_models/02'
    # if os.path.exists(args.model_dir+'/'+subj+"_"+obj+"_"+"best_model.pt"):
    #     model_file = args.model_dir +'/'+subj+"_"+obj+"_"+"best_model.pt"
    # else:
    #     model_file = args.model_dir + '/best_model.pt'
    model_file=args.model_dir+'/best_model' \
                              '.pt'
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
    trainer = GCNTrainer(opt,lbstokens=lbstokens)
    trainer.load(model_file)
    batch = DataLoader([data_file], opt['batch_size'], opt, vocab, evaluation=True, corefresolve=True)
    batch_iter = tqdm(batch)

    all_probs = []
    samples = []
    for i, b in enumerate(batch_iter):
        preds, probs, _,sample= trainer.predict(b)
        predictions += preds
        all_probs += probs
        # effsum+=lab_eff
        # lab_nums+=lab_num
        samples=samples+sample

    key+=batch.gold()

    # with open('samples.json','w') as f:
    #     json.dump(samples,f,indent=4)

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch, predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

