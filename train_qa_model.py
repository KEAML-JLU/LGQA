import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np
from qa_baselines import QA_baseline, QA_lm, QA_embedkgqa, QA_cronkgqa
from qa_sub import QA_Sub
from qa_datasets import QA_Dataset, QA_Dataset_Sub, QA_Dataset_Baseline
from torch.utils.data import Dataset, DataLoader
import utils
from tqdm import tqdm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info, get_neighbours, check_triples, rerank_ba,rerank_fl, rerank_st, rerank_tj
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='tcomplex.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)
parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)

parser.add_argument(
    '--model', default='sub', type=str,
    help="Which model to use."
)


parser.add_argument(
    '--subgraph_reasoning',
    help="whether use subgraph reasoning module",
    action="store_true"
)

parser.add_argument(
    '--time_sensitivity',
    help="whether use time sensitivity module",
    action="store_true"
)

parser.add_argument(
    '--aware_module',
    help="whether use aware module",
    action="store_true"
)

parser.add_argument(
    '--khop', default=3, type=int,
    help="khop subgraph extracted by kg"
)

parser.add_argument(
    '--dataset_name', default='wikidata_big', type=str,
    help="Which dataset."
)

parser.add_argument(
    '--supervision', default='hard', type=str,
    help="Which supervision to use."
)

parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)

parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=10, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)

parser.add_argument(
    '--valid_freq', default=1, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--batch_size', default=150, type=int,
    help="Batch size."
)

parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)

parser.add_argument(
    '--frozen', default=1, type=int,
    help="Whether entity/time embeddings are frozen or not. Default frozen."
)

parser.add_argument(
    '--lm_frozen', default=1, type=int,
    help="Whether language model params are frozen or not. Default frozen."
)

parser.add_argument(
    '--lr', default=2e-4, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)

parser.add_argument(
    '--eval_split', default='valid', type=str,
    help="Which split to validate on"
)

parser.add_argument(
    '--lm', default='distill_bert', type=str,
    help="Lm to use."
)
parser.add_argument(
    '--fuse', default='add', type=str,
    help="For fusing time embeddings."
)
parser.add_argument(
    '--extra_entities', default=False, type=bool,
    help="For some question types."
)
parser.add_argument(
    '--corrupt_hard', default=0., type=float,
    help="For some question types."
)

parser.add_argument(
    '--test', default="test", type=str,
    help="Test data."
)

args = parser.parse_args()
print_info(args)

with open('./LGQA/saved_pkl/e2rt.pkl', 'rb') as f:
    e2rt = pickle.load(f)
with open('./LGQA/saved_pkl/event2time.pkl', 'rb') as f:
    event2time = pickle.load(f)
with open('./LGQA/saved_pkl/e2tr.pkl', 'rb') as f:
    e2tr = pickle.load(f)


def eval(qa_model, dataset, batch_size=128, split='valid', k=200, subgraph_reasoning=False):
    num_workers = 4
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k  # not change name in fn signature since named param used in places
    k_list = [1, 10]
    # max num of subgraph candidate answers
    max_k = 100
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    for i_batch, a in enumerate(loader):
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-2]  # last one assumed to be target
        scores = qa_model.forward(a, dataset.dgl_graph)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['type']
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            entity_time_type = question['answer_type']
            predicted = topk_answers[i]

            if subgraph_reasoning:
                neighbours = []
                if question['type'] == 'before_after':
                    if 'event_head' in question['annotation'].keys():
                        event = question['annotation']['event_head']
                        if event[0] !='Q':
                            t = int(event)
                        else:
                            t = int(list(event2time[event])[0][3])
                        d = list(dataset[i])
                        d[7] = t
                        d[8] = t
                        predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'], d)[:k]
                    else:
                        predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'],
                                              dataset[i])[:k]

                if question['type'] == 'first_last':
                    if question['answer_type'] == 'entity':
                        predicted = rerank_fl(predicted, question['entities'], question['annotation']['adj'],
                                              dataset[i])[:k]

                if question['type'] == 'simple_time':
                    predicted = rerank_st(predicted, question['entities'], None,
                                          dataset[i])[:k]

                if question['type'] == 'time_join':
                    if len(question['entities']) == 2:
                        if 'event_head' in question['annotation'].keys():
                            event = question['annotation']['event_head']
                            if event[0] !='Q':
                                t = int(event)
                            else:
                                t = int(list(event2time[event])[0][3])
                            d = list(dataset[i])
                            d[7] = t
                            d[8] = t
                            predicted = rerank_tj(predicted, question['entities'], None, d)[:k]
                        else:
                            predicted = rerank_tj(predicted, question['entities'], None,
                                                  dataset[i])[:k]
            predicted = predicted[:k]
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
            question_types_count[question_type].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1

        eval_accuracy = hits_at_k / total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, entity_time_count]:
            # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value) / len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type=key,
                    hits_at_k=round(hits_at_k, 3),
                    num_questions=len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log



def append_log_to_file(eval_log, epoch, filename):
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()


def train(qa_model, dataset, valid_dataset, args, result_filename=None):
    num_workers = 5
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=dataset._collate_fn)
    max_eval_score = 0
    if args.save_to == '':
        args.save_to = 'gtr'
    if result_filename is None:
        result_filename = 'results/{dataset_name}/{model_file}.log'.format(
            dataset_name=args.dataset_name,
            model_file=args.save_to
        )
    checkpoint_file_name = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.save_to
    )

    if args.load_from == '':
        print('Creating new log file')
        f = open(result_filename, 'a+')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Config: \n')
        for key, value in vars(args).items():
            key = str(key)
            value = str(value)
            f.write('%s:\t%s\n' % (key, value))
        f.write('\n')
        f.close()

    max_eval_score = 0.

    print('Starting training')
    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for i_batch, a in enumerate(loader):
            qa_model.zero_grad()

            answers_khot = a[-2]  # last one assumed to be target
            scores = qa_model.forward(a, dataset.dgl_graph)

            loss = qa_model.loss(scores, answers_khot.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, args.max_epochs))
            loader.update()

        print('Epoch loss = ', epoch_loss)
        if (epoch + 1) % args.valid_freq == 0:
            print('Starting eval')
            eval_score, eval_log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size,
                                        split=args.eval_split, k=args.eval_k, subgraph_reasoning=args.subgraph_reasoning)
            if eval_score > max_eval_score:
                print('Valid score increased')
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            # log each time, not max
            # can interpret max score from logs later
            append_log_to_file(eval_log, epoch, result_filename)


def save_model(qa_model, filename):
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return


if args.model != 'embedkgqa':  # TODO this is a hack
    tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))
    print('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file))
else:
    tkbc_model = loadTkbcModel_complex('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

train_split = 'train'
test = args.test
# train_split = 'train_aware3'
# test = 'test_aware3'
test = 'test'


if args.model == 'bert' or args.model == 'roberta':
    qa_model = QA_lm(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model == 'embedkgqa':
    qa_model = QA_embedkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model == 'cronkgqa' and args.supervision != 'hard':
    qa_model = QA_cronkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    # valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model == 'sub':  # supervised models
    qa_model = QA_Sub(tkbc_model, args)
    if args.mode == 'train':
        dataset = QA_Dataset_Sub(split=train_split, dataset_name=args.dataset_name, args=args, tkbc_model=tkbc_model)
    # valid_dataset = QA_Dataset_TempoQR(split=args.eval_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_Sub(split=test, dataset_name=args.dataset_name, args=args, tkbc_model=tkbc_model)

else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)
if args.load_from != '':

    filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.load_from
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
    # TKG embeddings
    tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
    dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))


    qa_model.tkbc_model = tkbc_model
    num_entities = tkbc_model.embeddings[0].weight.shape[0]
    num_times = tkbc_model.embeddings[2].weight.shape[0]
    ent_emb_matrix = tkbc_model.embeddings[0].weight.data
    time_emb_matrix = tkbc_model.embeddings[2].weight.data

    full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
    # +1 is for padding idx
    qa_model.entity_time_embedding = torch.nn.Embedding(num_entities + num_times + 1,
                                              qa_model.tkbc_embedding_dim,
                                              padding_idx=num_entities + num_times)
    qa_model.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)

    for param in tkbc_model.parameters():
        param.requires_grad = False

else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

if args.mode == 'eval':
    score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k=args.eval_k,subgraph_reasoning = args.subgraph_reasoning)
    exit(0)

result_filename = 'results/{dataset_name}/{model_file}.log'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)

train(qa_model, dataset, test_dataset, args, result_filename=result_filename)


print('Training finished')
