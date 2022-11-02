import os
import csv
from collections import defaultdict

import numpy as np

from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score


def open_csv(path, delimiter=','):
    with open(path, 'r') as in_:
        reader = csv.reader(in_, delimiter=delimiter)
        f = list(reader)
    return f

def calc_ndcg(query_id, ranking, q_qrels,k):
    true_relevance = []
    prediction = []
    qrel_doc_ids = [x[2] for x in q_qrels]
    prediction_doc_ids = list(ranking.keys())
    for qrel in q_qrels:
        doc_id = qrel[2]
        true_relevance.append(float(qrel[3]))
        try:
            prediction.append(ranking[doc_id])
        except KeyError:
            prediction.append(0)

    for doc_id in list(set(prediction_doc_ids) - set(qrel_doc_ids)):
        prediction.append(ranking[doc_id])
        true_relevance.append(0)

    try:
        acc = ndcg_score(np.array([true_relevance]), np.array([prediction]),k=100)
        acc = round(acc,5)
    except:
        print('something wrong with ndcg calculation')
        exit()
    return acc


def calc_ap(query_id, q_ranking, q_qrels):
    true_relevance = []
    prediction = []
    qrel_doc_ids = [x[2] for x in q_qrels]
    prediction_doc_ids = list(q_ranking.keys())
    no_info_count = 0
    fail = False
    for qrel in q_qrels:
        doc_id = qrel[2]
        true_relevance.append(float(qrel[3]))
        try:
            prediction.append(q_ranking[doc_id])
        except KeyError:
            prediction.append(0)
    len_rel = len(prediction)
    for doc_id in list(set(prediction_doc_ids) - set(qrel_doc_ids)):
        prediction.append(q_ranking[doc_id])
        true_relevance.append(0)
    true_relevance = [0 if x < 1 else 1 for x in true_relevance]
    trues = np.array(true_relevance)
    preds = np.array(prediction)
    ap = average_precision_score(trues, preds)
    return ap


def evaluate(index_name, model_name):

    ndcgs = []
    aps = []
    
    qrel_dict = defaultdict(list)
    qrel_path = os.path.join('..', 'data', 'qrels', index_name + '-qrels.all')
    qrel_file = open_csv(qrel_path,delimiter='\t')
    for qrel in qrel_file:
        qrel[2] = qrel[2].lower()
        qrel_dict[qrel[0].strip()].append(qrel)

    results_dir = os.path.join('results', index_name, model_name)
    for query_file_name in os.listdir(results_dir):
        query_file_path = os.path.join(results_dir, query_file_name)
        results_file =  open_csv(query_file_path)
        results = {k:float(v) for k,v in dict(results_file).items()} 
        q_id = query_file_name[:-4]
        ndcg_q = calc_ndcg(q_id, results, qrel_dict[q_id],100)
        ap_q = calc_ap(q_id, results, qrel_dict[q_id])

        ndcgs.append(ndcg_q)
        aps.append(ap_q)

    print('MAP: {}\nNDCG@100: {}'.format(np.mean(aps), np.mean(ndcgs)))


if __name__ == '__main__':
    import argparse
    

    parser = argparse.ArgumentParser()

    parser.add_argument("-index_name", help="name of the index e.g. dabpedia", required=True)
    parser.add_argument("-model_name", help="which ranker model to use?", default="static")

    args = parser.parse_args()

    index_name = args.index_name
    model_name = args.model_name
    evaluate(index_name, model_name) 



