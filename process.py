import os
from pathlib import Path
import csv
import json
import pickle
import numpy as np
from rankers import static, icfwLA, BM25Elastic


def open_json(path):
    with open(path, 'r') as in_:
        f = json.load(in_)
    return f

def open_pickle(path):
    with open(path, 'rb') as in_:
        f = pickle.load(in_)
    return f


def rerank_query(query_data, datasetInfo,model):
    q_id, doc_score_lst = model.rerank(query_data,datasetInfo) 
    save_dir = os.path.join('results', index_name, model_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_str = os.path.join(save_dir, q_id+'.csv')
    with open(save_str, 'w') as out:
        writer = csv.writer(out)
        writer.writerows(doc_score_lst)



def rerank_query_set(data, datasetInfo, model_name):
    models = {
            'static': static,
            'icfwLA': icfwLA,
            'BM25Elastic': BM25Elastic,
            }
    model = models[model_name]
    for query_data in data:
        rerank_query(query_data, datasetInfo, model)




if __name__ == '__main__':
    import argparse
    

    parser = argparse.ArgumentParser()

    parser.add_argument("-index_name", help="name of the index e.g. dabpedia", required=True)
    parser.add_argument("-model_name", help="which ranker model to use?", default="static")
    parser.add_argument('-options', type=str, help='set of options in json format: {"k_1":1, b:...}', default="{}")

    args = parser.parse_args()

    index_name = args.index_name
    model_name = args.model_name

    dataDir = os.path.join('..','data',index_name)    

    data = []
    datasetInfo = open_json('datasetInfo.json')[index_name]
    for file_name in os.listdir(dataDir):
        if '.pickle' in file_name:
            path = os.path.join(dataDir,file_name)
            f = open_pickle(path)
            data.append(f)
    rerank_query_set(data, datasetInfo, model_name)

