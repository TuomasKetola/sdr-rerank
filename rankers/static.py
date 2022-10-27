

def rankerFunction(fields, query_data):

    arr_data = query_data['numpy_data']
    scores = arr_data['bm25_scores_arr'].sum(axis=1)
    doc_ids = arr_data['doc_ids']
    doc_score_lst = list(zip(doc_ids,scores.tolist()))
    return doc_score_lst


def rerank(query_data, datasetInfo):
    q_id = query_data['query_id'].strip()
    fields = datasetInfo['fields']
    doc_score_lst = rankerFunction(fields, query_data)
    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return q_id, doc_score_lst


