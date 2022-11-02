import numpy as np

def rankerFunction(fields, query_data, k_1=1.6, b=0.8, arr_based=True):
    if arr_based:

        arr_data = query_data['numpy_data']
        tfs = arr_data['tf_arr'].sum(axis=2) # term frequencies / squueze field dimention out
        dls = arr_data['fl_arr'].sum(axis=2) # doc lenghts  / squeeze field dimension out
        avgdl = np.mean(dls)
        dfs = arr_data['global_df_arr']  # df array of flattened collection
        N = arr_data['Nfs'].max() # number of documents
        idfs = np.log(1 + (N - dfs + 0.5) / (dfs + 0.5) ) # directly from elastic docs

        bm25numerator = tfs * (k_1+1)
        bm25denumerator = tfs + k_1 * (1 - b + b * (dls / avgdl) )

        bm25TF = bm25numerator / bm25denumerator

        bm25TFIDF = bm25TF * idfs
        scores = bm25TFIDF.sum(axis=1).tolist()
    else:
        # Here I can write up an example of non array / matrix based calculations
        pass
        

    doc_ids = arr_data['doc_ids']
    doc_score_lst = list(zip(doc_ids,scores))
    return doc_score_lst


def rerank(query_data, datasetInfo):
    q_id = query_data['query_id'].strip()
    fields = datasetInfo['fields']
    doc_score_lst = rankerFunction(fields, query_data)
    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return q_id, doc_score_lst


