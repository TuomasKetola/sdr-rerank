# sdr-rerank
Repo for various reranking algorithms. The underlying data is assumed to be structured, but atomic models can easily be used as well.

## Running rerankers and evalution
run make for basic instructions

```make process index_name={index_name} model_name={model_name}```
and
```make evaluation index_name={index_name} model_name={model_name}```

Alternatively start a virtualenviroment using: ```source bin/activate``` and run
```python process.py -index_name {index_name} -model_name {model_name}```
and
```python evaluate.py -index_name {index_name} -model_name {model_name}```

## Adding new rerankers:

There are 3 steps that need to be taken to process data with a new model and to evaluate the results
1. add a new model to ```/rankers``` director. The BM25Elastic.py script is a good place to start
2. import the ranker to process.py by editing the line ```from rankers import static, icfwLA, BM25Elastic```
3. add the model to the models dictionary in process.py
