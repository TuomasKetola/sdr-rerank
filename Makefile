
all::
	@echo potential models are:
	@grep "from rankers import" process.py
	@echo potential datasets are dbpedia, trec-web. Currently trec-web works a bit better
	@echo to process data: "make process index_name=index_name model_name=model_name"
	@echo to evaluate: "make process index_name=index_name model_name=model_name"

process::
	. bin/activate && python process.py -index_name $(index_name) -model_name $(model_name)

evaluate::
	. bin/activate && python evaluate.py -index_name $(index_name) -model_name $(model_name)

