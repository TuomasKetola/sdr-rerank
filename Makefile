
all::
	@echo potential models are:
	@grep "from rankers import" process.py
	@echo potential datasets are dbpedia, trec-web. Currently trec-web works a bit better

process::
	. bin/activate && python process.py -index_name $(index_name) -model_name $(model_name)

evaluate::
	. bin/activate && python evaluate.py -index_name $(index_name) -model_name $(model_name)

