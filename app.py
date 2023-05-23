import torch.cuda
from flask import Flask, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH_SIZE = 64

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('hateBERT')

half_model = None
if torch.cuda.is_available():
    half_model = model.to('cuda').half()


@app.post("/query/basic")
def query_unoptimized():
    query = request.json['query']
    app.logger.info(f'Received query = {query} on basic endpoint')

    encoding = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    prediction = "positive" if model(**encoding).logits.argmax(-1).item() == 1 else "negative"
    return {
        "query": query,
        "prediction": prediction
    }


@app.post("/query/batch")
def query_batch():
    queries = request.json['queryList']

    app.logger.info(f'Received queryList with length = {len(queries)} on batch endpoint')

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    predictions = []
    batches = chunks(queries, BATCH_SIZE)

    for batch in batches:
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        _predictions = list(map(lambda prediction: "positive" if prediction == 1 else "negative",
                                model(**encodings).logits.argmax(-1).tolist()))
        predictions.extend(_predictions)
        app.logger.info(f'Total predictions completed = {len(predictions)}')

    query_prediction_map = list(map(lambda x: {"query": x[0], "prediction": x[1]}, zip(queries, predictions)))
    return query_prediction_map


@app.post("/query/fast")
def query_fast():
    if half_model is None:
        return {
            "error": "GPU not available for inference. Cannot perform fast queries"
        }

    query = request.json['query']

    app.logger.info(f'Received query = {query} on fast endpoint')

    encoding = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    prediction = "positive" if half_model(**encoding).logits.argmax(-1).item() == 1 else "negative"
    return {
        "query": query,
        "prediction": prediction
    }


@app.post("/query/fast/batch")
def query_fast_batch():
    if half_model is None:
        return {
            "error": "GPU not available for inference. Cannot perform fast queries"
        }

    queries = request.json['queryList']

    app.logger.info(f'Received queryList with length = {len(queries)} on fast batch endpoint')

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    predictions = []
    batches = chunks(queries, BATCH_SIZE)

    for batch in batches:
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        _predictions = list(map(lambda prediction: "positive" if prediction == 1 else "negative",
                                half_model(**encodings).logits.argmax(-1).tolist()))
        predictions.extend(_predictions)
        app.logger.info(f'Total predictions completed = {len(predictions)}')

    query_prediction_map = list(map(lambda x: {"query": x[0], "prediction": x[1]}, zip(queries, predictions)))
    return query_prediction_map


if __name__ == '__main__':
    app.run(port=8080, debug=True)
