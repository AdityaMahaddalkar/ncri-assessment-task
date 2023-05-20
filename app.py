from flask import Flask, request

from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
model = AutoModelForSequenceClassification.from_pretrained('./best-f1-imdb')


@app.post("/query/basic")
def query_unoptimized():
    query = request.json['query']
    encoding = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    prediction = "positive" if model(**encoding).logits.argmax(-1).item() == 1 else "negative"
    return {
        "query": query,
        "prediction": prediction
    }


@app.post("/query/batch")
def query_batch():
    queries = request.json['queryList']
    encodings = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=128)
    predictions = list(map(lambda prediction: "positive" if prediction == 1 else "negative",
                           model(**encodings).logits.argmax(-1).tolist()))
    query_prediction_map = list(map(lambda x: {"query": x[0], "prediction": x[1]}, zip(queries, predictions)))
    return query_prediction_map


if __name__ == '__main__':
    app.run(port=8080)
