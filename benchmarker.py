import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8080/query"

dataset = pd.read_csv('IMDB Dataset.csv')


def benchmark_single_query(query):
    prediction = requests.post(BASE_URL + "/basic", json={"query": query})
    assert prediction.status_code == 200
    return prediction.json()


def benchmark_multiple_queries(queries):
    predictions = []
    for query in queries:
        prediction = requests.post(BASE_URL + "/basic", json={"query": query})
        assert prediction.status_code == 200
        predictions.append(prediction.json())
    return predictions


def benchmark_all_queries_fail_on_batch(queries):
    predictions = requests.post(BASE_URL + "/batch", json={"queryList": queries})
    assert predictions.status_code == 500  # This will try to allocate huge amount of memory for 50K+ queries and is supposed to fail
    return predictions.json()


def benchmark_batch_queries(queries):
    predictions = requests.post(BASE_URL + "/batch", json={"queryList": queries})
    assert predictions.status_code == 200
    return predictions.json()


def test_positive_query(benchmark):
    positive_query = dataset[dataset['sentiment'] == 'positive'].sample(1)['review'].values[0]
    result = benchmark.pedantic(benchmark_single_query, args=(positive_query,), rounds=50)
    assert result['prediction'] == 'positive'


def test_negative_query(benchmark):
    negative_query = dataset[dataset['sentiment'] == 'negative'].sample(1)['review'].values[0]
    result = benchmark.pedantic(benchmark_single_query, args=(negative_query,), rounds=50)
    assert result['prediction'] == 'negative'


def test_multiple_queries_on_basic_endpoint(benchmark):
    queries = dataset['review'][:100].values
    sentiment = dataset['sentiment'][:100].values
    predictions = benchmark.pedantic(benchmark_multiple_queries, args=(queries,), )
    count_correct = 0
    for x, y in zip(sentiment, map(lambda pred: pred['prediction'], predictions)):
        if x == y:
            count_correct += 1
    print(count_correct)
    assert count_correct / len(queries) > 0.8


def test_multiple_queries_on_batch_endpoint(benchmark):
    queries = dataset['review'][:50].values.tolist()
    sentiment = dataset['sentiment'][:50].values
    predictions = benchmark.pedantic(benchmark_batch_queries, args=(queries,), rounds=1)
    count_correct = 0
    for x, y in zip(sentiment, predictions):
        if x == y['prediction']:
            count_correct += 1
    print(count_correct)
    assert count_correct / len(queries) > 0.8


def test_all_queries_on_batch_endpoint(benchmark):
    queries = dataset['review'].values.tolist()
    sentiment = dataset['sentiment'].values
    predictions = benchmark.pedantic(benchmark_batch_queries, args=(queries,), rounds=1)
    count_correct = 0
    for x, y in zip(sentiment, predictions):
        if x == y['prediction']:
            count_correct += 1
    print(count_correct)
    assert count_correct / len(queries) > 0.8
