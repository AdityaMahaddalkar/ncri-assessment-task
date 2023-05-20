from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8080/query"

dataset = pd.read_csv('IMDB Dataset.csv')


def benchmark_single_query(query):
    prediction = requests.post(BASE_URL + "/basic", json={"query": query})
    return prediction.json()


def benchmark_multiple_queries(queries):
    predictions = {}

    def assign(query):
        predictions[query] = benchmark_single_query(query)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(assign, queries)

    return predictions


def benchmark_all_queries_fail_on_batch(queries):
    predictions = requests.post(BASE_URL + "/batch", json={"queryList": queries})
    return predictions.json()


def benchmark_batch_queries(queries):
    predictions = requests.post(BASE_URL + "/batch", json={"queryList": queries})
    return predictions.json()


def benchmark_multiple_query_chunks_with_batch(query_chunks):
    predictions = []

    def assign(query_chunk):
        _predictions = benchmark_batch_queries(query_chunk)
        predictions.extend(_predictions)

    with ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(assign, query_chunks)

    return predictions


def benchmark_single_query_on_fast(query):
    prediction = requests.post(BASE_URL + "/fast", json={"query": query})
    return prediction.json()


def benchmark_multiple_queries_on_fast_batch(queries):
    predictions = []
    prediction = requests.post(BASE_URL + "/fast/batch", json={"queryList": queries})
    predictions.append(prediction.json())
    return predictions


def test_positive_query(benchmark):
    positive_query = dataset[dataset['sentiment'] == 'positive'].sample(1)['review'].values[0]
    result = benchmark.pedantic(benchmark_single_query, args=(positive_query,), rounds=50)


def test_negative_query(benchmark):
    negative_query = dataset[dataset['sentiment'] == 'negative'].sample(1)['review'].values[0]
    result = benchmark.pedantic(benchmark_single_query, args=(negative_query,), rounds=50)


def test_multiple_queries_on_basic_endpoint(benchmark):
    queries = dataset['review'][:100].values
    sentiment = dataset['sentiment'][:100].values
    predictions = benchmark.pedantic(benchmark_multiple_queries, args=(queries,), rounds=1)


def test_multiple_queries_on_batch_endpoint(benchmark):
    queries = dataset['review'][:100].values.tolist()
    sentiment = dataset['sentiment'][:100].values
    predictions = benchmark.pedantic(benchmark_batch_queries, args=(queries,), rounds=1)


def test_all_queries_on_basic_endpoint(benchmark):
    queries = dataset['review'][:10000].values
    sentiment = dataset['sentiment'][:10000].values
    predictions = benchmark.pedantic(benchmark_multiple_queries, args=(queries,), rounds=1)


def test_all_queries_on_batch_endpoint(benchmark):
    queries = dataset['review'][:10000].values.tolist()
    sentiment = dataset['sentiment'][:10000].values
    predictions = benchmark.pedantic(benchmark_batch_queries, args=(queries,), rounds=1)


def test_all_queries_on_parallel_batch_endpoint(benchmark):
    queries = dataset['review'][:10000].values.tolist()

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    query_chunks = chunks(queries, 1000)

    predictions = benchmark.pedantic(benchmark_multiple_query_chunks_with_batch, args=(query_chunks,), rounds=1)


def test_single_on_fast_model(benchmark):
    positive_query = dataset[dataset['sentiment'] == 'positive'].sample(1)['review'].values[0]
    result = benchmark.pedantic(benchmark_single_query, args=(positive_query,), rounds=50)


def test_multiple_on_fast_model_batch(benchmark):
    queries = dataset['review'][:100].values.tolist()
    sentiment = dataset['sentiment'][:100].values
    predictions = benchmark.pedantic(benchmark_multiple_queries_on_fast_batch, args=(queries,), rounds=1)
