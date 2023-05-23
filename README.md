# Flask App for Classifying Movie Review

* This is an application for classifying movie reviews into `positive` or `negative`
* This is a ReST based application which accepts JSON object of single query or list of queries
* There are multiple ReST endpoints based on which type of query stream user wants to handle
  * Single query or basic endpoint handles single query
  * Batch endpoint handles a list of queries
  * Fast endpoints uses faster (half-precision) GPU model to handle queries
    * Fast endpoints also have individual basic and batch handling endpoints 
* This application is run using `gunicorn` or `waitress` WSGI runners
* To run in multithreaded mode, we can specify the number of threads in the WSGI runner's command line arguments
* Using multithreading, the Flask app can process multiple basic requests or multiple chunks of batches parallely to give large throughput
* Number of processing threads is limited by underlying CPU architecture

## Jupyter Notebook Information

* The Jupyter notebook alongside the application contains code to train and validate best `mobilebert` model
* We have used HuggingFace Dataset, AutoModelForClassification and Trainer APIs for training the model
* The model was trained and validated on 50,000 samples of the famous IMDB dataset
* Out of multiple model checkpoints, the model with highest F1-Score `f1 = 0.887260` was selected

## How To Run

To run the application:
1. From the root of the repository `pip install -r requirements.txt`
2. Check your operating system
   1. Unix-based system: use gunicorn to run the application `gunicorn -w 6 app:app`. Here `-w 6` represents the number of threads to be instantiated.
   2. Windows-based system: use waitress to run the application `waitress-serve --threads=12 app:app`. The threads flag is self-explanatory.
3. Please note, as the model itself will be instantiated `n` times, where `n = number of threads`, make sure you have enough RAM to contain the model as well as input queries to be executed upon
4. This is a very resource intensive application, so please run `n` threads where `n = 2 * CPUs` available on your machine

## Benchmark Results

* The following table represents benchmarking results
* Single query requests are ran 50 times for benchmark result
* Multiple query requests, batch query requests and batch chunk query requests are ran 1 time for benchmark result
* The benchmark results are tested on following environment `Processor	Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2201 Mhz, 6 Core(s), 12 Logical Processor(s)`
* All benchmark results are created by `pytest-benchmark` package
* Although batch chunk query requests will have the most throughput in benchmark, our machine ran out of memory while executing batch chunk query requests
* Please ignore the `fast` model results as there are issues using the GPU model

| Name (time in ms) | Min | Max | Mean | StdDev | Median | IQR | Outliers | OPS | Rounds | Iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_multiple_on_fast_model_batch | 16.8111 (1.0) | 16.8111 (1.0) | 16.8111 (1.0) | 0.0000 (1.0) | 16.8111 (1.0) | 0.0000 (1.0) | 0;0 | 59.4845 (1.0) | 1 | 1 |
| test_single_on_fast_model | 142.5689 (8.48) | 245.5728 (14.61) | 166.4766 (9.90) | 15.9205 (inf) | 162.3542 (9.66) | 9.0740 (inf) | 4;4 | 6.0069 (0.10) | 50 | 1 |
| test_positive_query | 308.9990 (18.38) | 1,247.8521 (74.23) | 431.1538 (25.65) | 165.9252 (inf) | 370.4056 (22.03) | 129.2990 (inf) | 6;4 | 2.3194 (0.04) | 50 | 1 |
| test_negative_query | 315.1848 (18.75) | 989.8919 (58.88) | 488.3924 (29.05) | 156.4563 (inf) | 461.3186 (27.44) | 168.1544 (inf) | 9;4 | 2.0475 (0.03) | 50 | 1 |
| test_multiple_queries_on_basic_endpoint | 20,141.0605 (>1000.0) | 20,141.0605 (>1000.0) | 20,141.0605 (>1000.0) | 0.0000 (1.0) | 20,141.0605 (>1000.0) | 0.0000 (1.0) | 0;0 | 0.0496 (0.00) | 1 | 1 |
| test_multiple_queries_on_batch_endpoint | 23,525.1823 (>1000.0) | 23,525.1823 (>1000.0) | 23,525.1823 (>1000.0) | 0.0000 (1.0) | 23,525.1823 (>1000.0) | 0.0000 (1.0) | 0;0 | 0.0425 (0.00) | 1 | 1 |
| test_all_queries_on_basic_endpoint | 1,623,763.8486 (>1000.0) | 1,623,763.8486 (>1000.0) | 1,623,763.8486 (>1000.0) | 0.0000 (1.0) | 1,623,763.8486 (>1000.0) | 0.0000 (1.0) | 0;0 | 0.0006 | 1 | 1 |


## What changes did I do to achieve higher throughput

### 1. Better WSGI runner with multithreading capability
* A basic Flask application runs on single thread and serves requests sequentially
* By using a WSGI runner like gunicorn or waitress, we can specify multiple threads, which create multiple instances of application to serve parallely thus achieving higher throughput

### 2. Smaller model on GPU
* Normal HuggingFace model uses CPU to infer queries
* Using a GPU and `model.half()` to switch to Floating Point 16 instead of 32, reduces precision while increasing the speed of inference, thus getting higher throughput

### 3. Batch processing
* Single serving endpoint serves only one query at a time
* By using batch processing, we can utilize HuggingFace model's batch inference ability to infer multiple queries in a single request
* Furthermore, using WSGI multithreading and our batch processing, our throughput will multiply by a large factor

### 4. Choosing a smaller BERT family model
* I chose Google's `mobilebert` as model to infer queries
* `mobilebert` is smaller and thus faster than it's alternatives `bert` or `distillbert`
* Smaller model means we can use multiple instances of the model to serve parallely

## Future Improvements

* The user has to wait till the query or queries are completely inferred by the model
  * We can overcome this by using concept of background tasks
  * In background tasks, once a task is submitted by the user, a task id is returned to the user
  * The user can query the application using this task id to check whether the task is completed or not
  * A backlog of tasks is maintained by handlers such as Celery or RabbitMQ
  * User doesn't have to wait for the whole inference to take place
  * Helps in async application (frontend apps)
* A single WSGI runner acts as Single Point of Failure and cannot handle work under heavy load
  * We can change this by using Kubernetes Cluster with Load Balancer
  * Kubernetes can handle multiple containers of WSGI runners, thus multiplying the throughput further
  * Load balancer can balance user requests based on container load
  * Furthermore, if we want update the model, Kubernetes can handle rolling updates, thus user will experience zero downtime
  * There will be no Single Point of Failure (SPoF) as Kubernetes handles multiple containers, and whenever the app crashes, Kubernetes rolls out a new container