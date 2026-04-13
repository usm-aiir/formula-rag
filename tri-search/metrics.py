import ranx
# {'precision@5': 0.37241379310344824, 'ndcg@5': 0.21019198059968947}

RUN_PATH = "./data/clip_results.tsv"

print( ranx.evaluate(
    qrels=ranx.Qrels.from_file("./data/task1_2022.tsv", kind="trec"),
    run=ranx.Run.from_file(RUN_PATH, kind="trec"),
    metrics=["precision@5", "ndcg@5"]
))