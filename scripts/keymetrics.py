#!/usr/bin/env python3
import json

metrics = None
with open("data/test.csv___full_metrics.json") as f:
    metrics = json.load(f)

for i in range(2, 11):
    divn = metrics["diversity"][str(i)]["correct"]["binary"]["maxavgpair"]
    print("Diversity@%d:\t%0.3f" % (i, divn))

for i in range(10):
    print("NDCG@%d:\t%0.3f" % (i+1, metrics["ndcg"][i]))

for i in range(10):
    print("MAP@%d:\t%0.3f" % (i+1, metrics["avgreward"][i]))
