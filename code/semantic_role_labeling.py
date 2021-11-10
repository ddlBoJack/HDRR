import os
import argparse
import json
import pickle
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor

with open("/home/share/maziyang/HDRR/activitynet/val.pkl", 'rb') as f:
  data = pickle.load(f)

print("The num of keys:")
print(len(data.keys()))

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")

for key in tqdm(data.keys()):
    data[key]['srl']={}
    for idx,token in enumerate(data[key]['tokens']):
        data[key]['srl'][idx]=[]
        predict = predictor.predict_tokenized(token)

        for item in predict['verbs']:
          data[key]['srl'][idx].append(item['tags'])

with open("/home/share/maziyang/HDRR/activitynet/val_semantic.pkl", 'wb') as ff:
  pickle.dump(data, ff)