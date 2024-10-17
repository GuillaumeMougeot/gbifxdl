import pandas as pd
import requests
import json
from tqdm import tqdm
from multiprocessing import Pool
import os
from functools import partial

def get(request, keys):
    get_template = "https://api.gbif.org/v1/species/match?name={}".format(request)
    species_dict = json.loads(requests.get(get_template).text)
    out = {}
    for k in keys:
        out[k] = species_dict[k] if k in species_dict.keys() else "UNKNOWN"
    return out

if __name__ == '__main__':
    path = "data/classif/XPRIZE_Brazil_Insects.csv"
    out_path = "data/classif/XPRIZE_Brazil_Insects-out.csv"
    df = pd.read_csv(open(path,'rb'), delimiter=";", encoding='latin-1')

    species = list(df["Species"])
    authors = list(df["Author"])
    names = [s + " " + a for s, a in zip(species, authors)]

    gbif_keys = ["usageKey", "scientificName", "rank", "status"]

    get_ = partial(get, keys=gbif_keys)

    with Pool(12) as p:
        outputs = list(tqdm(p.imap(get_, names), total=len(df)))
    
    for k in gbif_keys:
        df["GBIF"+k] = [element[k] for element in outputs]
    df.to_csv(out_path, index=False)