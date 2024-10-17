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
    root_path = "data/classif"
    in_path = os.path.join(root_path, "taxon.txt")
    csv_path = os.path.join(root_path, "taxon-out.csv")
    excel_path = os.path.join(root_path, "taxon-out.xlsx")

    gbif_keys = ["usageKey", "scientificName", "rank", "status"]

    df = pd.read_csv(open(in_path,'rb'), delimiter="\t", encoding='latin-1')
    df = df[(df.phylum == "Arthropoda") & (df.taxonRank == "ESPECIE")]
    # df = df[(df.phylum == "Arthropoda") & (df.taxonRank == "ESPECIE")].head()
    names = list(df[(df.phylum == "Arthropoda") & (df.taxonRank == "ESPECIE")].scientificName)

    get_ = partial(get, keys=gbif_keys)

    with Pool(12) as p:
        outputs = list(tqdm(p.imap(get_, names), total=len(names)))

    for k in gbif_keys:
        df["GBIF"+k] = [element[k] for element in outputs]
    df.to_csv(csv_path, index=False)
    # df.to_excel(excel_path, index=False)
