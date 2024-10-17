import pandas as pd
from pygbif import occurrences as occ
import requests
import hashlib
import urllib.request
import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor

def download_species(speciesKey, rootdir = "."):
    resp = occ.search(speciesKey = speciesKey)
    basedir = os.path.join(rootdir, str(speciesKey))

    for metadata in resp.get("results", []):
        medias = metadata.get("media", None)
        if medias:
            os.makedirs(basedir, exist_ok=True)
            for media in medias:
                # check if the identifier (url) is present
                url = media.get("identifier", None)
                if url:
                    with urllib.request.urlopen(url) as response:
                        if response.getcode() != 200:
                            print("Invalid url", url, response.getcode())
                            continue
                        info = response.info()
                        
                        img_type = info.get_content_type().lower()
                        if img_type not in ('image/png', 'image/jpeg', 'image/gif', 'image/jpg', 'image/tiff', 'image/tif'):
                            print("Invalid image type", img_type, url)
                            continue
                    
                    ext = "." + img_type.split("/")[1]
                        
                    response = requests.get(url, stream=True)
                    if not response.ok:
                        print(response)
                        continue
                        
                    basename = hashlib.sha1(url.encode("utf-8")).hexdigest()
                    img_name = basename + ext
                    abs_img_name = os.path.join(basedir, img_name)
                    with open(abs_img_name, 'wb') as handle:
                        for block in response.iter_content(1024):
                            if not block:
                                break
                    
                            handle.write(block)
                    # with open('image_name.jpg', 'wb') as handler:
                    #     handler.write(img_data)
    if os.path.exists(basedir) and len(os.listdir(basedir))==0:
        os.removedirs(basedir)
    return speciesKey

if __name__ == '__main__':
    # csv_path = "data/classif/taxon-out.csv"
    # out_path = "data/classif/ctfb-2"
    csv_path = "data/classif/noctu/XPRIZE_Brazil_Insects-out.csv"
    out_path = "/home/george/codes/gbif-request/data/classif/noctu/images"

    df = pd.read_csv(csv_path)

    all_species_keys = list(df[(df.GBIFrank == "SPECIES") & (df.GBIFstatus == "ACCEPTED")]['GBIFusageKey'])
    idx = np.arange(len(all_species_keys))

    # for k in all_species_keys:
    #     download_species(k, rootdir=out_path)

    download_species_ = partial(download_species, rootdir=out_path)
    # def download_species_(x):
    #     speciesKey, idx = x
    #     return download_species(speciesKey=speciesKey, idx=idx, rootdir=out_path, total=len(all_species_keys))
    # download_species_ = lambda x: download_species(x[0], idx=x[1], rootdir=out_path, total=len(all_species_keys))

    # r = process_map(download_species_, all_species_keys, max_workers=40, chunksize=10000)

    with Pool(40) as p:
        p.map(download_species_, all_species_keys)

    # with ThreadPoolExecutor(100) as exe:
    #     tqdm(exe.map(download_species_, all_species_keys), total=len(all_species_keys))
        # _ = [exe.submit(download_species_, k) for k in all_species_keys]

    # with Pool(40) as p:
    #     with tqdm(total=len(all_species_keys)) as pbar:
    #         for _ in p.imap_unordered(download_species_, range(0, len(all_species_keys))):
    #             pbar.update()

    # with Pool(1) as p:
    #     tqdm(p.imap(download_species_, range(0, len(all_species_keys))), total=len(all_species_keys))
