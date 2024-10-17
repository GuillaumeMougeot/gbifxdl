import gbif_dl
import pandas as pd
from pygbif import occurrences as occ

if __name__ == '__main__':
    csv_path = "data/classif/taxon-out.csv"
    out_path = "data/classif/ctfb"

    df = pd.read_csv(csv_path)

    all_species_keys = list(df[(df.GBIFrank == "SPECIES") & (df.GBIFstatus == "ACCEPTED")]['GBIFusageKey'])

    # print(occ.search(speciesKey = 1374468))

    # Cut the species in chunks to avoid MaxRetryError: HTTPConnectionPool
    chunk_size = 10
    assert chunk_size < len(all_species_keys)
    for i in range(0,len(all_species_keys)-chunk_size, chunk_size):
        print("[{}/{}] Chunk.".format(i//chunk_size, len(all_species_keys)//chunk_size))
        queries = {
            "speciesKey": all_species_keys[i:i+chunk_size],}

        data_generator = gbif_dl.api.generate_urls(
            queries=queries,
            label="speciesKey",
            one_media_per_occurrence = False,
            nb_samples = chunk_size * 10
        )

        stats = gbif_dl.stores.dl_async.download(data_generator, root=out_path)