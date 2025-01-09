from pathlib import Path
import pyarrow.parquet as pq
import dask.dataframe as dd
import time

# local postprocessing

def list_duplicates(parquet_path):

    df = dd.read_parquet(parquet_path)
    duplicates = df.groupby('img_hash').size().compute()
    duplicate_hashes = duplicates[duplicates > 1].index

    return duplicate_hashes

if __name__=='__main__':
    dup=list_duplicates(
        parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata.parquet"
    )
    print(dup)
