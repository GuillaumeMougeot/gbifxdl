# gbifxdl

gbif-noctu, gbif-ctfb : get taxonID from GBIF using species name.

gbif-get, gbif-dl : download images from GBIF using taxonID.

gbif-post, gbif-bulk: use of the asynchronous downloader of GBIF for query more than 100,000 records. Careful, this creates DOIs, do not overuse it.

## Usage

```
python gbif-bulk.py user=usr:pwd config=gbif_cfg.yaml
```

