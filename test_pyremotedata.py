import time, os, uuid
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
import pandas as pd

# error_url = "http://cettia-idf.fr/ajax/dldimg/img_user!2017!12!2441!1512399868_2441.jpg"
# work_url = "https://observation.org/photos/15658540.jpg"

occurrences = "/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546.parquet"
df = pd.read_parquet(occurrences)
url_list = df.identifier.tolist()[:500]
oldnames = [url.split("/")[-1] for url in url_list]
newnames = [f'{uuid.uuid4()}{os.path.splitext(basename)[1]}' for basename in oldnames]

with IOHandler(clean=True) as io:
    tmp_dir = io.lpwd()
    io.cd("datasets")
    io.cd("test")
    # try:
    #     io.execute_command("rm "+ error_url.split("/")[-1])
    # except:
    #     pass
    # try:
    #     io.execute_command("rm "+ work_url.split("/")[-1])
    # except:
    #     pass
    # time.sleep(0.5)
    # try:
    #     io.ls()
    # except:
    #     pass
    # io.put(work_url)
    # print(io.ls())
    # try:
    #     io.execute_command("mput -P 5 {}".format(" ".join([error_url, work_url])))
    # except Exception as e:
    #     print(str(e))
    # io.ls()
    # print("ABC")
    # print(io.ls())
    N = len(url_list)
    i = 0
    while i < N:
        print(i)
        batch_nnames = []
        batch_onames = []
        batch_urls = []
        while i < N and oldnames[i] not in batch_onames and len(batch_onames) < 64:
            batch_nnames.append(newnames[i])
            batch_onames.append(oldnames[i])
            batch_urls.append(url_list[i])
            i += 1
        print(i, len(batch_urls))
        # io.execute_command("mput -P 16 {}".format(" ".join(batch_urls)))
        for oname, nname, url in zip(batch_onames, batch_nnames, batch_urls):
            io.put(url, nname)
    io.cache["file_index"] = newnames
    print(f"{tmp_dir=}")
    for local_file, remote_file in RemotePathIterator(io):
        print(f'{remote_file}={os.path.getsize(local_file)} ({len(os.listdir(tmp_dir))})')
print("-----")
try:
    print(os.listdir(tmp_dir))
except:
    pass
print("DEF")


