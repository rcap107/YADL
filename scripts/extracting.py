'''This script iterats over the tar archive to measure the number and size of 
files by extension. 

This takes a very long time, so the script itself isn't very useful. 
'''
import tarfile
import os.path as osp
import os
from tqdm import tqdm
from collections import Counter
import pickle

ARCHIVE_PATH = osp.expanduser("~/store/d3m/datasets-master.tar.bz2")

tfile = tarfile.open(ARCHIVE_PATH, mode="r:bz2")

# names = tfile.getnames()
# members = tfile.getmembers()

# pickle.dump(members, open(osp.expanduser("~/store/d3m/list_members.pkl"), "wb"))
# pickle.dump(names, open(osp.expanduser("~/store/d3m/list_names.pkl"), "wb"))
sizes_by_ext = {}
count_by_ext = {}
count = 0
next_member = tfile.next() 

tgt_ext = [".txt", ".csv"]
progress = tqdm()
# while next_member is not None:
while next_member is not None:
    if next_member.isreg():
        name, ext = osp.splitext(next_member.name.strip())
        if ext in sizes_by_ext:
            sizes_by_ext[ext] += next_member.size
        else:
            sizes_by_ext[ext] = next_member.size
            
        if ext in count_by_ext:
            count_by_ext[ext] += 1
        else:
            count_by_ext[ext] = 1

        # if ext in tgt_ext:
        #     tfile.extract(next_member.name, osp.expanduser("~/store/d3m/extracted_files"))

    next_member = tfile.next()
    count += 1
    progress.update()
    
pickle.dump(sizes_by_ext, open(osp.expanduser("~/store/d3m/sizes_by_ext.pkl"), "wb"))
pickle.dump(count_by_ext, open(osp.expanduser("~/store/d3m/count_by_ext.pkl"), "wb"))
