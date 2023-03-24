import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
import asyncio
from telegram_bot import NotificationBot

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root_dataset_folder", type=str, help="Path to the root folder to work on."
    )

    parser.add_argument("--destination_folder", type=str, default=Path("data/digests"))

    parser.add_argument("--glob", type=str, help="Glob path to parse", default=None)

    parser.add_argument(
        "--n_jobs", default=1, type=int, help="Number of processes to be used. "
    )

    args = parser.parse_args()
    return args


def md5_for_file(f, block_size=2**20):
    md5 = hashlib.md5()
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()


def hash_dataset(idx, table_path, block_size=2**20):
    try:
        with open(table_path, "rb") as fp:
            digest = md5_for_file(fp, block_size)
            return (digest, table_path)
    except Exception:
        # Avoid stopping the conversion procedure, count the failures.
        return (table_path, None)


def hashing_datasets(dataset_list, n_jobs=1):
    n_jobs = 1

    r = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(hash_dataset)(idx, table_path)
        for idx, table_path in tqdm(
            enumerate(dataset_list), position=0, leave=False, total=len(dataset_list)
        )
    )

    return r


def tally_digests(digests):
    tally_dict = {}
    count_digests = Counter([x[0] for x in digests])
    for pair in digests:
        digest, path = pair
        if digest not in tally_dict:
            tally_dict[digest] = [path]
        else:
            tally_dict[digest].append(path)
    return tally_dict, count_digests


async def hash_directory(args):
    bot = NotificationBot()
    
    await bot.send_message("🔴 - Starting hashing run")
    
    stem = Path(args.root_dataset_folder).stem
    data_path = Path(args.root_dataset_folder)
    assert data_path.exists()

    print("Glob")
    if args.glob is None:
        all_datasets = glob.glob(f"{data_path}/**/*_candidates/**/learningData.csv", recursive=True)
    else:
        all_datasets = glob.glob(args.glob, recursive=True)
    print("Finished compilation of paths.")

    n_jobs = args.n_jobs
    digests = hashing_datasets(all_datasets, n_jobs)

    await bot.send_message("🏁 - Hashing complete")

    tally_dict, count_digests = tally_digests(digests)

    duplicates = {x: tally_dict[x] for x in count_digests if count_digests[x] > 1}
    
    pth_tally = Path(args.destination_folder, stem)
    os.makedirs(pth_tally, exist_ok=True)
    json.dump(tally_dict, open(Path(pth_tally, f"{stem}_tally.json"), "w"), indent=2)
    json.dump(
        duplicates, open(Path(pth_tally, f"{stem}_duplicates.json"), "w"), indent=2
    )

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(hash_directory(args))
