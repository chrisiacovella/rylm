# this is just a test file that will write out some data to the cache
# to see if caching is working

import pathlib
import os
# first see if the data.txt exists

# get the path to the tests directory

dir_path = pathlib.Path(f"rylm/tests/dataset_cache")
data_file_path = pathlib.Path(dir_path / "data.txt")
#print the full path
print(os.path.abspath(str(data_file_path)))
if not data_file_path.exists():
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(data_file_path, "w") as f:
        f.write("This is a test file for caching.\n")
    print(f"Created cache file at {data_file_path}")
else:
    with open(data_file_path, "r") as f:
        content = f.read()
    print(f"Cache file exists. Content:\n{content}")

####