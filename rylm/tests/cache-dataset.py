# this is just a test file that will write out some data to the cache
# to see if caching is working

import pathlib
# first see if the data.txt exists
import rylm.tests as tests

# get the path to the tests directory
from rylm.utils import get_path_string
root_path = get_path_string(tests)

dir_path = pathlib.Path(f"{root_path}/dataset_cache")
data_file_path = pathlib.Path(dir_path / "data.txt")
if not data_file_path.exists():
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(data_file_path, "w") as f:
        f.write("This is a test file for caching.\n")
    print(f"Created cache file at {data_file_path}")
else:
    with open(data_file_path, "r") as f:
        content = f.read()
    print(f"Cache file exists. Content:\n{content}")