# this file will do a simple 90/10 split of the regular and rolled inputs
# note: we will probably do a dynamic split in the CNN - TBD


from filesplit.split import Split
import math
import os

# Clean up old files, if they exist
if os.path.isfile("./cleansed_data/manifest"):
    os.remove("./cleansed_data/manifest")
if os.path.isfile("./cleansed_data/regular_input.test"):
    os.remove("./cleansed_data/regular_input.test")
if os.path.isfile("./cleansed_data/regular_input.train"):
    os.remove("./cleansed_data/regular_input.train")
if os.path.isfile("./cleansed_data/rolled_input.train"):
    os.remove("./cleansed_data/rolled_input.train")
if os.path.isfile("./cleansed_data/rolled_input.test"):
    os.remove("./cleansed_data/rolled_input.test")

with open("./cleansed_data/regular_input.txt", "r") as f:
    length_of_file = len(f.readlines())

lines_to_split = math.floor(length_of_file * .9)

# create split instance
split = Split("./cleansed_data/regular_input.txt", "./cleansed_data")

# split file
split.bylinecount(lines_to_split)

# rename the files to TRAIN / TEST
os.rename("./cleansed_data/regular_input_1.txt", "./cleansed_data/regular_input.train")
os.rename("./cleansed_data/regular_input_2.txt", "./cleansed_data/regular_input.test")

# ROLLED CODES NEXT
split = Split("./cleansed_data/rolled_input.txt", "./cleansed_data")

# split file
split.bylinecount(lines_to_split)

# rename the files to TRAIN / TEST
os.rename("./cleansed_data/rolled_input_1.txt", "./cleansed_data/rolled_input.train")
os.rename("./cleansed_data/rolled_input_2.txt", "./cleansed_data/rolled_input.test")
