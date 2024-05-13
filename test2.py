import os

path=f"datasets/usc_5/labels/train"

with open(f"{path}/10.txt", mode='r') as f:
    line=f.readline()
    print(line)
    print(type(line))
