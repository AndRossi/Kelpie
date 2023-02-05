import os
import pandas as pd
from collections import defaultdict

# file = './MOF-3000/ConvE-necessary/necessary.txt'
file = './MOF-3000/ConvE-sufficient/sufficient.log'

with open(file, 'r') as f:
    lines = f.read().split('\n')

nple_count = defaultdict(int)
count = 0

for i in range(0, len(lines)):
    if len(lines[i].split(';')) == 3:
        count += 1
        fact = lines[i]
        ix = i+1 if len(lines[i+1].split(';')) > 1 else i+2
        explains = lines[ix].split(',')
        for explain in explains:
            explain = explain.split(':')[0]
            n = len(explain.split(';')) // 3
            nple_count[n] += 1

print('total count:', count)
print(nple_count)