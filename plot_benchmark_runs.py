#!/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

path = sys.argv[1]

if not os.path.isdir(path):
    print(f'Directory not found: {path}')
    sys.exit(-1)

ax = None

for file in os.listdir(path):
    if file.endswith('.csv'):
        df = pd.read_csv(path + file, names=['angle', file])
        ax = df.plot(x='angle', ax=ax, xlabel='Angle (radians)', ylabel='Time (seconds)')
    
plt.show()