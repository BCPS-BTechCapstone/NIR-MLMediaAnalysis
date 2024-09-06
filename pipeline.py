#! ~/Github/NIR-Capstone-1D-CNN/.venv/bin/python
# Import Packages
import os, sys
import datetime as dt
import pandas as pd

global PATH, DATA_PATA
RAW_PATH = ""
EXTENSION = '.csv'
TIME_STEP = 15
TIME_BUFFER = 1

def getSystemArgs():
    print(sys.argv)

    
def getPath():
    global PATH, RAW_PATH
    cwd = os.getcwd()
    PATH = os.path.dirname(os.path.abspath(''))
    # print("Current working directory:", cwd)

    if cwd.endswith("Data") == False:
        RAW_PATH = cwd + "\Data"
    else:
        RAW_PATH = cwd

    os.chdir(RAW_PATH)
    #print(PATH, RAW_PATH)
    return


def getFileList():
    files = []
    for x in os.listdir():
        if x.endswith("i.csv"):
            continue
        elif x.endswith("r.csv"):
            continue
        elif x.endswith("a.csv"):
            files.append(x)
        elif x.endswith(".csv"):
            files.append(x)

    return files

