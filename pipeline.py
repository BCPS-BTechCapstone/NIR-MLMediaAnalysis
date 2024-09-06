# Import Packages
import os, sys
import datetime as dt
import pandas as pd

global PATH, DATA_PATH, RAW_PATH
EXTENSION = '.csv'
TIME_STEP = 15
TIME_BUFFER = 1
DATA_PATH = "Datasets"
RAW_PATH = "Testing_Data"
# RAW_PATH = "Raw_Data/"


def getPath():
    global PATH
    PATH = os.path.dirname(os.path.abspath(__file__))
    print("Current Path:", PATH)
    return


def getFileList():
    # global RAW_PATH
    files = []
    for x in os.listdir("/".join([PATH,RAW_PATH])):
        if x.endswith("i.csv"):
            continue
        elif x.endswith("r.csv"):
            continue
        elif x.endswith("a.csv"):
            files.append(x)
        elif x.endswith(".csv"):
            files.append(x)

    return files


getPath()
getFileList()