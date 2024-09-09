# Import Packages
import os, sys
import datetime as dt
import pandas as pd

global PATH, DATA_PATH, RAW_PATH, SAMPLE
EXTENSION = '.csv'
TIME_STEP = 15 # Time delta between samples, replace with sys.argv later
TIME_BUFFER = 0.125 # Time delta buffer to account for time build up due to sample read time
DATA_PATH = "Datasets"
RAW_PATH = "Testing_Data"
SAMPLES = ["Sample4"] # Replace with sys.arv later or ls
DT_FMT = '%Y%m%d'
TM_FMT = "%H%M%S"
# AUG_SPLIT = 1 #Implementing later with sys.argv
# RAW_PATH = "Raw_Data/"

# Add system.argv arguments later for streamlining pipeline

def getPath():
    # Find the directory path of the program folder
    global PATH
    PATH = os.path.dirname(os.path.abspath(__file__))
    # print("Current Path:", PATH)
    return


def getFileList(sample):
    # global RAW_PATH
    # Create a list of all files within the current folder
    files = []
    for x in os.listdir("/".join([PATH,RAW_PATH,sample])):
        # Add file name to list
        if x.endswith("i.csv"):
            continue
        elif x.endswith("r.csv"):
            continue
        elif x.endswith("a.csv"):
            continue
        elif x.endswith(".csv"):
            files.append(x)

    return files


def returnSample(df, filter):
    new_df = df.groupby("Sample").get_group(filter)
    return new_df


def sepSamples(fileList):
    infoList, sampleList = [], []
    for x in fileList:
        # Split the data from the extension, and then by the underscores
        info = x.split('.')
        info = info[0].split('_')
        if len(info) < 6:
            info.append("g")
            
        # print(info)
        infoList.append(info)

    # Make dataframe from the list of sample info
    infoDF = pd.DataFrame(infoList, columns=["Sample", "Protocol", "Spectra Number", "Date", "Time", "Type"])
    sampleOptions = infoDF['Sample'].unique()

    return infoDF, sampleOptions


def readSpectraFile(fileName, sample):
    fullFileName = fileName + EXTENSION
    fullPath = "/".join([PATH,RAW_PATH,sample,fullFileName])

    # Read whole csv of the spectra based on file path and put into dataframe
    df = pd.read_csv(fullPath, names=["Wavelength", "Absorbance", "Reference Signal", "Sample Signal"], on_bad_lines='skip', float_precision='high')
    
    # Drop extra metadata rows and fix index
    df.dropna(axis=0, how='any',inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(axis=0, index=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def iterateSpectra(df, maxTime, sample_num):
    # Iterate through the datafiles to read them based on time delta
    # Add FOR loop for augmenting data based on the "Spectra Number".
    currentTime = 0
    currentColumn = 1
    currentDelta = dt.timedelta(hours=currentTime)
    gradient_pos = 0
    df = df.dropna(axis=0, how='any')
    # Sort dataframe based on date and time
    sorted = df.sort_values(by=["Date", "Time"])
    sorted = sorted.reset_index(drop=True, inplace=False)
    # Get date of the first file using datetime (in case it goes over multiple days)
    start_date = dt.datetime.strptime(sorted.at[0, 'Date'], DT_FMT).date()
    start_time = dt.datetime.strptime(sorted.at[0, 'Time'], TM_FMT).time()
    # Combine date and time into datetime
    startInfo = dt.datetime(start_date.year, start_date.month, start_date.day, start_time.hour, start_time.minute, start_time.second)
    for row, sample in sorted.iterrows():
        # Get date and time of the current file
        cur_date = dt.datetime.strptime(sorted.at[row, 'Date'], DT_FMT).date()
        cur_time = dt.datetime.strptime(sorted.at[row, 'Time'], TM_FMT).time()
        current = dt.datetime(cur_date.year, cur_date.month, cur_date.day, cur_time.hour, cur_time.minute, cur_time.second)
        # Get the time delta between start and current file
        delta = current - startInfo
        delta = delta - dt.timedelta(minutes=(delta.seconds/60) % TIME_STEP)
        # Only deal with the file if the difference between deltas is >= the time buffer
        if delta >= currentDelta: # or currentTime == 0:
            fileName = sorted.at[row, 'Sample'] + '_' + sorted.at[row, 'Protocol'] + '_' + sorted.at[row, 'Spectra Number'] + '_' + sorted.at[row, 'Date'] + '_' + sorted.at[row, 'Time']
            if sorted.at[row, 'Type'] == 'a':
                fileName = fileName + '_' + sorted.at[row, 'Type']
            # print(fileName)

            # Make temp dataframe from the spectra using function
            temp_df = readSpectraFile(fileName, sample_num)
            # If the first file, create a wavelength column
            if delta == dt.timedelta(minutes=0):
                wavelengths = temp_df["Wavelength"]
                spectra_df = pd.DataFrame(wavelengths)

            time = delta.total_seconds()/60/60
            # Get only the absorbance column from the spectral data
            absorbances = temp_df["Absorbance"]
            if absorbances.iloc[0].endswith("inf") is False:
                # Insert the absorbance column at each time point
                spectra_df.insert(currentColumn, time, absorbances)
                currentColumn += 1

                # Add to delta time based on time step and buffer
                currentTime += time # TIME_STEP
                currentDelta = delta + dt.timedelta(minutes=TIME_STEP) - dt.timedelta(minutes=TIME_BUFFER)

            # filePath = PATH + "\\" + sorted.at[0, 'Sample'] + EXTENSION
            filePath = "/".join([PATH,DATA_PATH,sorted.at[0, 'Sample']+EXTENSION])
            spectra_df.to_csv(path_or_buf=filePath, index=False, mode='w')


    return


def runReader():
    getPath()
    # Iterate though list of samples
    for sample_num in SAMPLES:
        fileList = tuple(getFileList(sample_num))
        info_df, sampleOptions = sepSamples(fileList)
        # print(fileList)

        selectedSamples = sampleOptions
        maxTime = 48

        for sample in selectedSamples:
            # print(sample)
            df = returnSample(info_df, sample)
            print(df.head())
            iterateSpectra(df, maxTime, sample_num)
        # print(smallerDF)


runReader()