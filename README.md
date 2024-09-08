# Online NIR Cell Culture Analysis Using 1D-CNN

## Project Information
### Full Title:
Non-intrusive Testing of Liquid Culture Medium using Online NIR Spectroscopy and Machine Learning for Qualitative Analysis

### Contributers:
Benjamin Samuel\
Connor Reintjes\
Paola Gonzalez Perez\
Shiza Hassan

### Supervisor:
Dr. Amin R. Rajabzadeh

### Association:
McMaster Univeristy - W Booth School of Engineering Practice and Technology\
Capstone Project for 4TR1 & 4TR3

## Purpose
Implement  NIR spectroscopy and machine learning for the detection and real-time analysis of media within cultures for continuous quality testing

## Software Setup
### Python Environment:
#### Python Installation:
The python installation version can be verified using:
```bash
python3 --version
```
Python can be updated or be installed if required using:
```bash
sudo apt-get update
sudo apt-get install python3.10
```
Pip and Venv can then be installed using the following commands:
```bash
sudo apt install python3-pip
sudo apt install python3.10-venv
```
#### Create Virtual Environment:
The python environment is not included within the repository. A virtual environment can be created within the Linux terminal with:
```bash
python3 -m venv /path/to/virtual/environment
```
Here is an example when in the current directory:
```bash
python3 -m venv .venv
```
The python virtual environment and `.venv` folder has been created within the target directory. 
#### Activate/Deactivate Environment:
The virtual environmnent can be activated in the Linux terminal using:
```bash
source .venv/bin/activate
```
The when the virtual environment is activated, the filepath can be checked using:
```bash
which python 
```
To leave/deactivate the virtual environment, use this command within the terminal:
```bash
deactivate
```
#### Package Requirements
The list of all required python packages are within the `requirements.txt` file. Before downloading packages, ensure that the virtual environment is active, or the packages will be installed to the normal python environment. The packages can be downloaded using:
```bash
pip install -r requirements.txt
```
If contributing to the software and additional packages are used, the following command can be used to `freeze` all dependancies to the `requirements.txt` file.
```bash
pip freeze > requirements.txt
```
