# Prerequisites:
**Python _3.10.2_**
> pip3 install -r requirements.txt
# How to run
## Method 1) Jupyter notebook
### 1) Install Jupyter notebook
> pip3 install notebook
### 2) Update the path to the images
### 3) Run the notebook 

## Method 2) main.py file
### 1) Simply update the path to the images in the list we create at the begining of the main.py file and run the code.

# Difference between both

1) The notebook may have some redundant code and just a bit more messy (Most of the code has been refactored though) but they are easy to see the flow of the code. The main.py file and the source code files in the **src** directory are written to remove redundancy as best as possible, with **PEP8** guidelines and in a more **modular** fashion. They are easy to **test** as well and may require only a few modification when the code is being productionized.
