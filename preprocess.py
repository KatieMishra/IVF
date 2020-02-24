import os 
import numpy as np
import pandas as pd 
from skimage import data, io, filters
import PIL.Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import cv2

def readImages(dir_path):
    """ 
    Reads in images from a given file path.
    
    Argument:
    dir_path -- absolute directory path given as a string
    
    Returns:
    images -- nparray of the images of shape (m, height, width, channel)
    names -- list of the image names
    """
    images = []
    names = []
    for file in os.listdir(dir_path):
        img = io.imread(dir_path + "/" + file)
        images.append(img)
        names.append(file)
    return np.array(images), names


def processName(names):
    """
    Extracts the file name into the id number, NEGATIVE/DELIVERED/
    BIOCHEMICAL/SAB, and isDuplicate columns
    
    Argument:
    names -- list of file names
    
    Returns:
    pandas dataframe of file id, attribute, and isDuplicate
    """
    #Processes into dataframe 
    np_names = np.asarray(names).reshape(-1, 1)
    np_names = np.char.split(np_names, sep =' ') 
    df1 = pd.DataFrame(np_names, columns=['data'])
    df1[['id','PhotoNameType', "isDuplicate"]] = pd.DataFrame(df1.data.values.tolist(), index= df1.index)
    df1.drop('data', 1, inplace = True)
    df1['id'] = df1['id'].apply(lambda x: int(x))
    #Adds a column to signify whether there are multiple embryos of the photo (1)
    #or only one (0)
    df1['isDuplicate'] = df1['id'].duplicated(keep=False).astype(int).astype(str)
    
    #Cleans up PhotoNameType Column
    df1['PhotoNameType'] = df1['PhotoNameType'].apply(lambda x: x.split('.')[0])
    
    #return df1
    return df1

def processXLSXData(data):
    """
    Processes the CSV data 
    
    Argument: CSV file
    
    Returns: pandas dataframe 
    """
    df = pd.read_excel(data)
    df.rename(columns={'SART NUM': 'id'}, inplace = True)
    return df

def mergeData(df1, df2):
    """
    Merges the names dataframe and excel sheet by id number. This will retain the ordering 
    with the images file.
    
    Argument: df1 -- names dataframe df2 -- excel dataframe
    
    Returns: merge pandas dataframe
             y -- a nparray encoded as 0 if there was no pregnancy, 1 if there was a pregnancy
                  no live birth, and 2 if there was a pregnancy and live birth
    """
    df = pd.merge(df1, df2, how="left", on = "id")
    
    #Asserts that the every name id has an associated key
    assert(df1.shape[0] == df.shape[0])
    
    #Adds y to the dataframe 
    df['y'] = np.where(df['PREG']==False,0,1)
    df['y'] += np.where(df['PhotoNameType']=="DELIVERED",1,0)
    
    
    return df, df['y'].values

def resize(photos, width):
    """
    Resizes images to be a square with width width
    
    Arguments: photo -- photo as a np array of shape (width, height, 3)
               width -- int to specify width and hight
               
    Returns: resized np array
    """
    rescaled = []
    for i in range(photos.shape[0]):
        photo = photos[i,:,:,:]
        initHeight, initWidth = photo.shape[0], photo.shape[1]
        res = cv2.resize(photo, dsize=(width, width), interpolation = cv2.INTER_AREA)
        rescaled.append(res)
        #Prints photo
        #PIL.Image.fromarray(img)
    return np.array(rescaled)

all_images, names = readImages('/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/part_1/')
all_images = resize(all_images, 224)
namesDf = processName(names)
sheetDf = processXLSXData("/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/2018 FRESH TRANSFERS NAMES REMOVED.xlsx")
fullDf, y = mergeData(namesDf, sheetDf)