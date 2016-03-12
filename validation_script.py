# Script to Run the validation for Localization Algorithm
# Takes Input a Data file(.pkl) which containts the dataset
# Data set input needed is a pkl file with row format:
# class_wnid,imgid,class_idx,xmlidx,url,xmin,xmax,ymin,ymax = ret
# It uses a VGG16 model and writes the results in the same folder as input with _results.csv as suffix to input file name
# hyper Parameters used by algorithm are currently hardcoded in the script

import types
from library.localization import *
from deconv_utils import *
from library.image_utils import *
import csv
import sys
import datetime
import time
import os

#############################################################
#Image Processing Inputs
Param.num_dilation = 15
#############################################################

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def print_and_log(text, filename):
    print (get_time_stamp()+text)
    filename.write(get_time_stamp()+ " "+text+"\n")

#HyperParams
kmax_input = 30
n_neurons_input = 5

#Validation & Model File Inputs
data_file = sys.argv[1]
path = os.path.dirname(data_file)
name = os.path.basename(data_file)
data_extn = ".pkl"
results_path = path + "/results_k"+str(kmax_input)+"_n"+str(n_neurons_input)+"/"
results_file = results_path+ name + "_results.csv"
log_file = results_path+ name + "_log.txt"
model_file = 'Data/vgg16_weights.h5'
class_file = 'Data/CLASSES.pkl'

start_index = 0
result_csv = None
try:
    #Check if csv file exists and read the last index outputted
    results_csv = open(results_file,'r')
    lastline = results_csv.read()
    if lastline!="":
        lastline = lastline.split('\n')[-2]
        start_index = int(lastline.split(',')[0]) + 1
except IOError as e:
    pass
finally:
    if type(result_csv)!=types.NoneType:
        close(results_csv)

try:
    #Write to CSV file
    results_csv = open(results_file,'a')
    log_txt = open(log_file,'a')
    msg = ("#################################################################################################\n")
    print_and_log(msg, log_txt)

    #Load the VGG Model and ImageNet Class Mappings
    from library.classifiers.pretrained_vgg16 import PretrainedVGG
    model = PretrainedVGG(h5_file = model_file)
    print_and_log("Loaded Model", log_txt)

    CLASSES = pickle.load(open(class_file))
    print_and_log("Loaded Class File",log_txt)

    #Read Input File for the candidates
    candidates = pickle.load(open(data_file+data_extn))
    print_and_log("Loaded Input Data Set",log_txt)

    #Read Input File and start at the index higher than that of last line

    for index in range(start_index,len(candidates)):
        log_txt.flush()
        ret = candidates[index]
        class_wnid,imgid,class_idx,xmlidx,url,xmin,xmax,ymin,ymax = ret
        im = image_from_url(url)
        if type(im) == types.NoneType:
            #Debug Print
            msg = "[%d]: Skip (%d,%d): %s : URL is bad."%(index, class_idx,xmlidx, url)
            print_and_log(msg,log_txt)
        elif np.mean(im)>=253:
            #Debug Print
            msg = "[%d]: Skip (%d,%d): %s : URL is Empty(White) Image"%(index, class_idx,xmlidx,url)
            print_and_log(msg,log_txt)
        else:  
            bbox_coords = bbox(im,model,class_no=class_idx,n_neurons = n_neurons_input,kmax=kmax_input)
            xmin_out, xmax_out, ymin_out, ymax_out = bbox_coords
            precision = eval_precision(bbox_coords, (xmin,xmax,ymin,ymax))
            #Write Results to file
            result_row = "%d, %s, %d, %d, %f,"%(index, imgid, class_idx, xmlidx, precision)
            result_row += "%d, %d, %d, %d,"%(xmin, xmax, ymin, ymax,)
            result_row += "%d, %d, %d, %d"%(xmin_out, xmax_out, ymin_out, ymax_out)
            result_row += "\n"
            results_csv.write(result_row)
            results_csv.flush()
            #Debug Print
            msg = "[%d]: %d,%d,%f"%(index, class_idx,xmlidx,precision)
            print_and_log(msg,log_txt)
except IOError as e:
    print_and_log("File Reading Issue - Hopefully close and flushed\n")
finally:
    close(results_csv)
    close(log_txt)