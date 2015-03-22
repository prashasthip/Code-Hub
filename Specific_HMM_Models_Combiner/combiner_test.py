import json
import os
from os import listdir
from os.path import isfile, join
from model_combiner import *

models_dir = r'./Individual Models'

def get_models_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".txt" in f))]
    return onlyfiles

def read_models_list(filelist):
	models_list = []
	for f in filelist:
		models_list.append(json.loads(open(f).read()))
	return models_list

models_file = get_models_list(models_dir)
models_list = read_models_list(models_file)

combine(models_list,"recognizer_initial.txt",0.75,0.25)