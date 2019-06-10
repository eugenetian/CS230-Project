import csv
import os
import shutil
import random
import numpy as np

with open('Data_Entry_2017.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    dct = {}
    for row in csv_reader:
        name = row[0]
        status = row[1] 
        if name in dct:
            print("There is a repeat")
            print(name)
            break
        else:
            #print(name)
            if status == "No Finding":
                dct[name] = "No Finding"
            else:
                dct[name] = "Abnormality"

#---------------------------------------------------
    #items = os.listdir("./data/png")
    #path = "./splitdata/"
    #for name in items:
    #    if dct[name] == "No Finding":
    #       shutil.move("./data/png/" + name, path + "nofinding") 
    #    else:
    #        shutil.move("./data/png/" + name, path + "abnormality")
    #print(items)
    #path = "./splitdata/nofinding/"
    #items = os.listdir(path)
    #for name in items:
    #    shutil.move(path + name, "./data/png/")
    
    #keys = dct.keys()

    keys = []
    for name in os.listdir("./data/"):
        keys.append(name)
    print(len(keys))
        
    train = []
    val = []
    test = []

    train = np.random.choice(keys, 36000, replace = False)
    keys = list(set(keys) - set(train))
    
    val = np.random.choice(keys, 2000, replace = False)
    keys = list(set(keys) - set(val))

    test = np.random.choice(keys, 2000, replace = False)
    keys = list(set(keys) - set(test))

    def movethings(lst, foldername):
        path = "./newdata/" + foldername
        for name in lst:
            if dct[name] == "No Finding":
                shutil.move("./data/" + name, path + "nofinding")
            else:
                shutil.move("./data/" + name, path + "abnormality")

    movethings(train, "train/")
    movethings(val, "val/")
    movethings(test, "test/")



    #print(len(test))
    #print(len(val))
    #print(len(train))

    #testTrain = set(test).union(set(train))
    #allofthem = testTrain.union(set(val))
    #print(len(allofthem))
