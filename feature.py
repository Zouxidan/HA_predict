import numpy as np
import pandas as pd
from ilearnplus import FileProcessing as fp
import time


def get_feature(file):
    all_data = []
    #AAC
    start = time.time()
    train_feature=fp.Descriptor(file, 0)
    train_feature.Protein_AAC()
    acc=train_feature.encoding_array[1:,1:]
    acc=pd.DataFrame(acc)
    all_data.append(acc)
    end = time.time()
    print('*AAC Completed...\ntime: %s Seconds\n' % (end - start))


    #CKSAAP
    start = time.time()
    kw1={'kspace':1}
    train_feature=fp.Descriptor(file, kw1)
    train_feature.Protein_CKSAAP()
    cksaap=train_feature.encoding_array[1:,1:]
    cksaap = pd.DataFrame(cksaap)
    all_data.append(cksaap)
    end = time.time()
    print('*CKSAAP Completed...\ntime: %s Seconds\n' % (end - start))


    #DistancePair
    start = time.time()
    kw2 = {
        'distance': 4, 'cp': 'cp(19)'
    }
    train_feature = fp.Descriptor(file,kw2)
    train_feature.Protein_DistancePair()
    dcp=train_feature.encoding_array[1:,1:]
    dcp = pd.DataFrame(dcp)
    all_data.append(dcp)
    end = time.time()
    print('*DistancePair Completed...\ntime: %s Seconds\n' % (end - start))

    #DPC
    start = time.time()
    train_feature = fp.Descriptor(file, 0)
    train_feature.Protein_DPC()
    dpc=train_feature.encoding_array[1:,1:]
    dpc=pd.DataFrame(dpc)
    all_data.append(dpc)
    end = time.time()
    print('*DPC Completed...\ntime: %s Seconds\n' % (end - start))

    #TPC
    start = time.time()
    train_feature = fp.Descriptor(file, 0)
    train_feature.Protein_TPC()
    tpc=train_feature.encoding_array[1:,1:]
    tpc=pd.DataFrame(tpc)
    all_data.append(tpc)
    end = time.time()
    print('*TPC Completed...\ntime: %s Seconds\n' % (end - start))

    #PAAC
    start = time.time()
    kw3 = {
        'lambdaValue':10,'weight':0.7
    }
    train_feature = fp.Descriptor(file,kw3)
    train_feature.Protein_PAAC()
    paac=train_feature.encoding_array[1:,1:]
    paac = pd.DataFrame(paac)
    all_data.append(paac)
    end = time.time()
    print('*PAAC Completed...\ntime: %s Seconds\n' % (end - start))

    return all_data