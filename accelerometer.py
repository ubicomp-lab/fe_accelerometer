from matplotlib import style
from scipy import signal
from scipy import fftpack
from collections import OrderedDict
from numpy import linalg as LA
from optparse import OptionParser
import os
import pandas as pd
import datetime
import numpy as np
import pytz
import scipy
import matplotlib.pyplot as plt
import time
import pywt



parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="run feature extraction on FILE", metavar="FILE")
parser.add_option("-d", "--dir", dest="dir",
                  help="run feature extraction on all files in DIR", metavar="DIR")
parser.add_option("-o", "--out", dest="out",
                  help="output features to OUTDIR", metavar="OUTFILE")
parser.add_option("-v", "--verbose",
                  action="store_false", dest="verbose", default=True,
                  help="don't print verbose messages to stdout")
parser.add_option("-r", "--restart",
                    action="store_true" , default=False,
                  help="restart by rewriting output files present in OUTDIR")


out_dir = os.path.join(os.getcwd(),  'out')

def main():
    (options, args) = parser.parse_args()
    print(options.restart)
    if not options.filename and not options.dir:
        parser.error('missing -f or -d option')

    if options.filename and options.dir:
        parse.error('please choose only one option to run feature extraction')

    if options.out:
        if not os.path.exists(option.out):
            parse.error('output path does not exist, storing output files in out/ instead')
        else:
            out_dir = option.out

    if options.filename:
        if not os.path.isfile(options.filename):
            parse.error('input file does not exist')
        else:
            load_input(options.filename, options.restart)

    if options.dir:
        if not os.path.isdir(options.dir):
            parse.error('input directory does not exist')
        else:
            load_input(options.dir, options.restart)


def restart():
    return 1


def process_file(file):
    print("Running feature extraction on", file)
    return 1

def load_input(path, should_restart):




    out_file_ids= []
    for file in os.listdir(out_dir):
        if file.endswith(".csv"):
            out_file_ids.append(file)
    s_out_ids = pd.Series(out_file_ids).str[4:]


    # if should restart



    if os.path.isfile(path):
        print(out_file_ids)
        print(os.path.basename(path))
        if os.path.basename(path) in s_out_ids.to_list():
            print("features already extracted from this file. TIP: run with restart option to rewrite")
            return 1

    else:
        in_file_ids = []
        in_files = []

        print("Looking for files in", path)
        for file in os.listdir(path):
            if file.endswith(".csv"):
                in_file_ids.append(file)


        print("Found", len(in_file_ids), "files")


        s_in_ids = pd.Series(in_file_ids)


        s_in_ids = s_in_ids[~s_in_ids.isin(s_out_ids)].to_list()
        print(s_in_ids)






        print("Starting feature extraction process...")

        for s_in_id in s_in_ids:
            process_file((os.path.join(os.path.dirname(path), s_in_id)))
    return 1




if __name__ == '__main__':
    main()
