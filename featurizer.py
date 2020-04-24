import os
import io
import pandas as pd
import configparser

class Featurizer():
    def __init__(self):
        self.in_path = os.getcwd()
        self.out_dir = os.path.join(os.getcwd(), 'out/')
        self.should_restart = False
        self.file_list= []
        self.read_config()

    def read_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        config_dict = {}
        accelerometer_config_dict = {}
        try:
            window_size_in_minutes = int(config['accelerometer']['window_size_in_minutes'])
            mode = int(config['accelerometer']['mode'])
            if window_size_in_minutes < 1 or window_size_in_minutes > 30:
                print("Illegal value found for config in accelerometer section: 'window_size_in_minutes'")
                raise Exception()
            if mode < 0 or mode > 3:
                print("Illegal value found for config in accelerometer section: 'mode'")
                raise Exception()
            accelerometer_config_dict.update({"window_size_in_minutes":window_size_in_minutes, "mode":mode})
        except:
            print("Error reading config file.TIP:")
            exit(1)
        config_dict.update({'accelerometer':accelerometer_config_dict})
        print(config_dict)

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def set_in_path(self, in_path):
        self.in_path = in_path

    def featurize(self, file):
        print("Extracting features from", file)
        return 1

    def prepare(self):
        extracted_list = []
        for file in os.listdir(self.out_dir):
            if file.endswith(".csv"):
                extracted_list.append(file)
        if any(extracted_list):
            extracted_list = pd.Series(extracted_list).str[4:].to_list()
        if os.path.isfile(self.in_path):
            if os.path.basename(self.in_path) in extracted_list.to_list():
                print("Features already extracted from this file. TIP: run with restart option to rewrite")
                return 1
            else:
                self.file_list.append(self.in_path)
        else:
            found_file_list = []
            print("Looking for files in", self.in_path)
            for file in os.listdir(self.in_path):
                if file.endswith(".csv"):
                    found_file_list.append(file)
            print("Found", len(found_file_list), "files")
            pending_file_list = pd.Series(found_file_list)
            already_extracted_list = pending_file_list[pending_file_list.isin(extracted_list)].to_list()
            self.file_list = pending_file_list[~pending_file_list.isin(extracted_list)].to_list()
            if len(already_extracted_list):
                print("Features already extracted from", len(already_extracted_list),
                "out of", len(found_file_list), "files")
                print("Continuing feature extraction with the remaining", len(self.file_list), "files.",
                "TIP: run with restart option to rewrite")

    def run(self):
        print("Starting feature extraction process...")
        for file in self.file_list:
            print("Extracting features from", file)
        return 1
