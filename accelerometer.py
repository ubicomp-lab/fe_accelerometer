import pandas as pd
import datetime
import numpy as np
import pytz
import scipy
import matplotlib.pyplot as plt
from collections import OrderedDict
from numpy import linalg as LA
import time
from matplotlib import style
from scipy import signal
from scipy import fftpack
import pywt


class Accelerometer:


        # acc_features = ['int_desc', 'int_rms', 'mag_desc', 'pear_coef', 'sma', 'svm', 'ecdf_5', 'fft', 'psd', 'lmbs']
    acc_features = ['int_desc', 'int_rms', 'mag_desc', 'pear_coef', 'sma', 'svm', 'ecdf_5', 'fft', 'psd']


    def __init__(self, config):
        self.df_acc = pd.read_csv(config['path'], header=None)
        self.df_acc.columns = ['_id1', '_id2', 'timestamp', 'device_id', 'double_x', 'double_y', 'double_z',
        'accuracy', 'label']
        self.df_acc.device_id='John Doe'
        self.df_acc = self.df_acc.sort_values(by=['timestamp']).reset_index(drop=True)[['timestamp', 'device_id', 'double_x',
         'double_y', 'double_z', 'accuracy', 'label']]
        self.window_size_in_minutes = config['window_size_in_minutes']
        self.mode = config['mode']





    def run_feature_extraction(self):
        window = 0
        window_start_time = self.df_acc.timestamp.iloc[0]
        window_end_time = self.df_acc.timestamp.iloc[-1]
        window_next_time = window_start_time + (datetime.timedelta(minutes=self.window_size_in_minutes).seconds * 10**3)
        window_id = 0
        window_start_index = 0
        self.df_acc['window_id'] = -1
        df_feature_windows = pd.DataFrame(columns= ['_window_id', 'start_time', 'end_time'])
        estimated_windows = (self.df_acc.timestamp.iloc[-1] - self.df_acc.timestamp.iloc[0]) / (self.window_size_in_minutes * 60000)
        print("Estimated no. of windows: ", estimated_windows)
        while window_next_time < window_end_time:
            window += 1
            print("window: ", window)
            print("Percentage: ", (window/estimated_windows) * 100, "%")
            window_indices = self.df_acc.iloc[window_start_index:].timestamp < window_next_time
            self.df_acc.window_id.iloc[window_start_index:][window_indices] = window_id
            feature_dict = self.featurize_window(self.df_acc.iloc[window_start_index:][window_indices], self.acc_features, self.mode, self.window_size_in_minutes)
            feature_dict.update({'_window_id': window_id, 'start_time': window_start_time, 'end_time': window_next_time - 1, 'sample_count':window_indices[window_indices].index.size})
            df_feature_windows = df_feature_windows.append(feature_dict, ignore_index=True)
            window_start_time = window_next_time
            window_next_time = window_next_time + (datetime.timedelta(minutes=self.window_size_in_minutes).seconds * 10**3)
            window_id = window_id + 1
            window_start_index = window_indices[~window_indices].index[0]

        window_indices = self.df_acc.iloc[window_start_index:].timestamp < window_next_time
        feature_dict = self.featurize_window(self.df_acc.iloc[window_start_index:], self.acc_features, self.mode, self.window_size_in_minutes)
        self.df_acc.window_id.iloc[window_start_index:][window_indices] = window_id
        feature_dict.update({'_window_id': window_id, 'start_time': window_start_time, 'end_time': window_next_time, 'sample_count':window_indices[window_indices].index.size})
        df_feature_windows = df_feature_windows.append(feature_dict, ignore_index=True)
        df_feature_windows.start_time = df_feature_windows.start_time.astype(np.int64)
        df_feature_windows.end_time = df_feature_windows.end_time.astype(np.int64)
        df_feature_windows.sample_count = df_feature_windows.sample_count.astype(np.int64)
        df_feature_windows._window_id = df_feature_windows._window_id.astype(np.int64)
        return df_feature_windows, self.df_acc



    def ecdfRep(data, components):
    #
    #   rep = ecdfRep(data, components)
    #
    #   Estimate ecdf-representation according to
    #     Hammerla, Nils Y., et al. "On preserving statistical characteristics of
    #     accelerometry data using their empirical cumulative distribution."
    #     ISWC. ACM, 2013.
    #
    #   Input:
    #       data        Nxd     Input data (rows = samples).
    #       components  int     Number of components to extract per axis.
    #
    #   Output:
    #       rep         Mx1     Data representation with M = d*components+d
    #                           elements.
    #
    #   Nils Hammerla '15
    #
        m = data.mean(0)
        data = np.sort(data, axis=0)
        data = data[np.int32(np.around(np.linspace(0,data.shape[0]-1,num=components))),:]
        data = data.flatten(1)
        return np.hstack((data, m))
    """
    modes
    0 - time domain only
    1 - time + frequency domain
    2 - time + frequency + stats
    3 - statistical methods only3
    """
    def featurize_window(self, df_fw, feature_list, mode, window_size_in_minutes):
        local_dict = OrderedDict()


        if mode > 0 and mode < 3:
            if df_fw.index.size >= (30*window_size_in_minutes):
                df_fw.double_x = df_fw.double_x.replace({0:1e-08})
                df_fw.double_y = df_fw.double_y.replace({0:1e-08})
                df_fw.double_z = df_fw.double_z.replace({0:1e-08})
                f_x = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_x)
                f_y = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_y)
                f_z = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_z)
                r = (np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2)).replace({0:1e-08})
                f_r = scipy.interpolate.interp1d(df_fw.timestamp, r)
                xnew = []
                step = (df_fw.timestamp.iloc[-1] - df_fw.timestamp.iloc[0]) /df_fw.index.size
                for ti in range(df_fw.timestamp.iloc[0], df_fw.timestamp.iloc[-1], int(step)):
                    xnew.append(ti)

                f_fs = window_size_in_minutes * 60 / df_fw.index.size
                L = 512 # change it to 512
                local_dict.update({'skip_fft':True, 'fx': f_x(xnew), 'fy': f_y(xnew), 'fz': f_z(xnew), 'fr': f_r(xnew), 'fs': f_fs, 'L': L})
            else:
                local_dict.update({'skip_fft':False})
            if df_fw.index.size == 0:
                local_dict['skip_td'] = False
            else:
                local_dict['skip_td'] = True

        if mode == 0:
            local_dict['skip_fft'] = True
        if mode == 3:
            local_dict['skip_fft'] = True
            local_dict['skip_td'] = True


        feat_dict = OrderedDict()



        for feature in feature_list:
            if feature == 'int_desc':
                if local_dict['skip_td']:
                    int_desc = np.sqrt((df_fw.double_x ** 2).describe() + (df_fw.double_y **2).describe() + (df_fw.double_z ** 2).describe())
                    feat_dict.update({'int_mean': int_desc[1], 'int_std': int_desc[2],
                                      'int_min': int_desc[3],'int_25': int_desc[4], 'int_50': int_desc[5],'int_75': int_desc[6]})
                else:
                    feat_dict.update({'int_mean': np.nan, 'int_std': np.nan,
                                      'int_min': np.nan,'int_25': np.nan, 'int_50': np.nan,'int_75': np.nan})
            elif feature == 'int_rms':
                if local_dict['skip_td']:
                    int_rms = np.sqrt((df_fw.double_x**2).sum() + (df_fw.double_y**2).sum() + (df_fw.double_z**2).sum()) / np.sqrt(df_fw.index.size)
                    feat_dict.update({'int_rms':int_rms})
                else:
                    feat_dict.update({'int_rms': np.nan})
            elif feature == 'mag_desc':
                if local_dict['skip_td']:
                    mag_desc = np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2).describe()
                    feat_dict.update({'mag_mean': mag_desc[1], 'mag_std': mag_desc[2], 'mag_min': mag_desc[3],
                                      'mag_25': mag_desc[4], 'mag_50': mag_desc[5],'mag_75': mag_desc[6]})
                else:
                    feat_dict.update({'mag_mean': np.nan, 'mag_std': np.nan, 'mag_min': np.nan,
                      'mag_25': np.nan, 'mag_50': np.nan,'mag_75': np.nan})
            elif feature == 'pear_coef':
                if local_dict['skip_td']:
                    cov_matrix =  np.cov(np.stack((df_fw.double_x,df_fw.double_y, df_fw.double_z), axis=0))
                    pear_coef_xy = cov_matrix[0,1] / (df_fw.double_x.std() * df_fw.double_y.std())
                    pear_coef_yz = cov_matrix[1,2] / (df_fw.double_y.std() * df_fw.double_z.std())
                    pear_coef_xz = cov_matrix[0,2] / (df_fw.double_x.std() * df_fw.double_z.std())
                    feat_dict.update({'pear_coef_xy':pear_coef_xy, 'pear_coef_yz':pear_coef_yz,'pear_coef_xz':pear_coef_xz })
                else:
                    feat_dict.update({'pear_coef_xy':np.nan, 'pear_coef_yz':np.nan,'pear_coef_xz':np.nan})
            elif feature == 'sma':
                if local_dict['skip_td']:
                    sma = (np.abs(df_fw.double_x.to_numpy()).sum() + np.abs(df_fw.double_y.to_numpy()).sum() + np.abs(df_fw.double_z.to_numpy()).sum()) / df_fw.index.size
                    feat_dict.update({'sma':sma})
                else:
                    feat_dict.update({'sma':np.nan})
            elif feature == 'svm':
                if local_dict['skip_td']:
                    svm = np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2).sum() / df_fw.index.size
                    feat_dict.update({'svm':svm})
                else:
                    feat_dict.update({'svm':np.nan})
            elif feature == 'fft':
                if local_dict['skip_fft']:
                    L = local_dict['L']
                    dfx = fftpack.fft(local_dict['fx'], 512)
                    dfy = fftpack.fft(local_dict['fy'], 512)
                    dfz = fftpack.fft(local_dict['fz'], 512)
                    dfr = fftpack.fft(local_dict['fr'], 512)
                    # DC component
                    # Remove the L part!
                    feat_dict.update({'fdc_x': np.mean(np.real(dfx)), 'fdc_y': np.mean(np.real(dfy)),
                                      'fdc_z':  np.mean(np.real(dfz)), 'fdc_r':  np.mean(np.real(dfr))})
                    # Energy
                    feat_dict.update({'feng_x': (np.sum(np.real(dfx)**2 + np.imag(dfx)**2)) / L, 'feng_y': (np.sum(np.real(dfy)**2 + np.imag(dfy)**2)) / L,
                                      'feng_z':  (np.sum(np.real(dfz)**2 + np.imag(dfz)**2)) / L, 'feng_r':  (np.sum(np.real(dfr)**2 + np.imag(dfr)**2)) / L})
                    # Entropy
                    ck_x = np.sqrt(np.real(dfx)**2  + np.imag(dfx)**2)
                    cj_x = ck_x / np.sum(ck_x)
                    e_x = np.sum(cj_x * np.log(cj_x))

                    ck_y = np.sqrt(np.real(dfy)**2  + np.imag(dfy)**2)
                    cj_y = ck_y / np.sum(ck_y)
                    e_y = np.sum(cj_y * np.log(cj_y))

                    ck_z = np.sqrt(np.real(dfz)**2  + np.imag(dfz)**2)
                    cj_z = ck_z / np.sum(ck_z)
                    e_z = np.sum(cj_z * np.log(cj_z))

                    ck_r = np.sqrt(np.real(dfr)**2  + np.imag(dfr)**2)
                    cj_r = ck_r / np.sum(ck_r)
                    e_r = np.sum(cj_r * np.log(cj_r))

                    feat_dict.update({'fent_x': e_x, 'fent_y':  e_y,'fent_z':  e_z, 'fent_r': e_r})

                    # Correlation
                    # Fix the length, should be FFT wndow size 512

                    fcorr_xy = np.dot(np.real(dfx) / L, np.real(dfy) / L)
                    fcorr_xz = np.dot(np.real(dfx) / L, np.real(dfz) / L)
                    fcorr_yz = np.dot(np.real(dfy) / L, np.real(dfz) / L)

                    feat_dict.update({'fcorr_xy': fcorr_xy,'fcorr_xz':  fcorr_xz, 'fcorr_yz': fcorr_yz})

                else:
                    feat_dict.update({'fdc_x': np.nan, 'fdc_y':  np.nan,'fdc_z':  np.nan, 'fdc_r': np.nan})
                    feat_dict.update({'feng_x':  np.nan, 'feng_y':  np.nan, 'feng_z':   np.nan, 'feng_r':   np.nan})
                    feat_dict.update({'fent_x': np.nan, 'fent_y':  np.nan,'fent_z':  np.nan, 'fent_r': np.nan})
                    feat_dict.update({'fcorr_xy': np.nan,'fcorr_xz':  np.nan, 'fcorr_yz': np.nan})
            elif feature == 'psd':
                if local_dict['skip_fft']:
                    fs = local_dict['fs']
                    psd_window = signal.get_window('boxcar', len(local_dict['fx'])) # do not pass this window
                    freqs_x, pxx_denx = signal.periodogram(local_dict['fx'], window=psd_window, fs=fs)
                    freqs_y, pxx_deny = signal.periodogram(local_dict['fy'], window=psd_window, fs=fs)
                    freqs_z, pxx_denz = signal.periodogram(local_dict['fz'], window=psd_window, fs=fs)
                    freqs_r, pxx_denr = signal.periodogram(local_dict['fr'], window=psd_window, fs=fs)
                    feat_dict.update({'psd_mean_x': np.mean(pxx_denx), 'psd_mean_y': np.mean(pxx_deny),
                                      'psd_mean_z': np.mean(pxx_denz), 'psd_mean_r': np.mean(pxx_denr)})

                    feat_dict.update({'psd_max_x': np.max(pxx_denx),
                                      'psd_max_y': np.max(pxx_deny),
                                      'psd_max_z': np.max(pxx_denz),
                                      'psd_max_r': np.max(pxx_denr)})


                    freqs_05_3_x = np.argwhere((freqs_x >= 0.5) & (freqs_x <= 3))
                    freqs_05_3_y = np.argwhere((freqs_y >= 0.5) & (freqs_y <= 3))
                    freqs_05_3_z = np.argwhere((freqs_z >= 0.5) & (freqs_z <= 3))
                    freqs_05_3_r = np.argwhere((freqs_r >= 0.5) & (freqs_r <= 3))


                    # max b/w 0.3 - 3Hz
                    # 0.5 - 3 Hz if missing, maybe not 0.0
                    feat_dict.update({'psd_max_x_05_3': np.max(pxx_denx[freqs_05_3_x]) if freqs_05_3_x.any() else 0.0,
                      'psd_max_y_05_3': np.max(pxx_deny[freqs_05_3_y]) if freqs_05_3_y.any() else 0.0,
                      'psd_max_z_05_3': np.max(pxx_denz[freqs_05_3_z]) if freqs_05_3_z.any() else 0.0,
                      'psd_max_r_05_3': np.max(pxx_denr[freqs_05_3_r]) if freqs_05_3_r.any() else 0.0})
                else:
                    feat_dict.update({'psd_mean_x': np.nan, 'psd_mean_y':np.nan,
                                      'psd_mean_z': np.nan, 'psd_mean_r': np.nan})
                    feat_dict.update({'psd_max_x': np.nan,
                                      'psd_max_y': np.nan,
                                      'psd_max_z': np.nan,
                                      'psd_max_r': np.nan})
            elif feature == 'lmbs':
                if local_dict['skip_td']:
                    lmb_f_05_3 = np.linspace(0.5, 3, 10000)
                    lmb_psd_x = signal.lombscargle(df_fw.timestamp, df_fw.double_x, lmb_f_05_3, normalize=False)
                    lmb_psd_y = signal.lombscargle(df_fw.timestamp, df_fw.double_y, lmb_f_05_3, normalize=False)
                    lmb_psd_z = signal.lombscargle(df_fw.timestamp, df_fw.double_z, lmb_f_05_3, normalize=False)

                    feat_dict.update({'lmb_psd_max_x_05_3': np.max(lmb_psd_x) if lmb_psd_x.any() else 0.0,
                      'lmb_psd_max_y_05_3': np.max(lmb_psd_y) if lmb_psd_y.any() else 0.0,
                      'lmb_psd_max_z_05_3': np.max(lmb_psd_z) if lmb_psd_z.any() else 0.0})
                else:
                    feat_dict.update({'lmb_psd_max_x_05_3': np.nan,
                      'lmb_psd_max_y_05_3': np.nan,
                      'lmb_psd_max_z_05_3': np.nan})


        return feat_dict

def main():
    sample_config = {'window_size_in_minutes':1, 'path': "../../../data/accelerometer/835b51bd-ee31-49e8-a653-cb75a7e4c98e.csv", 'mode':1}
    accelerometer = Accelerometer(sample_config)
    accelerometer.run_feature_extraction()
if __name__ == '__main__':
    main()
