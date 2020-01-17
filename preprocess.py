import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
from scipy.optimize import curve_fit

class capacitor:

    def __init__(self, file, batch, label):
        self.label = label
        self.cycle = None
        self.step = None
        self.detail = np.array([])
        self.life = None
        self.loglife = None
        self.batch = batch
        self.read_xls(file)


        if batch == 'b1':
            self.get_ini_end_voltage()

        self.cal_time(reset=False)
        self.cal_cycle_capacitance()
        self.set_life(rated=True, set=True, ratio=0.903)


    def read_xls(self, file):


        cycle = pd.read_excel(file, 1)

        if self.batch == 'b1':
            self.cycle = cycle.iloc[:, [0, 1, 2]]
            self.cycle.columns = ['cycle', 'charge_capacity(mAh)',
                                  'discharge_capacity(mAh)']
        else:
            self.cycle = cycle.iloc[:, [0, 1, 2, 3, 4]]
            self.cycle.columns = ['cycle',
                                  'charge_capacity(mAh)',
                                  'discharge_capacity(mAh)',
                                  'charge_energy(mWh)',
                                  'discharge_energy(mWh)']

        step = pd.read_excel(file, 0)
        if self.batch == 'b1':
            self.step = step.iloc[:, [0,1,2,3,4,5]]
            self.step.columns = ['cycle',
                                 'step',
                                 'status',
                                 'step_time',
                                 'capacity(mAh)',
                                 'energy(mWh)']
        else:
            self.step = step.iloc[:, [0,1,2,3,4,5,6,7]]
            self.step.columns = ['cycle',
                                 'step',
                                 'status',
                                 'step_time',
                                 'capacity(mAh)',
                                 'energy(mWh)',
                                 'ini_voltage(V)',
                                 'end_voltage(V)']


        sheet_idx = 2
        i = 1
        file_ori = file
        # Multiple files
        while 1:
            while 1:
                try:
                    if self.batch == 'b1':
                        detail_info = pd.read_excel(file, sheet_idx).iloc[:, [0,1,2,4,5,6,7]]
                        detail_info.columns = ['cycle',
                                               'step',
                                               'status',
                                               'record_time',
                                               'voltage(V)',
                                               'current(mA)',
                                               'capacity(mAh)']
                    else:
                        detail_info = pd.read_excel(file, sheet_idx).iloc[:, [0,1,2,4,5,6,7,8]]
                        detail_info.columns = ['cycle',
                                               'step',
                                               'status',
                                               'record_time',
                                               'voltage(V)',
                                               'current(mA)',
                                               'capacity(mAh)',
                                               'energy(mWh)']

                    if self.detail.size == 0:
                        self.detail = detail_info
                    else:
                        self.detail = pd.DataFrame(np.vstack((self.detail, detail_info)))
                        if self.batch == 'b1':
                            self.detail.columns = ['cycle',
                                                   'step',
                                                   'status',
                                                   'record_time',
                                                   'voltage(V)',
                                                   'current(mA)',
                                                   'capacity(mAh)']
                        else:
                            self.detail.columns = ['cycle',
                                                   'step',
                                                   'status',
                                                   'record_time',
                                                   'voltage(V)',
                                                   'current(mA)',
                                                   'capacity(mAh)',
                                                   'energy(mWh)']
                    print(sheet_idx)
                    sheet_idx += 1
                except:
                    break


            file = file_ori[:-4] + f'__{i}' + '.xls'
            if not os.path.exists(file):
                break
            sheet_idx = 0
            i = i + 1



    # cycle life
    def set_life(self, rated=True, set=False,start_cycle=2, ratio=0.8):
        if rated:
            end_capacitance = ratio
        else:
            end_capacitance = self.get_data_from_cycle('cycle', 'discharge_capacitance(F)', start_cycle) * ratio

        try:
            life_cand = self.cycle.loc[self.cycle['discharge_capacitance(F)'] < end_capacitance, 'cycle'].values
            life = life_cand[1]
            loglife = np.log10(life)
        except:
            life = 0
            loglife = 0

        if set:
            self.life = life
            self.loglife = loglife
        else:
            return life

    # get cyclic data at some cycle
    def get_data_from_cycle(self, sheet, column, cycle, discharge=True):
        assert sheet in ['step', 'cycle', 'detail']
        if sheet == 'cycle':
            cyc_data = self.cycle.loc[self.cycle['cycle'] == cycle, column].values[0]
            return cyc_data

        elif sheet == 'step':
            cyc_data = self.step.loc[self.step['cycle'] == cycle, :]
            if discharge:
                return cyc_data.loc[cyc_data['status'] == 0, column].values[0]
            else:
                assert cycle > 1
                return cyc_data.loc[cyc_data['status'] == 1, column].values[0]
        else:
            cyc_data = self.detail.loc[self.detail['cycle'] == cycle,: ]
            if discharge:
                return cyc_data.loc[cyc_data['status'] == 0, column]
            else:
                return cyc_data.loc[cyc_data['status'] == 1, column]



    # Convert time(h:min:second) to seconds
    def cal_time(self, reset=True):
        if reset:
            t1 = []
            t2 = []
            trans = 0

            for rt in self.step['step_time']:
                dt = datetime.strptime(rt, "%H:%M:%S.%f")
                seconds = dt.hour*3600+dt.minute*60+dt.second+dt.microsecond/1e6
                t1.append(seconds)
                t2.append(seconds + trans)
                trans = seconds + trans
            self.step['step_time'] = pd.Series(t1)



            t1 = []
            t2 = []
            trans = 0

            for rt in self.detail['record_time']:
                dt = datetime.strptime(rt, "%H:%M:%S.%f")
                seconds = dt.hour*3600+dt.minute*60+dt.second+dt.microsecond/1e6
                if (seconds == 0)and(len(t2) >= 1) : trans = t2[-1]
                t1.append(seconds)
                t2.append(seconds + trans)
            self.detail['record_time'] = pd.Series(t1)

        else:
            t1 = []
            t2 = []
            last_s = 0
            for rt in self.step['step_time']:
                t = rt.split(':')
                hour = int(t[0])
                minute = int(t[1])
                second = float(t[2])
                total_s = hour * 3600 + minute * 60 + second
                t1.append(total_s)

            self.step['step_time'] = pd.Series(t1)

            t1 = []
            t2 = []
            last_s = 0
            reset_s = 0
            for rt in self.detail['record_time']:
                t = rt.split(':')
                hour = int(t[0])
                minute = int(t[1])
                second = float(t[2])
                total_s = hour * 3600 + minute * 60 + second
                if total_s == 0:
                    t1.append(0)
                else:
                    t1.append(total_s-last_s)
                last_s = total_s
            self.detail['record_time'] = pd.Series(t1)

            t = []
            total_s = 0
            for rt in self.detail['record_time']:
                if rt == 0:
                    t.append(0)
                    total_s = 0
                else:
                    total_s += rt
                    t.append(total_s)
            self.detail['total_time'] = pd.Series(t)

    # only for batch1, obtain ini_voltage and end_voltage for each step
    def get_ini_end_voltage(self):
        ini_v = [0]
        end_v = [0]
        for cycle in self.cycle['cycle'][1:]:
            vol_c = self.get_data_from_cycle('detail', 'voltage(V)', cycle, discharge=False).values
            vol_d = self.get_data_from_cycle('detail', 'voltage(V)', cycle, discharge=True).values
            if len(vol_c) > 0 and len(vol_d) > 0:
                ini_v.append(vol_c[0])
                ini_v.append(vol_d[0])
                end_v.append(vol_c[-1])
                end_v.append(vol_d[-1])
        ini_v.append(vol_c[0])
        end_v.append(vol_c[-1])


        ini_v = np.array(ini_v)
        end_v = np.array(end_v)
        self.step['ini_voltage(V)'] = ini_v
        self.step['end_voltage(V)'] = end_v

    # capacitance at each cycle
    def cal_cycle_capacitance(self, has_ini_and_end=True):
        # skip the first cycle
        caps = [0, 0]

        if has_ini_and_end:
            end_V = self.step.loc[self.step['status'] == 0, ['end_voltage(V)']].values
            ini_V = self.step.loc[self.step['status'] == 0, ['ini_voltage(V)']].values
            t = self.step.loc[self.step['status'] == 0, ['step_time']].values
            charge_capacitance = -20*t/(end_V - ini_V)/1000

            if self.batch == 'b1':
                charge_capacitance = np.append(charge_capacitance, 0)
            self.cycle['discharge_capacitance(F)'] = charge_capacitance

        else:

            for cycle in self.cycle['cycle'][1:]:
                if cycle % 100 == 0:
                    print(cycle)
                # charge
                vol = self.get_data_from_cycle('detail', 'voltage(V)', cycle, discharge=False)
                if len(vol) > 0:
                    vol = vol.values
                    curr = self.get_data_from_cycle('detail', 'current(mA)', cycle, discharge=False).values[0]
                    step_time = self.get_data_from_cycle('step', 'step_time', cycle, discharge=False)
                    ini_vol = vol[0]
                    end_vol = vol[-1]
                    capacitance = curr * step_time / (end_vol - ini_vol)
                    caps.append(capacitance)
                else:
                    break

                # discharge
                vol = self.get_data_from_cycle('detail', 'voltage(V)', cycle)
                if len(vol) > 0:
                    vol = vol.values
                    curr = self.get_data_from_cycle('detail', 'current(mA)', cycle).values[0]
                    step_time = self.get_data_from_cycle('step', 'step_time', cycle)
                    ini_vol = vol[0]
                    end_vol = vol[-1]
                    capacitance = curr * step_time / (end_vol - ini_vol)
                    caps.append(capacitance)
                else:
                    caps.append(0)



            if len(caps) % 2 == 1:
                caps.append(0)
            caps = np.array(caps).reshape(-1, 2)

            self.cycle['charge_capacitance(F)'] = caps[:, 0] / 1000
            self.cycle['discharge_capacitance(F)'] = caps[:, 1] / 1000


    # voltage drop at different intervals of each cycle
    def V_drop_with_cycle(self, cycle, x):

        assert x >= 1
        voltage = self.get_data_from_cycle('detail', 'voltage(V)', cycle).values
        # interval = 10s
        if self.batch == 'b1' or self.batch == 'b2':
            V_drop = voltage[x-1] - voltage[x]
        # interval = 1s
        else:
            V_drop = voltage[(x-1)*10] - voltage[x*10]


        return V_drop


    def IR_drop(self, cycle):

        voltage = self.get_data_from_cycle('detail', 'voltage(V)', cycle).values[0]
        last_vol = self.get_data_from_cycle('detail', 'voltage(V)', cycle - 1, discharge=False).values[-1]
        drop = voltage - last_vol
        return drop



    def decay_fit(self):
        N = np.array([i for i in range(1, 10001)])
        C_decay = [0]
        for cycle in range(2, 10000):
            C = self.get_data_from_cycle('cycle', 'discharge_capacitance(F)', cycle)
            C_decay.append(C)
        C_decay = np.array(C_decay)
        popt, pcov = curve_fit(power_func, N[30:657], C_decay[30:657], maxfev=50000000)
        return popt

class data:

    def __init__(self, file):
        self.caps = []

        self.bin_file = file


    def read_data(self, batch, path):
        files = os.listdir(path)
        files = [file for file in files if '__' not in file]
        files.sort(key=lambda x: int(x[:-4]))
        for i, file in enumerate(files):
            print(file)
            cap = capacitor(path+'/'+file, batch, i+1)


            self.caps.append(cap)


    def save_data(self):

        with open(self.bin_file, 'wb') as f:
            pickle.dump(self, f)


    def produce_minidata(self):
        cycle = 1000
        for i in range(len(self.caps)):
            self.caps[i].step = self.caps[i].step.loc[self.caps[i].step['cycle'] <= cycle, :]
            self.caps[i].detail = self.caps[i].detail.loc[self.caps[i].detail['cycle'] <= cycle, :]
        self.bin_file = self.bin_file + '_Mini'
        with open(self.bin_file, 'wb') as f:
            pickle.dump(self, f)





# The index we use
def index_split(caps):
    n_sample = len(caps)
    train_index = []
    test_index = []
    index = 1
    while index < n_sample:
        test_index.append(round(index))
        index = index + 4
    train_index = [i for i in range(n_sample) if i not in test_index]
    train_index = np.array(train_index)
    test_index = np.array(test_index)



    # print("Train Index:", train_index, ",Test Index:", test_index)


    return train_index, test_index

def power_func(x, a, b):
    return a * np.power(x, b)
