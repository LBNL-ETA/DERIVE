import os
import sys
import pandas as pd
import numpy as np

try:
    root = os.path.dirname(os.path.realpath(__file__))
except:
    root = os.getcwd()
    
sys.path.append(os.path.join(root, '..', 'environment'))

from simplecoupled import simplecoupled
from configuration import get_simplecoupled_config

class me_handler(object):
    def __init__(self, fmu_path=None, log_level=0, config=None, kind=None):
        self.fmu_path = fmu_path
        self.log_level = log_level
        self.config = config
        if config == None:
            self.config = kind
        self.init = True
        self.fmu_init_param = {}
        
    def setup_experiment(self, start_time=0, stop_time=1, stop_time_defined=False, tolerance=None):
        if self.config == None:
            print('WARNING: Loading default configuration for IES.')
            self.config = get_simplecoupled_config(start_time=self.start_time, final_time=self.final_time,
                                                   step_size=self.step_size, log_level=self.log_level, pv=False)        
        
        self.step_size = self.config['step_size']
        self.step_size_internal = self.config['step_size_internal']
        self.simulator = simplecoupled(self.config)
        if self.fmu_init_param != {}:
            self.simulator.set(list(self.fmu_init_param.keys()), list(self.fmu_init_param.values()))
        self.time = self.simulator.fmu_time
        
    def set(self, keys, values):
        param = {k:v for k,v in zip(keys, values)}
        if 'step_size' in param.keys():
            self.step_size = param['step_size']
            del param['step_size']
        if 'final_time' in param.keys():
            self.final_time = param['final_time']
            del param['final_time']
        if 'start_time' in param.keys():
            self.start_time = param['start_time']
            del param['start_time']
        if self.init:
            self.fmu_init_param = param
        else:
            self.fmu_init_param = {}
            self.simulator.set(keys, values)
            
    def resample_get(self, df):
        if 'soc' in df.name.lower():
            return df.resample('{}S'.format(self.step_size)).apply(lambda x: x.iloc[-1])
        else:
            return df.resample('{}S'.format(self.step_size)).mean()
        
    
    def get(self, key):
        if isinstance(key, str):
            key = [key]
        if self.init:
            print('WARNING: Function "get" during initialization.')
            res = []
            for k in key:
                f, n = k.split('.', 1)
                fx = self.simulator.fmu_names.index(f)
                res.append(self.simulator.fmus[fx].get(n))
            res = np.array(res)
            return res.reshape(res.shape[0])
        
        data = self.simulator.data[key].resample('{}S'.format(self.step_size_internal)).mean()
        data.index = data.index.shift(-1)
        data = pd.DataFrame(data.apply(lambda x: self.resample_get(x)))
        return data.iloc[-1].values

    def initialize(self):
        pass
    
    def simulate(self, inputs=pd.DataFrame(), current_t=0, step_size=1):
        self.simulator.simulate(inputs=inputs)
        self.simulator.data = self.simulator.data.resample('{}S'.format(self.step_size)).mean()
        self.init = False
        
    def do_step(self, inputs=pd.DataFrame(), current_t=0, step_size=None):
        res = self.simulator.do_step(inputs, step_size)    
        self.time = self.simulator.fmu_time
        self.init = False
        return res
    
    def terminate(self):
        self.simulator.terminate()
        self.init = True

class inputs_handler(object):
    def __init__(self, parameter):
        self.parameter = parameter

    def get_inputs(self, actions):
        actions = pd.DataFrame(actions)
        actions.index = [pd.to_datetime(x, format='%H') for x in actions.index]
        #actions = actions.resample('{}S'.format(self.parameter['step_size'])).ffill()
        actions.index = [(x - actions.index[0]).total_seconds() for x in actions.index]
        actions = actions.reset_index().values
        return actions