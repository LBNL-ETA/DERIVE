import sys
import random
import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class ies_fmu_handler(py_environment.PyEnvironment): 
    ''' 
    Training environment using an fmu for controls development and evaluation.        
    '''
    def __init__(self, parameter, pyfmi=None, parameter_sim=None):
        '''
        Setup the py_environment.
        
        Inputs
        parameter (dict): Configuraiton dictionary.
        pyfmi (class): pyfmi handler. None defaults to pyfmi package. Default = None.       
        '''
        
        # Parse inputs
        self.parameter = parameter
        self.parameter_sim = parameter_sim
        
        self.init = True
        
        # Establish sclaing for RL
        self.scaler = self.parameter['scaler']
        
        # Valirables for RL
        self.reward = 0
        self.data = pd.Series(dtype=np.float64)

        self._action_spec = array_spec.BoundedArraySpec(shape=(len(parameter['action_names']),), 
            minimum=parameter['action_min'], maximum=parameter['action_max'], dtype=np.float64, name='action')        
        self._observation_spec = array_spec.BoundedArraySpec(shape=(len(parameter['observation_names']),), 
            dtype=np.float64, name='observation')

        # FMI interface (load FMU)
        self.setup_pyfmi(pyfmi)
            
        # Use Python preprocessing before calling FMU
        if parameter['preprocessor']:
            self.preprocessor = eval('{}(self.parameter)'.format(parameter['preprocessor']))
        else:
            self.preprocessor = None
            
        # Use Python postprocessing after calling FMU
        if parameter['postprocessor']:
            self.postprocessor = eval('{}(self.parameter)'.format(parameter['postprocessor']))
        else:
            self.postprocessor = None
        
    def setup_pyfmi(self, pyfmi):
        if pyfmi != None:
            self.load_fmu = pyfmi            
        else:
            from pyfmi import load_fmu
            self.load_fmu = load_fmu
        self.fmu_loaded = False
            
    def configure_fmu(self, ext_param, start_time):
        '''
        Load and setup the FMU.
        
        Inputs
        ext_param (dict): Initial temperature for model, in K.
        start_time (float): Start time of the model, in sceonds.
        '''
        # Load FMU
        self.fmu = self.load_fmu(self.parameter['fmu_path'],
                                 log_level=self.parameter['fmu_loglevel'])
        if self.parameter_sim:
            self.fmu.config = self.parameter_sim
        
        # Parameterize FMU
        param = self.parameter['param']
        param.update(ext_param)
        if param != {}:
            self.fmu.set(list(param.keys()), list(param.values()))
            
        # Initizlaize FMU
        self.fmu.setup_experiment(start_time=start_time,
                                  stop_time=start_time+1,
                                  stop_time_defined=False)
        self.fmu.initialize()
        self.fmu_loaded = True
        
    def action_spec(self):
        '''tf_agents wrapper function'''
        return self._action_spec
    
    def observation_spec(self):
        '''tf_agents wrapper function'''
        return self._observation_spec
    
    def get_info(self):
        '''function returns variables calculated in the environment'''
        return self.data.index, self.data.values
        
    def evaluate_fmu(self, inputs, states):
        ''' evaluate the fmu '''

        if not self.fmu_loaded:
            # Careful, always initialized with 0 as start_time
            self.configure_fmu({}, self.parameter['start_time'])
            
        if self.mode == 'cosim':
            # Set Inputs
            fmu_inputs = {}
            for k, v in self.parameter['inputs_map'].items():
                fmu_inputs[k] = inputs.loc[v]

            # Add inputs from parameters
            for k, v in self.parameter['inputs'].items():
                 fmu_inputs[k] = v
            
            res = self.fmu.do_step(inputs=pd.DataFrame(fmu_inputs, index=[self.fmu.time]))
            res = res.iloc[:-1]
            res = res.resample('{}S'.format(self.parameter['step_size'])).mean()
        else:
            inputs = inputs.copy().rename(columns={v:k for k,v in self.parameter['inputs_map'].items()})
            del inputs['time']
            self.fmu.simulate(inputs=inputs)
            res = self.fmu.get(self.parameter['observation_names'])
            res = res.resample('{}S'.format(inputs.index[1]-inputs.index[0])).mean()
            res.index = [(x - res.index[0]).total_seconds() for x in res.index]

        return res
    
    def _step(self, inputs):
        
        # Parse inputs
        if len(inputs.shape) == 1 or inputs.shape[-1] == 1:
            self.mode = 'cosim'
            if len(inputs) == len(self.parameter['input_labels'])-1:
                # add time
                start_time = self.fmu.time if self.fmu_loaded else self.parameter['start_time']
                inputs = np.insert(inputs, 0, start_time, axis=0)
            data = pd.Series(inputs, index=self.parameter['input_labels'])
        else:
            self.mode = 'sim'
            data = pd.DataFrame(inputs, columns=self.parameter['input_labels'])
            data.index = data['time'].values
        
        # Compute preprocessing (if specified)
        if self.preprocessor:
            data = self.preprocessor.do_calc(data, self.init)
        
        # Evaluate FMU 
        res = self.evaluate_fmu(data, self.state)
        
        if self.mode == 'cosim':
            for k, v in res.items():
                data[k] = v.values[0]
        else:
            data.loc[res.index[-1], :] = np.nan
            for k, v in res.items():
                data[k] = v
            
        # Compute postprocessing (if specified)
        if self.postprocessor:
            data = self.postprocessor.do_calc(data, self.init)
        
        axis = 0 if self.mode == 'cosim' else 1
        data['reward_0'] = data[[x for x in self.parameter['observation_names'] if 'cost' in x]].sum(axis=axis) \
            / self.scaler['energy_cost']
        data['reward_1'] = data[[x for x in self.parameter['observation_names'] if 'demand' in x]].sum(axis=axis) \
            / self.scaler['demand_cost']
        data['reward_2'] = data[[x for x in self.parameter['observation_names'] if 'co2' in x]].sum(axis=axis) \
            / self.scaler['emissions']
        
        for i in range(len(self.scaler)):
            rn = 'reward_{}'.format(i)
            data[rn] = data[rn].sum() if self.mode == 'cosim' else data[rn]
        
        reward = np.array([data['reward_0']*-1, data['reward_1']*-1, data['reward_2']*-1])
        
        if self.parameter['print_stats']:
            print({k:round(v,1) for k,v in self.fmu.simulator.timing.items()})
            
        self.state = data[self.parameter['observation_names']].values
        
        self.data = data
        self.init = False
        return ts.transition(self.state, reward=reward, discount=0.0)
        
    def _reset(self): 
        '''reset is called at the start of the episode'''
        self.state = np.array([0] * len(self.parameter['observation_names']))
        if self.fmu_loaded:
            self.fmu.terminate()
        self.fmu_loaded = False
        self.init = True
        return ts.restart(self.state)