import os
import sys
import time
import tempfile
import warnings
import numpy as np
import pandas as pd
import datetime as dtm

from pyfmi import load_fmu
from pyfmi import Master
from pyfmi.fmi_coupled import CoupledFMUModelME2

warnings.filterwarnings('ignore', message='The model,')
warnings.filterwarnings('ignore', message='`np.float` is a deprecated alias for the builtin `float`.')

def get_temp_name():
    return os.path.basename(tempfile.NamedTemporaryFile(prefix='').name)

def read_table(x, y, name, master):
    table = pd.DataFrame(index=range(x), columns=range(y))
    for xi in range(x):
        for yi in range(y):
            table.loc[xi, yi] = master.get('{}[{},{}]'.format(name, xi+1, yi+1))[0]
    table = table.set_index([0])
    table.index.name = None
    return table

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class simplecoupled(object):
    def __init__(self, config):
        self.config = config
        self.instance_id = config['instance_id']
        self.timing = {}
        
        self.fmu_kind = 'me' if self.config['master'] == 'me' else 'cs'
        self._load_fmus()
        self._setup_master()     
        self.fmu_time = self.config['start_time']
        
    def _load_fmus(self):
        
        st = time.time()
        # Load Grid FMU
        grid = load_fmu(os.path.join(self.config['dir_fmu'], self.fmu_kind, self.config['grid_fmu']),
                        log_level=self.config['log_level'], kind=self.fmu_kind)
        if self.fmu_kind == 'me':
            self.fmus = [['grid', grid]]
        else:
            self.fmus = [grid]
            self.fmu_names = ['grid']

        # Load SmartBuilding FMUs
        smartbuilding = []
        self.conn = []
        if not self.config['external_building']:
            for i in range(1, self.config['feeder_nodes']):
                if self.config['smartbuilding_fmus']:
                    smartbuilding.append(load_fmu(os.path.join(self.config['dir_fmu'], self.fmu_kind, self.config['site_fmu']),
                                                  log_level=self.config['log_level'], kind=self.fmu_kind))
                    if self.fmu_kind == 'me':
                        self.fmus.append(['site{}'.format(i), smartbuilding[i-1]])
                    else:
                        self.fmus.append(smartbuilding[-1])
                        self.fmu_names.append('site{}'.format(i))

                if self.config['building_fmus']:
                    if not self.config['eplus_emulated']:
                        smartbuilding.append(load_fmu(os.path.join(self.config['dir_fmu'], self.fmu_kind,
                                                                   self.config['building_fmus'][i-1]),
                                                      log_level=self.config['log_level'], kind=self.fmu_kind))
                        smartbuilding[-1].instantiate(name=get_temp_name())
                    else:
                        smartbuilding.append(load_fmu(os.path.join(self.config['dir_fmu'], self.fmu_kind,
                                                                   self.config['eplus_emulated']),
                                                      log_level=self.config['log_level'], kind=self.fmu_kind))
                        csvname = os.path.join(self.config['path_forecast'], 
                                               self.config['building_fmus'][i-1].replace('.fmu', '.csv'))
                        smartbuilding[-1].set('fileName', csvname)
                    if self.fmu_kind == 'me':
                        self.fmus.append(['building_load{}'.format(i), smartbuilding[i-1]])
                    else:
                        self.fmus.append(smartbuilding[-1])
                        self.fmu_names.append('building_load{}'.format(i))

            if self.config['smartbuilding_fmus']:
                print('FIXME')
#                 if self.config['building_fmus']:
#                     for i in range(1, self.config['feeder_nodes']):
#                         self.conn.append([self.fmus[self.fmu_names.index('building_load{}'.format(i))], 'P',
#                                           self.fmus[self.fmu_names.index('site{}'.format(i))], 'P_building'])
#                         self.conn.append([self.fmus[self.fmu_names.index('site{}'.format(i))], 'P',
#                                           grid, 'P_load[{}]'.format(i+1)])
#                         if self.config['use_smartinverter']:
#                             self.conn.append([self.fmus[self.fmu_names.index('site{}'.format(i))], 'P_pv',
#                                               grid, 'P_pv[{}]'.format(i+1)])
#                 else:
#                     for i in range(1, self.config['feeder_nodes']):
#                         self.conn.append([smartbuilding[i-1], 'P', grid, 'P_load[{}]'.format(i+1)])
#                         if self.config['use_smartinverter']:
#                             self.conn.append([smartbuilding[i-1], 'P_pv', grid, 'P_pv[{}]'.format(i+1)])
            else:
                if self.config['building_fmus']:
                    for i in range(1, self.config['feeder_nodes']):
                        self.conn.append([self.fmus[self.fmu_names.index('building_load{}'.format(i))], 'P',
                                          grid, 'P_building[{}]'.format(i+1)])
                else:
                    print('FIXME!!')
                    
        # Load EvTransport FMUs
        if self.config['n_ev_transportation']:
            for m in range(self.config['n_ev_transportation']):
                ev = load_fmu(os.path.join(self.config['dir_fmu'], self.fmu_kind, self.config['ev_fmu']),
                              log_level=self.config['log_level'], kind=self.fmu_kind)
                if self.fmu_kind == 'me':
                    self.fmus.append([f'ev_transport[{m+1}]', ev])
                else:
                    self.fmus.append(ev)
                    self.fmu_names.append(f'ev_transport[{m+1}]')
                for i in range(1, self.config['feeder_nodes']):
                    self.conn.append([self.fmus[self.fmu_names.index(f'ev_transport[{m+1}]')], f'P_site[{i}]',
                                      grid, f'P_charger[{m+1},{i+1}]'])                    
        self.timing['load_fmus'] = time.time() - st
            
    def _setup_master(self):
        
        st = time.time()
        # Instantiate Master
        if self.fmu_kind == 'me':
            self.master = CoupledFMUModelME2(self.fmus, self.conn)
        else:
            self.master = Master(self.fmus, self.conn)

        # Configure dynamics
        if self.config['smartbuilding_fmus']:
            for i in range(1, self.config['feeder_nodes']):
                if self.fmu_kind == 'me':
                    self.master.set('grid.Y_load[{}].T_const'.format(i+1), self.config['dynamics']/4) # s
                    self.master.set('site{}.Building.timestep'.format(i), self.config['dynamics']) # s
                    if self.config['use_smartinverter']:
                        self.master.set(f'grid.VoltVarWatt[{i+1}].QMaxInd', self.config['smartinverter_QMax'][i])
                        self.master.set(f'grid.VoltVarWatt[{i+1}].QMaxCap', self.config['smartinverter_QMax'][i])
                else:
                    self.fmus[self.fmu_names.index('grid')].set('Y_load[{}].T_const'.format(i+1), self.config['dynamics']/4) # s
                    if not self.config['building_fmus']:
                        self.fmus[self.fmu_names.index('site{}'.format(i))].set('Building.timestep', self.config['dynamics']) # s
                    if self.config['use_smartinverter']:
                        self.fmus[self.fmu_names.index('grid')].set(f'VoltVarWatt[{i+1}].QMaxInd', self.config['smartinverter_QMax'][i])
                        self.fmus[self.fmu_names.index('grid')].set(f'VoltVarWatt[{i+1}].QMaxCap', self.config['smartinverter_QMax'][i])
        else:
            for i in range(self.config['feeder_nodes_total']):
                if self.fmu_kind == 'me':
                    print('FIXME!!')
#                     self.master.set('grid.Y_load[{}].T_const'.format(i+1), self.config['dynamics']/4) # s
#                     self.master.set('site{}.Building.timestep'.format(i), self.config['dynamics']) # s
#                     if self.config['use_smartinverter']:
#                         self.master.set(f'grid.VoltVarWatt[{i+1}].QMaxInd', self.config['smartinverter_QMax'][i])
#                         self.master.set(f'grid.VoltVarWatt[{i+1}].QMaxCap', self.config['smartinverter_QMax'][i])
                else:
#                     self.fmus[self.fmu_names.index('grid')].set('Grid.Load[{}].T_const'.format(i+1), self.config['dynamics']/4) # s
                    self.fmus[self.fmu_names.index('grid')].set('T_const', self.config['dynamics']/4) # s                
                    if not self.config['building_fmus'] and not self.config['external_building']:
                        self.fmus[self.fmu_names.index('site{}'.format(i))].set('Building.timestep', self.config['dynamics']) # s
                    if self.config['use_smartinverter']:
                        self.fmus[self.fmu_names.index('grid')].set(f'Grid.VoltVarWatt[{i+1}].QMaxInd', self.config['smartinverter_QMax'][i])
                        self.fmus[self.fmu_names.index('grid')].set(f'Grid.VoltVarWatt[{i+1}].QMaxCap', self.config['smartinverter_QMax'][i])
        
        # Configure grid
        for k, v in self.config['grid_fmu_param'].items():
            self.fmus[self.fmu_names.index('grid')].set(k, v)

        # Configure weather data
        # TODO: Use PG&E distribution
        all_weather = [x for x in os.listdir(self.config['dir_weather']) if x.endswith('.mos')]
        np.random.seed(self.config['seed'])
        if self.config['weather_file']:
            selected_weather = self.config['weather_file']
        else:
            selected_weather = np.random.choice(all_weather, size=self.config['feeder_nodes_total'])
        if self.config['smartbuilding_fmus']:
            for i in range(1, self.config['feeder_nodes']):
                if self.fmu_kind == 'me':
                    self.master.models[self.master.names['site{}'.format(i)]].set('weather_file', \
                        os.path.join(self.config['dir_weather'], selected_weather[i]))
                    self.master.models[self.master.names['site{}'.format(i)]].set('building_scale', self.config['building_scale'][i])
                    self.master.models[self.master.names['site{}'.format(i)]].set('building_ft2', self.config['building_ft2'][i])
                    self.master.models[self.master.names['site{}'.format(i)]].set('der_scale', self.config['der_scale'][i])
                    self.master.models[self.master.names['site{}'.format(i)]].set('battery_scale', self.config['battery_scale'][i])
                    self.master.models[self.master.names['site{}'.format(i)]].set('charger_scale', self.config['charger_scale'][i])
                else:
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('weather_file', \
                        os.path.join(self.config['dir_weather'], selected_weather[i]))
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('building_scale', self.config['building_scale'][i])
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('building_ft2', self.config['building_ft2'][i])
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('der_scale', self.config['der_scale'][i])
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('battery_scale', self.config['battery_scale'][i])
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('charger_scale', self.config['charger_scale'][i])
        else:
            for i in range(self.config['feeder_nodes_total']):
                if self.fmu_kind == 'me':
                    print('FIXME!!')
#                     self.master.models[self.master.names['site{}'.format(i)]].set('weather_file', \
#                         os.path.join(self.config['dir_weather'], selected_weather[i]))
#                     self.master.models[self.master.names['site{}'.format(i)]].set('building_scale', self.config['building_scale'][i])
#                     self.master.models[self.master.names['site{}'.format(i)]].set('building_ft2', self.config['building_ft2'][i])
#                     self.master.models[self.master.names['site{}'.format(i)]].set('der_scale', self.config['der_scale'][i])
#                     self.master.models[self.master.names['site{}'.format(i)]].set('battery_scale', self.config['battery_scale'][i])
#                     self.master.models[self.master.names['site{}'.format(i)]].set('charger_scale', self.config['charger_scale'][i])
                else:
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].weather_file', \
                        os.path.join(self.config['dir_weather'], selected_weather[i]))
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].building_scale', self.config['building_scale'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].building_ft2', self.config['building_ft2'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].der_scale', self.config['der_scale'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].battery_scale', self.config['battery_scale'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].charger_scale', self.config['charger_scale'][i])
                
        # Configure PV size and charger reconnect
        for i in range(1, self.config['feeder_nodes_total']):
            if self.config['smartbuilding_fmus']:
                if self.fmu_kind == 'me':
                    self.master.set('site{}.ctrl_PV.k'.format(i), self.config['selected_pvs'][i])
                    self.master.set('grid.chaCtrl{}.tDelay'.format(i), self.config['charger_reconnect'][i])
                else:
                    self.fmus[self.fmu_names.index('site{}'.format(i))].set('ctrl_PV.k', self.config['selected_pvs'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'chaCtrl[{i+1}].tDelay', self.config['charger_reconnect'][i])
            else:
                if self.fmu_kind == 'me':
                    print('FIXME!!')
#                     self.master.set('site{}.ctrl_PV.k'.format(i), self.config['selected_pvs'][i])
                else:
                    self.fmus[self.fmu_names.index('grid')].set(f'Site[{i+1}].ctrl_PV.k', self.config['selected_pvs'][i])
                    self.fmus[self.fmu_names.index('grid')].set(f'Grid.chaCtrl[{i+1}].tDelay', self.config['charger_reconnect'][i])
            
        # Configure EvTransport
        if self.config['n_ev_transportation']:
            
            for m, kv in enumerate(self.config['ev_fmu_param']):
                for k, v in kv.items():
                    if self.fmu_kind == 'me':
                        print('FIXME!!')
    #                     self.master.set('site{}.ctrl_PV.k'.format(i), self.config['selected_pvs'][i])
                    else:
                        self.fmus[self.fmu_names.index(f'ev_transport[{m+1}]')].set(k, v)
            
#             for m in range(self.config['n_ev_transportation']):
#                 if self.fmu_kind == 'me':
#                     print('FIXME!!')
# #                     self.master.set('site{}.ctrl_PV.k'.format(i), self.config['selected_pvs'][i])
#                 else:
#                     self.fmus[self.fmu_names.index(f'ev_transport[{m+1}]')].set(f'fileName', self.config['ev_fmu_param'][m]['fileName'])
        # Initialize FMUs
        if self.fmu_kind == 'cs':
            for fmu in self.fmus:
                fmu.setup_experiment(start_time=self.config['start_time'], stop_time=self.config['final_time'],
                                     tolerance=self.config['tolerance'])        
                fmu.initialize()

        # Configure Storage size
        # TODO
        
        self.options = self.master.simulate_options()
        if self.fmu_kind == 'me':
            #self.options['ncp'] = 0
            self.options['ncp'] = (self.config['final_time'] - self.config['start_time']) / self.config['step_size']
            self.options['CVode_options']['verbosity'] = 50
            #options['CVode_options']['rtol'] = 5e-4 # default: 1e-4
            #options['CVode_options']['atol'] = 1e-3 #1e-4 # default: 1e-6
            self.options['result_handling'] = 'memory'
            self.options['filter'] = self.config['filter']
        else:
            self.options['step_size'] = self.config['step_size_internal']
            #self.options['result_handling'] = 'none'
            #self.options['result_handling'] = 'memory'
            #self.options['execution'] = 'parallel'
            #self.options['num_threads'] = 6
            self.options['initialize'] = False
            for k in self.options['filter'].keys():
                self.options['filter'][k] = self.config['filter']
            fmu_resname = {}
            for name in self.fmu_names:
                fmu_resname[self.fmus[self.fmu_names.index(name)]] = '{}_{}_result.mat'.format(name, self.instance_id)
            self.options['result_file_name'] = fmu_resname
        self.timing['setup_master'] = time.time() - st   
        
    def get_controlvars(self):
        ctrl = pd.DataFrame(index=np.arange(self.config['start_time'], self.config['final_time']+self.config['step_size_internal'],
                                            self.config['step_size_internal']))
        for k in self.config['control_vars'].keys():
            if '{}' in k:
                for i in range(1, self.config['feeder_nodes']):
                    ctrl[k.format(i)] = 0
            else:
                ctrl[k] = 0
        return ctrl
        
    def read_table(self, x, y, name):
        return read_table(x, y, '.'.join(name.split('.')[1:]),
                          self.fmus[self.fmu_names.index(name.split('.')[0])])
    
    def set(self, keys, values):
        for k, v in zip(keys, values):
            f = k.split('.')[0]
            n = '.'.join(k.split('.')[1:])
            self.fmus[self.fmu_names.index(f)].set(n, v)
        
    def get(self, keys):
        res = np.array([])
        for k in keys:
            f = k.split('.')[0]
            n = '.'.join(k.split('.')[1:])
            res = np.append(res, self.fmus[self.fmu_names.index(f)].get(n))
        return res
        
    def simulate(self, inputs=pd.DataFrame(), reset=True, start_time=None, final_time=None):
        if not start_time:
            start_time = self.config['start_time']
        if not final_time:
            final_time = self.config['final_time']

        st = time.time()
        if not inputs.empty:
            if self.fmu_kind == 'me':
                inputs = [inputs.columns, inputs.reset_index().values]
            else:
                inputs = ([[self.fmus[self.fmu_names.index(x.split('.')[0])], 
                            '.'.join(x.split('.')[1:])] for x in inputs.columns],
                          inputs.reset_index().values)
        else:
            if self.fmu_kind == 'me':
                inputs = {}
            else:
                inputs = None
        with HiddenPrints() if self.config['log_level']==0 else open(os.devnull, 'r'):
            self.res = self.master.simulate(input=inputs, start_time=start_time,
                                            final_time=final_time, options=self.options)
        self.timing['simulate'] = time.time() - st
        
        st = time.time()
        self._processresults()
        self._postprocess()
        self.timing['processing'] = time.time() - st
        
        # Reset master
        st = time.time()
        if reset:
            if self.config['building_fmus']:
                del self.fmus
                del self.master
                del self.conn
                self._load_fmus()
                self._setup_master()
            else:
                self.master.reset()
                self._setup_master()
        self.timing['reset'] = time.time() - st
        
        return self.data
    
    def do_step(self, inputs=pd.DataFrame(), step_size=None):
        if 'time' in inputs:
            del inputs['time']
        start_time = self.fmu_time
        if step_size:
            final_time = start_time + step_size
        else:
            final_time = start_time + self.config['step_size']
        res = self.simulate(inputs=inputs, start_time=start_time, 
                            final_time=final_time, reset=False)
        self.options['initialize'] = False
        self.fmu_time = final_time
        return res
        
    
    def _processresults(self, year=dtm.datetime.now().year):
        if self.fmu_kind == 'me':
            data = pd.DataFrame(dict(self.res))
            data['time'] = res.data_matrix[:,0]
            data.index = data['time'].values
        else:
            data = pd.DataFrame()
            for i, k in enumerate(self.fmu_names):
                t = pd.DataFrame(dict(self.res[i]))
                t.index = t['time'].values
                t.columns = ['{}.{}'.format(k, c) for c in t.columns]
                if data.empty:
                    data = t
                else:
                    data = pd.concat([data, t], axis=1)
        data['time'] = data.index
        data.index = [x.replace(microsecond=0, nanosecond=0, year=year) for x in pd.to_datetime(data.index, unit='s')]
        self.data = data
        
    def _postprocess(self):
#         self.data['P_head'] = self.data['grid.P_head_kW.y']
        self.data['P_head'] = self.data['grid.Grid.P_head']
        obj_cols = []
        for k in self.config['objective']:
            if '{}' in k:
                for i in range(1, self.config['feeder_nodes']):
                    obj_cols.append(k.format(i+1))
            else:
                obj_cols.append(k)
        self.objective = self.data[obj_cols]
    
    def get_objective(self):
        obj = {}
        obj['co2'] = self.objective['grid.Grid.co2'].resample('1H').mean().sum()
        obj['cost'] = self.objective[[f'grid.Site[{x+1}].cost' for x in range(1, \
            self.config['feeder_nodes'])]].resample('1H').mean().sum().sum()
        return obj
    
    def terminate(self):
        for name, fmu in zip(self.fmu_names, self.fmus):
            try:
                fmu.terminate()
                #fmu.free_instance()
            except:
                print('WARNING: Could not terminate fmu "{}".'.format(name))
        del self.fmus
        del self.master
        del self.conn
        