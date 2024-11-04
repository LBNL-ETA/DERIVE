import os
import sys
import itertools
import numpy as np

try:
    root = os.path.dirname(os.path.realpath(__file__))
    jupyter = False
except:
    root = os.getcwd()
    jupyter = True

sys.path.append(os.path.join(root))
if not jupyter:
    from ies_processors import ies_preprocessor, ies_postprocessor, ies_resetprocessor

def get_ies_configuration(parameter_sim, scale_inputs=False):
    
    parameter = {}
    parameter['scale_inputs'] = scale_inputs
    parameter['scaler'] = {'energy_cost': -1, 'demand_cost': -1/21.67, 'emissions': -1} # weekdays per month
    parameter['preprocessor'] = None
    parameter['postprocessor'] = None
    
    parameter['fmu_path'] = None
    parameter['fmu_loglevel'] = 0 # Loglevel, 0-none, 5-debug
    parameter['step_size'] = parameter_sim['step_size']
    parameter['start_time'] = parameter_sim['start_time']
    parameter['final_time'] = parameter_sim['final_time']
    parameter['param'] = {'step_size': parameter_sim['step_size'],
                          'start_time': parameter_sim['start_time'],
                          'final_time': parameter_sim['final_time']} # FMU parameters
    parameter['inputs'] = {} # FMU inputs
    parameter['print_stats'] = False
    parameter['fmu_param'] = {} # FMU parameters
    
    # parameter for buildings
#     parameter['input_labels'] = ['time'] # Names of inputs
    parameter['action_names'] = [] # Names of actions
    parameter['action_min'] = [] # Minimum action
    parameter['action_max'] = [] # Maximum action
    parameter['inputs_map'] = {} # Map of input colums to fmu inputs
    parameter['observation_names'] = ['sim_time', 'grid.P', 'weaBus.HGloHor', 'weaBus.HDirNor',
                                      'weaBus.HDifHor', 'weaBus.TDryBul', 'grid.tariff_energy', 'grid.tariff_demand']
    parameter['hidden_observation_names'] = ['grid.Grid.co2', 'grid.Grid.P_head',
                                             'grid.Site[2].pv.weaBus.HGloHor', 'grid.Site[2].pv.weaBus.HDirNor',
                                             'grid.Site[2].pv.weaBus.HDifHor', 'grid.Site[2].pv.weaBus.TDryBul']
    parameter['external_observations'] = {'sim_time': 0, 'grid.P': 0, 'weaBus.HGloHor': 0, 'weaBus.HDirNor': 0,
                                          'weaBus.HDifHor': 0, 'weaBus.TDryBul': 20, 'grid.tariff_energy': 0,
                                          'grid.tariff_demand': 0}
    parameter['hidden_input_names'] = []
    for i in range(1, parameter_sim['feeder_nodes']):
        t = f'grid.P_set_battery[{i+1}]'
        parameter['action_names'].append(t)
        P_batt_max = parameter_sim['battery_scale'][i] * 5/2 * parameter_sim['building_ft2'][i] \
            * parameter_sim['der_scale'][i] # from Modelica model
        if parameter['scale_inputs']:
            parameter['scaler'][t] = P_batt_max
            parameter['action_min'].append(-1 if 'P_set_battery' in t else 0)
            parameter['action_max'].append(1 if 'P_set_battery' in t else 0)  
        else:
            parameter['action_min'].append(-P_batt_max if 'P_set_battery' in t else 0)
            parameter['action_max'].append(P_batt_max if 'P_set_battery' in t else 0)         
#         parameter['input_labels'].append(t) 
        parameter['inputs_map'][t] = t
        parameter['observation_names'].append(f'grid.Site[{i+1}].P')
        parameter['observation_names'].append(f'grid.Site[{i+1}].pv.P')
        parameter['observation_names'].append(f'grid.Site[{i+1}].battery.P')
        parameter['observation_names'].append(f'grid.Site[{i+1}].battery.SOC')
        parameter['observation_names'].append(f'grid.Site[{i+1}].building.P')
        parameter['observation_names'].append(f'grid.Site[{i+1}].charger.P')
        if parameter_sim['site_fmu'] == 'SmartBuilding_external.fmu' and not parameter_sim['eplus_emulated']:
            if parameter_sim['external_building']: # With MPC control
                # P_building
                parameter['action_names'].append(f'grid.P_building[{i+1}]')
#                 parameter['input_labels'].append(f'grid.P_building[{i+1}]')
                parameter['action_min'].append(-1e9)
                parameter['action_max'].append(1e9)
                # T_in
                parameter['action_names'].append(f'grid.Site[{i+1}].building.T_in')
#                 parameter['input_labels'].append(f'grid.Site[{i+1}].building.T_in')
                parameter['hidden_input_names'].append(f'grid.Site[{i+1}].building.T_in')
                parameter['action_min'].append(0)
                parameter['action_max'].append(50)
                # T_slab
                parameter['action_names'].append(f'grid.Site[{i+1}].building.T_slab')
#                 parameter['input_labels'].append(f'grid.Site[{i+1}].building.T_slab')
                parameter['hidden_input_names'].append(f'grid.Site[{i+1}].building.T_slab')
                parameter['action_min'].append(0)
                parameter['action_max'].append(50)
            else:
                for c in ['T_set_cool', 'T_set_heat']:
                    parameter['action_names'].append('building_load{}.{}'.format(i, c)) 
                    parameter['action_min'].append(-1)
                    parameter['action_max'].append(50)
#                     parameter['input_labels'].append('building_load{}.{}'.format(i, c))
                    #parameter['observation_names'].append('building_load{}.{}_int'.format(i, c))
                parameter['fmu_param']['building_load{}.T_set_cool'.format(i)] = 24.0
                parameter['fmu_param']['building_load{}.T_set_heat'.format(i)] = 21.0            
                parameter['hidden_observation_names'] += \
                    ['building_load{}.T_{}_{}'.format(i, x[0], x[1]) for x in itertools.product(range(3), range(5))]
            # T_in
            parameter['observation_names'].append(f'grid.Site[{i+1}].building.T_in')
            parameter['external_observations'][f'grid.Site[{i+1}].building.T_in'] = 22.0
            # T_slab
            parameter['observation_names'].append(f'grid.Site[{i+1}].building.T_slab')
            parameter['external_observations'][f'grid.Site[{i+1}].building.T_slab'] = 22.0  
            
    if parameter_sim['use_ltc']:
        parameter['observation_names'].append(f'grid.Grid.LTC_ctrl1.y')
        parameter['observation_names'].append(f'grid.Grid.LTC_ctrl1.totAct')       
    
    if parameter_sim['n_ev_transportation']:
        for e in range(parameter_sim['n_ev_transportation']):
            for p in range(parameter_sim['n_cha']):
                parameter['action_names'].append(f'ev_transport[{e+1}].PPlugCtrl[{p+1}]') 
                parameter['action_min'].append(-1e6)
                parameter['action_max'].append(1e6)
                parameter['observation_names'].append(f'ev_transport[{e+1}].SOC[{p+1}]')
#                 parameter['input_labels'].append(f'ev_transport{e}.PPlugCtrl[{p+1}]')
            for p in range(parameter_sim['n_evs']):
#                 parameter['observation_names'].append(f'ev_transport[{e+1}].EV_trans.EVs[{p+1}].P')
                parameter['observation_names'].append(f'ev_transport[{e+1}].EV_trans.EVs[{p+1}].SOC')

    return parameter

def get_ies_gym_configuration(parameter_sim, warmup_time=24*60*60, forecast=True, reward_col=['reward_feeder_cost'],
                              scale_inputs=False):
    '''
    This function is used to configure the OpenAI environment wrapper.
    
    Inputs
    ------
    parameter_sim (dict): The simulation configuration dictionary.
    warmup_time (float): Warmup time of simulation. (default = 24*60*60)
    forecast (bool): Flag to return forecast columns through states. (default = True)
    reward_col (list): Name of colums to compute rward. (default = ['reward_feeder_cost'])
    scale_inputs (bool): Scale inputs to -1/1. (default = False)
    
    Returns
    -------
    config (dict): The configuration dictionary.
    '''
    parameter_tf = get_ies_configuration(parameter_sim, scale_inputs=scale_inputs)
    
    parameter = {}
    # fmi_gym parameter
    parameter['seed'] = 1
    parameter['store_data'] = True

    # fmu parameter
    dtype = 'float'
    parameter['fmu_step_size'] = parameter_sim['step_size']
    parameter['fmu_path'] = True
    parameter['fmu_start_time'] = parameter_sim['start_time']
    parameter['fmu_warmup_time'] = parameter_sim['start_time'] + warmup_time
    parameter['fmu_final_time'] = parameter_sim['final_time']
    parameter['fmu_kind'] = parameter_sim
    parameter['fmu_param'] = parameter_tf['fmu_param']

    # data exchange parameter
#     parameter['input_labels'] = parameter_tf['input_labels'][1:]
    parameter['action_names'] = parameter_tf['action_names']
    parameter['action_min'] = np.array(parameter_tf['action_min'], dtype=dtype)
    parameter['action_max'] = np.array(parameter_tf['action_max'], dtype=dtype)
    parameter['observation_names'] = parameter_tf['observation_names']

    # ies_preprocessor parameter
    parameter['preprocessor'] = ies_preprocessor
    parameter['site_fmu'] = parameter_sim['site_fmu']
    parameter['feeder_nodes'] = parameter_sim['feeder_nodes']
    parameter['hidden_observation_names'] = parameter_tf['hidden_observation_names']
    parameter['external_observations'] = parameter_tf['external_observations']
    parameter['hidden_input_names'] = parameter_tf['hidden_input_names']
    
    # ies_postprocessor parameter
    parameter['postprocessor'] = ies_postprocessor
    parameter['tariff'] = 'e19-2020' # 'e19-2020' or 'tou8-2020'
    parameter['year'] = 2021
    parameter['scale_inputs'] = scale_inputs
    parameter['scaler'] = parameter_tf['scaler']
    parameter['reward_col'] = reward_col
    parameter['emissions_file'] = os.path.join(root, '..', 'resources', 'emissions', 'annual_emissions_20210601.csv')
    parameter['path_forecast'] = os.path.join(root, '..', 'resources', 'forecast')
    parameter['forecast_hours'] = 24
    parameter['add_forecast'] = forecast
    if parameter['add_forecast']:
        # Weather
        for c in ['TDryBul', 'HDifHor', 'HDirNor', 'HGloHor']:
            for h in range(24):
                n = f'weaBus.forecast.{c}_{str(h).zfill(2)}'
                parameter['observation_names'].append(n)
                parameter['external_observations'][n] = 0.0
        # Grid
        for c in ['ECo2pW']:
            for h in range(24):
                n = f'grid.forecast.{c}_{str(h).zfill(2)}'
                parameter['observation_names'].append(n)
                parameter['external_observations'][n] = 0.0
        # Building
        for i in range(1, parameter_sim['feeder_nodes']):
            for c in ['P-building', 'P-light', 'Q-people', 'P-equip', 'T-cool', 'T-heat']:
                for h in range(24):
                    n = f'grid.Site[{i+1}].building.forecast.{c}_{str(h).zfill(2)}'
                    parameter['observation_names'].append(n)
                    parameter['external_observations'][n] = 0.0
            for c in ['P-pv']:
                for h in range(24):
                    n = f'grid.Site[{i+1}].pv.forecast.{c}_{str(h).zfill(2)}'
                    parameter['observation_names'].append(n)
                    parameter['external_observations'][n] = 0.0
        # EVs
        if parameter_sim['n_ev_transportation']:
            for t in range(parameter_sim['n_ev_transportation']):
                for i in range(parameter_sim['n_evs']):
                    for c in ['Loc', 'P-drive']:
                        for h in range(24):
                            n = f'ev_transport[{t+1}].forecast.ev[{i+1:04d}].{c}_{str(h).zfill(2)}'
                            parameter['observation_names'].append(n)
                            parameter['external_observations'][n] = 0.0            
        parameter['observation_names'].sort()
    
    # ies_resetprocessor
    parameter['resetprocessor'] = ies_resetprocessor
    parameter['cleanup_energyplus'] = True

    return parameter