import os
import re
import sys
import copy
import time
import json
import numpy as np
import pandas as pd

try:
    root = os.path.dirname(os.path.realpath(__file__))
except:
    root = os.getcwd()
    
# sys.path.append(os.path.join(root, '..', 'resources', 'controller'))
# from rctuning import read_and_convert_ies, do_tuning, new_param_to_mpc

sys_remove = []
for p in sys.path:
    if 'Christoph' in p:
        sys_remove.append(p)
for p in sys_remove:
    sys.path.remove(p)
    
# DOPER
#sys.path.append(os.path.join(root, '..', '..', 'ESTCP-Controller', 'doper_private', 'src'))
from doper import DOPER, get_solver, get_root, standard_report
from doper.models.basemodel import base_model, default_output_list
from doper.models.battery import add_battery, convert_battery
from doper.models.genset import add_genset
from doper.models.loadControl import add_loadControl
from doper.examples import parameter_add_battery, default_parameter
from doper.computetariff import compute_periods
# sys.path.append(os.path.join(root, '..', '..', 'ESTCP-Controller', 'doper_private', 'examples', 'data'))
#from tariff import get_e19_2020_tariff, get_tou8_2020_tariff
from doper.data.tariff import get_tariff
#sys.path.append('/home/Christoph/Documents/PrivateRepos/ies/resources/controller')
sys.path.append(os.path.join(root, '..', 'resources', 'controller'))
from model import add_hvac
from DefaultConfiguration import example_parameter_add_hvac
#from rctuning import read_and_convert_ies, do_tuning, new_param_to_mpc
    
from pyomo.environ import Objective, minimize

def make_heur_table(vals=[60e3, -50e3, -25e3], horizon=24, cha_vals=[], cha_names=[],
                    ev_times=[8, 8, 16, 16]):
    heur = pd.DataFrame(index=range(horizon))
    heur['P_set_battery'] = 0
    heur['T_set_cool'] = 0
    heur['T_set_heat'] = 0
    heur.loc[:6, 'P_set_battery'] = vals[0]
    heur.loc[11:17, 'P_set_battery'] = vals[1]
    heur.loc[17:, 'P_set_battery'] = vals[2]
    if cha_names:
        for n in cha_names:
            heur[n] = 0
            heur.loc[:ev_times[0], n] = cha_vals[0]
            heur.loc[ev_times[1]:ev_times[2], n] = cha_vals[1]
            heur.loc[ev_times[3]:, n] = cha_vals[2]            
    return heur
    
# Heuristic control lookup
parameter_heur_combined = {'lookup': make_heur_table([60e3, -50e3, -25e3]),
                           'min_soc_bidir': {},
                           'individual_ctrl_adj': {}}
parameter_heur_individual = {'lookup': [make_heur_table([60e3, -50e3, -25e3]),
                                        make_heur_table([50e3, -40e3, -25e3])],
                             'min_soc_bidir': {},
                             'individual_ctrl_adj': {}}

class heuristic_agent(object):
    def __init__(self, parameter_gym, parameter_ctrl=parameter_heur_combined, env=None):
        self.parameter_gym = parameter_gym
        self.parameter_ctrl = parameter_ctrl
        self.env = env
        self.lookup = parameter_ctrl['lookup']

    def forward(self, state, external_control={}):
        state = pd.Series(state, index=self.parameter_gym['observation_names'])
        ts = pd.to_datetime(state['sim_time'], origin=str(self.parameter_gym['year']), unit="s")
        #hour = max(ts.hour+1, max(self.lookup.index))
        hour = ts.hour
        actions = []
        ev_lowsoc_power = self.parameter_ctrl['ev_lowsoc_power'] if 'ev_lowsoc_power' in self.parameter_ctrl else 0
        for a in self.parameter_gym['action_names']:
            if a in external_control.keys():
                actions.append(external_control[a])
            else:
                cf = 0
                if type(self.lookup) == type([]):
    #                 i = int(''.join(filter(str.isdigit, a.split('.')[0]))) - 1
                    if 'P_set_battery' in a:
                        i = int(a.split('P_set_battery[')[1].split(']')[0]) - 2
                    elif 'T_set' in a:
                        i = int(a.split('building_load')[1].split('.')[0]) - 1
                    else:
                        print(f'ERROR: Can not identify "{a}".')
                    lookup = self.lookup[i]
                else:
                    lookup = self.lookup
                for c in lookup.columns:
                    if c in a:
                        if c in self.parameter_ctrl['min_soc_bidir'].keys():
                            # check SOC
                            min_soc_bidir = self.parameter_ctrl['min_soc_bidir'][c]
                            if 0 < state[min_soc_bidir[0]] < min_soc_bidir[2]:
                                # deep discharged => charge
                                actions.append(max(ev_lowsoc_power, lookup.loc[hour, c]))
                            elif 0 < state[min_soc_bidir[0]] < min_soc_bidir[1]:
                                # discharged => stop discharging
                                actions.append(max(0, lookup.loc[hour, c]))
                            else:
                                scale = 1
                                if c in self.parameter_ctrl['individual_ctrl_adj']:
                                    scale = self.parameter_ctrl['individual_ctrl_adj'][c]
                                # operate
                                actions.append(lookup.loc[hour, c] * scale)
                        else:
                            actions.append(lookup.loc[hour, c])
                        cf += 1
                if cf == 0:
                    print('WARNING: Could not find "{}" in lookup table. Defaulting to "0".'.format(a))
                    actions.append(0)
                if cf > 1:
                    print('WARNING: Could not find multiple "{}" in lookup table. Actions might be too long.'.format(a))
        return np.array(actions)
    
class zero_agent(object):
    def __init__(self, parameter_gym, parameter_ctrl={}, env=None):
        self.parameter_gym = parameter_gym
        self.parameter_ctrl = parameter_ctrl
        self.env = env

    def forward(self, state, external_control={}):
        return np.array([0] * len(self.parameter_gym['action_names']))
    
class random_agent(object):
    def __init__(self, parameter_gym, parameter_ctrl={}, env=None):
        self.parameter_gym = parameter_gym
        self.env = env
        self.i = 0

    def forward(self, state, external_control={}):
        self.env.action_space.seed(self.parameter_gym['seed']+self.i)
        self.i += 1
        return self.env.action_space.sample()
    
ies_battery_parameter = {'soc_min': 0.1, 'soc_max': 1.0, 'capacity': 5*50, 'power': 5/2*50}
    
def get_mpc_local_parameter(rc_model=False, parameter_gym={}, ext_batt_par=ies_battery_parameter):
    mpc_par = default_parameter()
    mpc_par = parameter_add_battery(mpc_par)
    mpc_par['controller']['solver'] = 'cbc'
    mpc_par['objective'] = {'weight_energy': 22, 'weight_demand': 1, 'weight_ghg': 0,
                            'weight_export': 0, 'weight_regulation': 0, 'weight_degradation': 0}
    mpc_par['system'] = {'pv': True, 'battery': True, 'genset': False, 'load_control': False,
                         'reg_bidding': False, 'reg_response': False}
    mpc_par['site']['export_max'] = 1e9
    mpc_par['site']['import_max'] = 1e9
    mpc_par['outputs'] = default_output_list(mpc_par)
    mpc_batts = []
    mpc_loads = []
    if not 'feeder_nodes_start' in parameter_gym.keys():
        parameter_gym['feeder_nodes_start'] = 1
    for ii, i in enumerate(range(parameter_gym['feeder_nodes_start'], parameter_gym['feeder_nodes'])):
        batt_par = mpc_par['batteries'][0].copy()
        batt_par['name'] = str(ii)
        batt_par['soc_min'] = ext_batt_par['soc_min']
        batt_par['soc_max'] = ext_batt_par['soc_max']
        #batt_par['capacity'] = ext_batt_par['capacity'] # kWh
        #batt_par['power_charge'] = ext_batt_par['power'] # kW
        battery_scale = parameter_gym['fmu_kind']['battery_scale'][i]
        building_ft2 = parameter_gym['fmu_kind']['building_ft2'][i]
        der_scale = parameter_gym['fmu_kind']['der_scale'][i]
        capacity = battery_scale*5*building_ft2*der_scale
        #print(i, battery_scale, building_ft2, der_scale, capacity)
        batt_par['capacity'] = capacity / 1e3 # kWh
        power = battery_scale*5/2*building_ft2*der_scale
        batt_par['power_charge'] = power / 1e3 # kW
        batt_par['power_discharge'] = batt_par['power_charge'] 
        mpc_batts.append(batt_par)
        #mpc_loads.append({'cost': 1.0, 'name': '{}-{}'.format(i, 'high'), 'outageOnly': True, 'sheddable': False})
        #mpc_loads.append({'cost': 0.5, 'name': '{}-{}'.format(i, 'low'), 'outageOnly': True, 'sheddable': False})
    mpc_par['outputs'].append({'name': 'batCharge_individual', 'data': 'battery_charge_grid_power',
                               'index': 'batteries', 'df_label': 'Battery Charging Power %s [kW]'})
    mpc_par['outputs'].append({'name': 'batDischarge_individual', 'data': 'battery_discharge_grid_power',
                               'index': 'batteries', 'df_label': 'Battery Discharging Power %s [kW]'})
    mpc_par['outputs'].append({'name': 'ghg_intensity', 'data': 'grid_co2_intensity',
                               'df_label': 'GHG Intensity [co2pW]'})
        
    mpc_par['batteries'] = mpc_batts
    #mpc_par['load_control'] = mpc_loads
    mpc_par['single_day'] = True
    #print('NOT single day!')
    #mpc_par['single_day'] = False
    
    mpc_par['retune_rc'] = False
    mpc_par['read_rc'] = True
    mpc_par['rc_table'] = os.path.join(root, '..', 'resources', 'controller', 'rc_table.csv')
    if parameter_gym['hvac_control']:
        mpc_par['outputs'].append({'name': 'outside_temperature', 'data': 'outside_temperature',
                    'df_label': 'Outdoor Temperature [C]'})
        mpc_par['outputs'].append({'name': 'building_load_dynamic', 'data': 'building_load_dynamic',
                                   'index': 'nodes', 'df_label': 'Building&HVAC load [kW]'})
        mpc_par['outputs'].append({'name': 'p_zones_tot', 'data': 'p_zones_tot',
                        'df_label': 'Building&HVAC load internal [W]'})
        # mpc_par['outputs'].append({'name': 'zone_occload', 'data': 'zone_occload',
        #                 'df_label': 'Occupant load internal [kW]'})

        # Per zone
        mpc_par['outputs'].append({'name': 'p_cooling', 'data': 'p_cooling',
                        'index': 'zones','df_label': 'Cooling Electric Power %s [W]'})
        mpc_par['outputs'].append({'name': 'p_heating', 'data': 'p_heating',
                        'index': 'zones','df_label': 'Heating Electric Power %s [W]'})
        mpc_par['outputs'].append({'name': 'zone_air_temp', 'data': 'zone_air_temp',
                        'index': 'zones', 'df_label': 'Zone Air Temperature %s [C]'})
        mpc_par['outputs'].append({'name': 'zone_slab_temp', 'data': 'zone_slab_temp',
                        'index': 'zones', 'df_label': 'Zone Slab Temperature %s [C]'})
        mpc_par['outputs'].append({'name': 'zone_temp_max', 'data': 'zone_temp_max',
                        'index': 'zones', 'df_label': 'Zone Temperature Max %s [C]'})
        mpc_par['outputs'].append({'name': 'zone_temp_min', 'data': 'zone_temp_min',
                        'index': 'zones', 'df_label': 'Zone Temperature Min %s [C]'})
        mpc_par['outputs'].append({'name': 'p_zones', 'data': 'p_zones',
                        'index': 'zones', 'df_label': 'Building&HVAC load %s [W]'})
        n_zones = int(parameter_gym['feeder_nodes'] - parameter_gym['feeder_nodes_start'])
        mpc_par = example_parameter_add_hvac(mpc_par, n_zones=n_zones)
        mpc_par['system']['hvac_control'] = True
        mpc_par['zone']['n_zones'] = parameter_gym['feeder_nodes'] - parameter_gym['feeder_nodes_start']
        
        year = parameter_gym['year']
        start_time = pd.to_datetime(parameter_gym['fmu_warmup_time'], unit='s', origin=year)
        final_time = pd.to_datetime(parameter_gym['fmu_final_time'], unit='s', origin=year)
        
        if mpc_par['retune_rc']:
            for i, n in enumerate(range(parameter_gym['feeder_nodes_start'], parameter_gym['feeder_nodes'])):
                filepath = os.path.join(parameter_gym['fmu_kind']['path_forecast'], 
                                        parameter_gym['fmu_kind']['building_fmus'][n])
                filepath = filepath.replace('.fmu', '.csv')
                mpc_rc_parm, rctuner, res = do_tuning(filepath, start_time, final_time, year=year)
                #duration, objective, df, new_param, inputs, res_lhs, rctype = res
                for k, v in mpc_rc_parm.items():
                    mpc_par['zone'][k][i] = v
                #print(i, n, mpc_rc_parm)
        elif mpc_par['read_rc']:
            rcs = pd.read_csv(mpc_par['rc_table'], index_col=[0])
            for i, n in enumerate(range(parameter_gym['feeder_nodes_start'], parameter_gym['feeder_nodes'])):
                f = parameter_gym['fmu_kind']['building_fmus'][n-1]
                building_type = f.split('_')[1]
                building_id = int(f.split('_')[-1].replace('.fmu', ''))
                season = parameter_gym['fmu_kind']['run_period']
                mpc_rc_parm = rcs[(rcs['id']==building_id) & (rcs['type']==building_type) & (rcs['period']==season)].iloc[0]
                mpc_rc_parm = new_param_to_mpc(mpc_rc_parm)
                for k, v in mpc_rc_parm.items():
                    mpc_par['zone'][k][i] = v
                #print(i, n, mpc_rc_parm)
        else:
            print('**** NO RC tuning implemented ****')
    else:
        mpc_par['system']['hvac_control'] = False
            
    
    return mpc_par
    
def control_model(inputs, parameter):
    model = base_model(inputs, parameter)
    model = add_battery(model, inputs, parameter)
    #model = add_genset(model, inputs, parameter)
    #model = add_loadControl(model, inputs, parameter)
    if parameter['system']['hvac_control']:
        model = add_hvac(model, inputs, parameter)

    # Set battery to initial soc when 0
    if parameter['soc_midnight'] != None:
        ix_0 = [i for i, ix in enumerate(inputs.index) if ix.hour == 0][0] + 1
        for ib, b in enumerate(model.batteries):
            soc_init = parameter['soc_midnight'][ib]
            model.battery_energy[model.ts.at(ix_0), b] = model.bat_capacity[b] * soc_init
            model.battery_energy[model.ts.at(ix_0), b].fixed = True
            # Free last state
            model.battery_energy[model.ts.at(len(model.ts)), b].fixed = False
    
    def objective_function(model):
        return model.sum_energy_cost * parameter['objective']['weight_energy'] \
               + model.sum_demand_cost * parameter['objective']['weight_demand'] \
               + model.co2_total * parameter['objective']['weight_ghg']
    model.objective = Objective(rule=objective_function, sense=minimize, doc='objective function')
    return model

class mpc_handler(object):
    def __init__(self, parameter_gym, mpc_par, env):
        self.parameter_gym = parameter_gym
        self.mpc_par = mpc_par
        self.env = env
        self.tariff = get_tariff(parameter_gym['tariff'])
        self.init_mpc()
        self.init = True
        
        if not 'feeder_nodes_start' in self.parameter_gym.keys():
            self.parameter_gym['feeder_nodes_start'] = 1
        
    def init_mpc(self):
        # GHG optimization
        if len(self.parameter_gym['reward_col']) == 1 and 'co2' in self.parameter_gym['reward_col'][0]:
            self.mpc_par['objective'] = {'weight_energy': 0, 'weight_demand': 0, 'weight_ghg': 1}
        elif len(self.parameter_gym['reward_col']) > 1:
            print('WARNING: Multiple objectives; MPC assuming cost + GHG.')
            self.mpc_par['objective'] = {'weight_energy': 22, 'weight_demand': 1, 'weight_ghg': 22*2}
        
#         self.solver_path = get_solver(self.mpc_par['controller']['solver'], 
#                                       solver_dir=os.path.join(get_root(), self.mpc_par['controller']['solver_dir']))
        # Use old CBC (v2.10.5) as newer ones can't solve/CBC issue?
        self.solver_path = '/home/Christoph/Documents/PrivateRepos/ESTCP-Controller/doper_private/src/DOPER/solvers/Linux64/cbc'
        self.ctrl = DOPER(model=control_model,
                          parameter=self.mpc_par,
                          solver_path=self.solver_path,
                          solver_name='cbc',
                          output_list=self.mpc_par['outputs'])
        
    def build_inputs(self, state):
        state = pd.Series(state, index=self.parameter_gym['observation_names'])
        #print(sorted(state.index))
        
        start_time = pd.to_datetime(state['sim_time'], unit='s',
                                    origin=str(self.parameter_gym['year']))
        data = pd.DataFrame(index=[start_time+pd.DateOffset(hours=h) for h in range(24)])
        # Static
        data['grid_available'] = 1
        data['battery_reg'] = 0

        # Dynamic
        data['oat'] = state[[ix for ix in state.index if ix.startswith('weaBus.forecast.TDryBul')]].sort_index().values
        data['grid_co2_intensity'] = state[[ix for ix in state.index if ix.startswith('grid.forecast.ECo2pW')]].sort_index().values
        data['generation_pv'] = 0
        data['load_demand'] = 0
        for n in range(self.parameter_gym['feeder_nodes_start'], self.parameter_gym['feeder_nodes']):
            pv = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].pv.forecast.P-pv')]].sort_index()
            data['generation_pv'] += pv.values / 1e3 # kW
            load = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.P-building')]].sort_index()
            #data[f'load_circuit_{n}'] = load.values / 1e3 # kW
            #load_scale = self.parameter_gym['fmu_kind']['building_scale'][n]
            load_scale = 1
            data['load_demand'] += load.values / 1e3 * load_scale # kW
            
        
            
        if self.parameter_gym['hvac_control']:
            data['load_demand'] = 0
            #ep = rctuner.inputs
            #ep_to_data_map = {'outside_temperature': 'oat', 'T_set_cool': 'temp_max_0', 'T_set_heat': 'temp_min_0',
            #                  'Plight': 'lights_0', 'Pequip': 'plug_load_0', 'Qpeople': 'occupant_load_0', 'zone_qi-sol': 'shg_0'}
            #data.loc[:, ep_to_data_map.values()] = ep.loc[ep_index, ep_to_data_map.keys()].values
            # For RC
            for z, n in enumerate(range(self.parameter_gym['feeder_nodes_start'], self.parameter_gym['feeder_nodes'])):
                
                #load_scale = self.parameter_gym['fmu_kind']['building_scale'][z]
                cool = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.T-cool')]].sort_index()
                data[f'temp_max_{z}'] = cool.values
                heat = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.T-heat')]].sort_index()
                data[f'temp_min_{z}'] = heat.values

                lights = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.P-light')]].sort_index()
                data[f'lights_{z}'] = lights.values #* load_scale
                plug = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.P-equip')]].sort_index()
                data[f'plug_load_{z}'] = plug.values #* load_scale
                occ = state[[ix for ix in state.index if ix.startswith(f'grid.Site[{n+1}].building.forecast.Q-people')]].sort_index()
                data[f'occupant_load_{z}'] = occ.values #* load_scale
                ghi = state[[ix for ix in state.index if ix.startswith(f'weaBus.forecast.HGloHor')]].sort_index()
                gain_ghi = self.mpc_par['zone']['shg_gain'][z]
                data[f'shg_{z}'] = ghi.values * gain_ghi #* load_scale
            
        if self.mpc_par['single_day']:
            if self.init:
                self.data_init = data.copy(deep=True)
            data = self.data_init.loc[start_time:].append(self.data_init.loc[:start_time].iloc[:-1])
            
            # Append second day
            data = data.append(self.data_init.loc[start_time:].append(self.data_init.loc[:start_time].iloc[:-1]))
            data.index = [start_time+pd.DateOffset(hours=h) for h in range(len(data))]

        # Tariff
        data, _ = compute_periods(data, self.tariff, self.mpc_par)
        self.mpc_par['tariff']['export'] = {0: 0.0}
        self.data = data
        return data
    
    def update_states(self, state):
        state = pd.DataFrame(state, index=self.parameter_gym['observation_names'])
        if self.parameter_gym['hvac_control']:
            self.mpc_par['zone']['temps_initial'] = []
        states_initialized = True
        for ns, s in enumerate(range(self.parameter_gym['feeder_nodes_start'], self.parameter_gym['feeder_nodes'])):
            self.mpc_par['batteries'][ns]['soc_initial'] = round(float(state.loc[f'grid.Site[{s+1}].battery.SOC'].values[0]), 4)
            if self.mpc_par['batteries'][ns]['soc_initial'] > self.mpc_par['batteries'][ns]['soc_max']:
                self.mpc_par['batteries'][ns]['soc_initial'] = self.mpc_par['batteries'][ns]['soc_max']
            elif self.mpc_par['batteries'][ns]['soc_initial'] < self.mpc_par['batteries'][ns]['soc_min']:
                self.mpc_par['batteries'][ns]['soc_initial'] = self.mpc_par['batteries'][ns]['soc_min']
            if self.parameter_gym['hvac_control']:
                t_in = float(state.loc[f'grid.Site[{s+1}].building.T_in'].values[0])
                if np.isnan(t_in) or t_in < 5:
                    t_in = 22.5
                    states_initialized = False
                t_slab = float(state.loc[f'grid.Site[{s+1}].building.T_slab'].values[0])
                if np.isnan(t_slab) or t_slab < 5:
                    t_slab = 22.5
                    states_initialized = False
                self.mpc_par['zone']['temps_initial'].append([t_in, t_slab])
        #print(self.mpc_par['zone']['temps_initial'])
        return states_initialized
    
    def init_states(self, iterations=5):
        #print('INFO: Initializing thermal states through MPC.')
        for i in range(iterations):
            df = self.ctrl.do_optimization(self.input_data, parameter=self.mpc_par)[2]
            self.mpc_par['zone']['temps_initial'] = []
            for ns, s in enumerate(range(self.parameter_gym['feeder_nodes_start'], self.parameter_gym['feeder_nodes'])):
                t_in = df[df.index.hour == df.index[0].hour][f'Zone Air Temperature {ns} [C]'].values[-1]
                t_slab = df[df.index.hour == df.index[0].hour][f'Zone Slab Temperature {ns} [C]'].values[-1]
                #print(i, [t_in, t_slab])
                self.mpc_par['zone']['temps_initial'].append([t_in, t_slab])
        
    def forward(self, state, external_control={}):
        st = time.time()
        
        self.input_data = self.build_inputs(state)
        states_initialized = self.update_states(state)
        # Set to initial at starttime
        if self.init:
            self.soc_midnight = {}
            for bn, b in enumerate(self.mpc_par['batteries']):
                self.soc_midnight[bn] = round(b['soc_initial'], 4)
        if self.mpc_par['single_day']:
            self.mpc_par['soc_midnight'] = self.soc_midnight
        else:
            self.mpc_par['soc_midnight'] = None
        
        if not states_initialized and self.parameter_gym['hvac_control']:
            self.init_states()
        

        self.res = self.ctrl.do_optimization(self.input_data, parameter=self.mpc_par)
        duration, self.objective, self.df, self.model, self.result, self.termination, _ = self.res
        
        if self.df.empty:
            temp_id = str(np.random.randint(1e6))
            print(f'ERROR: Not optimal. Log id: {temp_id}')
            self.input_data.to_csv(f'{temp_id}.csv')
            with open(f'{temp_id}.json', 'w') as f:
                f.write(json.dumps(self.mpc_par))
        
        # Action
        action = pd.Series(dtype=np.float64)
        #t = self.model.ts[1]
        ix = self.df.index[0]
        for a in self.parameter_gym['action_names']:
            if 'P_set_battery' in a:
                #b = int(a.split('ite')[1].split('.')[0].replace('[','').replace(']','')) - 1
                b = int(a.split('[')[1].replace(']','')) - 2
                #if 'Site[' in a:
                #    b += 1
                
                action.loc[a] = (self.df.loc[ix, f'Battery Charging Power {b} [kW]'] - \
                                 self.df.loc[ix, f'Battery Discharging Power {b} [kW]']) * 1e3 # W
                
                #action.loc[a] = (self.model.battery_charge_grid_power[t,b].value - \
                #                self.model.battery_discharge_grid_power[t,b].value) * 1e3 # W
            elif 'T_set_' in a:
                if self.parameter_gym['hvac_control']:
                    b = int(a.split('building_load')[1].split('.')[0]) - 2
                    b = b if b > 0 else 0
                    t = self.df.loc[self.df.index[1], f'Zone Air Temperature {b} [C]']
                    hys = 0.01
                    if 'T_set_cool' in a and self.df.loc[ix, f'Cooling Electric Power {b} [W]'] > 0 and \
                        t + hys < self.df.loc[self.df.index[1], f'Zone Temperature Max {b} [C]']:
                        action.loc[a] = t + hys
                    elif 'T_set_heat' in a and self.df.loc[ix, f'Heating Electric Power {b} [W]'] > 0 and \
                        t - hys > self.df.loc[self.df.index[1], f'Zone Temperature Min {b} [C]']:
                        action.loc[a] = t - hys
                    else:
                        action.loc[a] = 0
                else:
                    action.loc[a] = 0
            elif 'P_building' in a and self.parameter_gym['hvac_control']:
                b = int(a.split('P_building[')[1].split(']')[0]) - 2
                sb = b + self.parameter_gym['feeder_nodes_start']
                scale_building = self.parameter_gym['fmu_kind']['building_scale'][sb]
                scale_building *= self.parameter_gym['fmu_kind']['building_ft2'][sb] / 50e3
                action.loc[a] = self.df.loc[self.df.index[0], f'Building&HVAC load {b} [W]'] / scale_building
            elif 'building.T_in' in a and self.parameter_gym['hvac_control']:
                b = int(a.split('Site[')[1].split(']')[0]) - 2
                action.loc[a] = self.df.loc[self.df.index[1], f'Zone Air Temperature {b} [C]']
            elif 'building.T_slab' in a and self.parameter_gym['hvac_control']:
                b = int(a.split('Site[')[1].split(']')[0]) - 2
                action.loc[a] = self.df.loc[self.df.index[1], f'Zone Slab Temperature {b} [C]']
            elif a in external_control.keys():
                action.loc[a] = external_control[a]
            else:
                print(f'WARNING: Action "{a}" not in MPC result, set to 0.')
                action.loc[a] = 0
                
        self.duration = {'control': time.time()-st, 'mpc': duration}
        
        self.init = False
        return action.values
    
class mpc_individual_handler(object):
    def __init__(self, parameter_gym, mpc_par, env):
        self.parameter_gym = parameter_gym
        self.parameter_gym['hvac_control'] = mpc_par['hvac_control']
        self.mpc_par = mpc_par
        self.env = env
        
        if parameter_gym['fmu_kind']['n_ev_transportation'] != None:
            raise ValueError('ERROR: Individual MPC only works without EVs.')
        
        # handlers
        self.mpc_handlers = []
        actions_per_node = int(len(self.parameter_gym['action_names']) / (self.parameter_gym['feeder_nodes']-1))
        for b in range(1, self.parameter_gym['feeder_nodes']):
            par = copy.deepcopy(self.parameter_gym)
            par['feeder_nodes_start'] = b
            par['feeder_nodes'] = b + 1
            par['action_names'] = self.parameter_gym['action_names'][(b-1)*actions_per_node:(b)*actions_per_node]
            par['action_names'] = [re.sub(r'\d+', '2', x) for x in par['action_names']]
            new_mpc_par = get_mpc_local_parameter(parameter_gym=par)
            for k,v in mpc_par.items():
                new_mpc_par[k] = v
            self.mpc_handlers.append(mpc_handler(par, new_mpc_par, env))
        
    def forward(self, state, external_control={}):
        if external_control != {}:
            print('ERROR: The "external_control" is not implemented for individual MPC.')
        actions = [ctrl.forward(state, external_control=external_control) for ctrl in self.mpc_handlers]
        actions = np.array(actions).flatten()
        self.df = [ctrl.df for ctrl in self.mpc_handlers]
        return actions
    
def get_control(name='heuristic_combined', parameter_gym={}):
    if name == 'heuristic-combined':
        return heuristic_agent, parameter_heur_combined
    elif name == 'heuristic-individual':
        return heuristic_agent, parameter_heur_individual
    elif name == 'zero':
        return zero_agent, {}
    elif name == 'random':
        return random_agent, {}
    elif name == 'mpc-combined_norc':
        parameter_gym['hvac_control'] = False
        return mpc_handler, get_mpc_local_parameter(parameter_gym=parameter_gym)
    elif name == 'mpc-individual_norc':
        return mpc_individual_handler, {'hvac_control': False}
    elif name == 'mpc-combined_rc':
        parameter_gym['hvac_control'] = True
        return mpc_handler, get_mpc_local_parameter(parameter_gym=parameter_gym)
    elif name == 'mpc-individual_rc':
        return mpc_individual_handler, {'hvac_control': True}
    else:
        print('ERROR: Controller "{}" not defined.'.format(name))