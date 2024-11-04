import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
from io import StringIO
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)

try:
    root = os.path.dirname(os.path.realpath(__file__))
except:
    root = os.getcwd()

sys.path.append(os.path.join(root, '..', 'resources', 'external'))
from tariff import get_tariff

def compute_default_hvac(hour_start=0):
    default_hvac = pd.DataFrame(index=range(24))
    default_hvac['cool'] = 26.7
    default_hvac['heat'] = 15.6
    default_hvac.loc[6:21, 'cool'] = 24.0
    default_hvac.loc[6:21, 'heat'] = 21.0
    default_hvac = pd.concat([default_hvac.loc[hour_start:], default_hvac.loc[:hour_start-1]])
    return default_hvac

class ies_preprocessor(object):
    
    def __init__(self, parameter):
        self.parameter = parameter
        
        self.default_hvac = compute_default_hvac()
    
    def do_calc(self, data, init):
        ts = data.index[0]
        hour = pd.to_datetime(data['time'].iloc[0], unit='s').hour
        # Condition temperature control
        hys_hvac = 0.5 # K
        if self.parameter['site_fmu'] == 'SmartBuilding_external.fmu' and \
            not self.parameter['fmu_kind']['external_building'] and not self.parameter['fmu_kind']['eplus_emulated']:
            for i in range(1, self.parameter['feeder_nodes']):                
                if data.loc[ts, 'building_load{}.T_set_cool'.format(i)] <= 0 or \
                    data.loc[ts, 'building_load{}.T_set_cool'.format(i)] > self.default_hvac.loc[hour, 'cool'] or \
                    data.loc[ts, 'building_load{}.T_set_cool'.format(i)] < self.default_hvac.loc[hour, 'heat']:
                    data.loc[ts, 'building_load{}.T_set_cool'.format(i)] = self.default_hvac.loc[hour, 'cool']
                if data.loc[ts, 'building_load{}.T_set_heat'.format(i)] <= 0 or \
                    data.loc[ts, 'building_load{}.T_set_heat'.format(i)] < self.default_hvac.loc[hour, 'heat'] or \
                    data.loc[ts, 'building_load{}.T_set_heat'.format(i)] > self.default_hvac.loc[hour, 'cool']:
                    data.loc[ts, 'building_load{}.T_set_heat'.format(i)] = self.default_hvac.loc[hour, 'heat']
                if data.loc[ts, 'building_load{}.T_set_cool'.format(i)] < data.loc[ts, 'building_load{}.T_set_heat'.format(i)] + hys_hvac:
                    data.loc[ts, 'building_load{}.T_set_cool'.format(i)] = data.loc[ts, 'building_load{}.T_set_heat'.format(i)] + hys_hvac
                if data.loc[ts, 'building_load{}.T_set_heat'.format(i)] > data.loc[ts, 'building_load{}.T_set_cool'.format(i)] - hys_hvac:
                    data.loc[ts, 'building_load{}.T_set_heat'.format(i)] = data.loc[ts, 'building_load{}.T_set_cool'.format(i)] - hys_hvac
        
        # Scale inputs
        for c in data.columns:
            if c in self.parameter['scaler'].keys():
                data[c] = data[c] * self.parameter['scaler'][c]
                
        return data
    
def calc_cost(load, tariff, ts=60, return_all=False):
    tz='Etc/GMT+8'
    tz_local='America/Los_Angeles'
    daytype_map = {0: 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday', 4: 'weekday',
                   5: 'weekend', 6: 'weekend'} # Monday=0, Sunday=6
    load = load.copy(deep=True).resample('{}S'.format(60*ts)).mean()
    load.columns = ['power']
    # Shift by 1 timestep
    load.index = load.index.shift(-1)
    ix_st = load.index.tz_localize(tz).tz_convert(tz_local)
    m = ix_st[0].month
    season = tariff['seasons_map'][tariff['seasons'][m]]
    load['hour'] = ix_st.hour
    load['period'] = [tariff[season]['hours'][daytype_map[x.weekday()]][x.hour] for x in ix_st]
    load['$/kWh'] = load['period'].replace(tariff[season]['energy'])
    load['energy_cost'] = load['power'] * load['$/kWh']/(60/float(ts))
    load['power_max'] = load['power'].abs().cummax()
    load['coincident_cost'] = load['power_max'] * tariff[season]['demand_coincident']
    load['period_max'] = load['power'].abs().groupby(load['period']).cummax()
    load['$/kW'] = load['period'].replace(tariff[season]['demand'])
    load['demand_cost'] = load['period_max'] * load['$/kW']
    periods = load['period'].unique()    
    load['sum_demand_cost'] = load['coincident_cost']
    if 0 in periods:
        load['sum_demand_cost'] += load['demand_cost'].mask(load['period']!=0, np.nan).ffill().fillna(0)
    if 1 in periods:
        load['sum_demand_cost'] += load['demand_cost'].mask(load['period']!=1, np.nan).ffill().fillna(0)        
    if 2 in periods:
        load['sum_demand_cost'] += load['demand_cost'].mask(load['period']!=2, np.nan).ffill().fillna(0)
        
    if return_all:
        return load
    else:
        return load[['energy_cost', 'sum_demand_cost']]
    
class ies_postprocessor(object):
    
    def __init__(self, parameter):
        self.parameter = parameter
        self.tariff = get_tariff(parameter['tariff'])
        self.episode_duration = round((parameter['fmu_final_time'] \
                                       - parameter['fmu_warmup_time']) / (24*60*60), 0)
        self.mode = 'cosim'
        self.add_forecast = parameter['add_forecast']
    
    def do_calc(self, data, init):
        axis = 1   
        year = str(self.parameter['year'])

        if init:
            # Initialize loads
            self.loads = pd.DataFrame()
            
            self.forecasts = {}
            col_map = {'T-out': 'TDryBul', 'S-dhi': 'HDifHor', 'S-dni': 'HDirNor', 'S-ghi': 'HGloHor', 
                       'kgco2/kw': 'ECo2pW'}
            if self.add_forecast:
                # Initialize E+ forecast
                added_load_offset = []
                for nf in range(1, self.parameter['fmu_kind']['feeder_nodes']):
                    f = self.parameter['fmu_kind']['building_fmus'][nf-1]
                    df = pd.read_csv(os.path.join(self.parameter['path_forecast'], f.replace('.fmu', '.csv')),
                                     index_col=[0], skiprows=2)
                    #df.index = pd.to_datetime(df.index)
                    df.index = pd.to_datetime(df['time'], origin=year, unit='s')
                    df = df.shift(-1).resample('1H').mean()
                    # Compute averages
                    scale_building = self.parameter['fmu_kind']['building_scale'][nf]
                    scale_building *= self.parameter['fmu_kind']['building_ft2'][nf] / 50e3
                    df['P-building'] = df['P'] * scale_building
                    df['P-light'] = df[[c for c in df.columns if c.startswith('P-light')]].sum(axis=1) * scale_building
                    df['Q-people'] = df[[c for c in df.columns if c.startswith('Q-people')]].sum(axis=1) * scale_building
                    df['P-equip'] = df[[c for c in df.columns if c.startswith('P-equip')]].sum(axis=1) * scale_building
                    if 'mediumoffice' in self.parameter['fmu_kind']['building_fmus'][nf-1]:
                        plugload_offset = 15e3 * scale_building
                        #print(f'INFO: Applying "plugload_offset" of {plugload_offset} W.')
                        added_load_offset.append(plugload_offset)
                    elif 'midriseapartment' in self.parameter['fmu_kind']['building_fmus'][nf-1]:
                        plugload_offset = 15e3 * scale_building
                        #print(f'INFO: Applying "plugload_offset" of {plugload_offset} W.')
                        added_load_offset.append(plugload_offset)
                    else:
                        plugload_offset = 0
                    df['P-equip'] += plugload_offset
                    # Assign forecasts
                    if not f'weaBus' in self.forecasts.keys():
                        self.forecasts[f'weaBus'] = df[['T-out', 'S-dhi', 'S-dni', 'S-ghi']].rename(columns=col_map)
                    hvac = compute_default_hvac(hour_start=df.index[0].hour)
                    df['T-cool'] = [hvac.loc[ix.hour, 'cool'] for ix in df.index]
                    df['T-heat'] = [hvac.loc[ix.hour, 'heat'] for ix in df.index]
                    self.forecasts[f'grid.Site[{nf+1}].building'] = df[['P-building','P-light', 'Q-people', 'P-equip',
                                                                        'T-cool', 'T-heat']]
                if len(added_load_offset) > 0:
                    print(f'INFO: Applied "plugload_offset" of {added_load_offset} W.')

                # Initialize PV forecast
                selected_pvs = self.parameter['fmu_kind']['selected_pvs']
                for ns in range(1, self.parameter['fmu_kind']['feeder_nodes']):
                    s  = selected_pvs[ns]
                    weather = self.parameter['fmu_kind']['weather_file'][ns]
                    f = f'pv_{weather}_{s}.csv'
                    df = pd.read_csv(os.path.join(self.parameter['path_forecast'], f), index_col=[0])
                    df.index = pd.to_datetime(df.index)
                    df = df.shift(-1).resample('1H').mean()
                    scale_pv = self.parameter['fmu_kind']['building_ft2'][ns] / 50e3
                    self.forecasts[f'grid.Site[{ns+1}].pv'] = df.rename(columns=col_map) * scale_pv
                    
                # Initialize EV forecast
                transport_map = {'Loc': 'ev_to_cha', 'P-drive': 'ev_drive_p'}
                if self.parameter['fmu_kind']['n_ev_transportation']:
                    for nt in range(self.parameter['fmu_kind']['n_ev_transportation']):
                        with open(self.parameter['fmu_kind']['ev_fmu_param'][nt]['fileName']) as f:
                            df_raw = f.read()

                        df = pd.DataFrame()
                        for k, v in transport_map.items():
                            t = pd.read_csv(StringIO(df_raw.split(v)[1].split('#1')[0]), skiprows=1, header=None, index_col=0)
                            t.index = pd.to_datetime(t.index+self.parameter['fmu_kind']['start_time'], origin=year, unit='s')
                            t = t[t.index.microsecond==0] # remove steps for Modelica
                            t.index.name = None
                            t.columns = [f'ev[{i:04d}].{k}' for i in t.columns]
                            if df.empty:
                                df = t.copy()
                            else:
                                df = pd.concat([df, t], axis=1)
                        # assuming df is only for 1 day; duplicating until end
                        if len(df) < 2000:
                            df = df[df.index.date==df.index[0].date()]
                        final_forecast_date = pd.to_datetime(self.parameter['fmu_kind']['final_time'], origin=year, unit='s')
                        df_base = df.copy()
                        date_offset = 1
                        while df.index[-1].date() < final_forecast_date+pd.DateOffset(days=2):
                            t = df_base.copy()
                            t.index = [ix+pd.DateOffset(days=date_offset) for ix in df_base.index]
                            df = pd.concat([df, t], axis=0)
                            date_offset += 1
                            
                        df = df.shift(-1).resample('1H').mean()
                        self.forecasts[f'ev_transport[{nt+1}]'] = df 
                    
            # GHG forecast (needed for calculation too)
            df = pd.read_csv(os.path.join(self.parameter['emissions_file']), index_col=[0])
            df.index = pd.to_datetime(df.index)
            df = df.shift(-1).resample('1H').mean()
            self.forecasts[f'grid'] = df.rename(columns=col_map)

        # Time
        data['sim_time'] = data.index
        
        # Grid power
        data['grid.P'] = data['grid.Grid.P_head'].values
        
        # Weather data
        data['weaBus.HDifHor'] = data['grid.Site[2].pv.weaBus.HDifHor']
        data['weaBus.HDirNor'] = data['grid.Site[2].pv.weaBus.HDirNor']
        data['weaBus.HGloHor'] = data['grid.Site[2].pv.weaBus.HGloHor']
        data['weaBus.TDryBul'] = data['grid.Site[2].pv.weaBus.TDryBul'] - 273.15
        
        # Building temperature
        if self.parameter['site_fmu'] == 'SmartBuilding_external.fmu' and not self.parameter['fmu_kind']['external_building']:
            for i in range(1, self.parameter['feeder_nodes']):
                cols = [c for c in data.columns if c.startswith('building_load{}.T_'.format(i)) and not 'set' in c]
                data[f'grid.Site[{i+1}].building.T_in'] = data[cols].mean(axis=axis)
                # Attention assumes single RC model
                data[f'grid.Site[{i+1}].building.T_slab'] = data[f'grid.Site[{i+1}].building.T_in']
                
        # Compute forecast
        for site in self.forecasts.keys():
            now = pd.to_datetime(data.index[0], unit='s', origin=str(self.forecasts[site].index[0].year))
            now_f = now+pd.DateOffset(hours=self.parameter['forecast_hours'])
            t = self.forecasts[site].loc[now:now_f]
            for c in t.columns:
                for nt, ts in enumerate(t.index):
                    data[f'{site}.forecast.{c}_{str(nt).zfill(2)}'] = t.loc[ts, c]
        
        # Individual building cost
        for i in range(1, self.parameter['feeder_nodes']):     
            load = data[[f'grid.Site[{i+1}].P']].copy() / 1e3 # in kW
            load.index = pd.to_datetime(load.index, unit='s', origin=year)
            self.loads.loc[load.index[0], [f'grid.Site[{i+1}].P']] = load.values[0]
            load = calc_cost(self.loads[[f'grid.Site[{i+1}].P']], self.tariff, return_all=True)
            data['reward_site{}_energy_cost'.format(i+1)] = \
                load['energy_cost'].iloc[-1] * self.parameter['scaler']['energy_cost']
            data['reward_site{}_demand_cost'.format(i+1)] = \
                load['sum_demand_cost'].diff().iloc[-1] * self.parameter['scaler']['demand_cost'] * self.episode_duration
            demand_max = load['power_max'].diff().iloc[-1]
            if init:
                data['reward_site{}_demand_cost'.format(i+1)] = 0
                demand_max = 0
            data['reward_site{}_cost'.format(i+1)] = data['reward_site{}_energy_cost'.format(i+1)]  + \
                data['reward_site{}_demand_cost'.format(i+1)]
            data['reward_site{}_demand'.format(i+1)] = demand_max
            
            now = pd.to_datetime(data.index[0], unit='s', origin=str(self.forecasts['grid'].index[0].year))
            emissions = data[f'grid.Site[{i+1}].P'] / 1e3 * self.forecasts['grid'].loc[now.replace(minute=0), 'ECo2pW'] # kgCO2 per kW
            data['reward_site{}_co2'.format(i+1)] = emissions * self.parameter['scaler']['emissions']
        data['reward_site_cost'] = data[['reward_site{}_cost'.format(i+1) for i in range(1, self.parameter['feeder_nodes'])]].sum(axis=axis)
        data['reward_site_demand'] = data[['reward_site{}_demand'.format(i+1) for i in range(1, self.parameter['feeder_nodes'])]].sum(axis=axis)
        data['reward_site_co2'] = data[['reward_site{}_co2'.format(i+1) for i in range(1, self.parameter['feeder_nodes'])]].sum(axis=axis)
            
        # Whole feeder cost
        load = data[['grid.P']].copy() / 1e3 # in kW
        load.index = pd.to_datetime(load.index, unit='s', origin=year)
        self.loads.loc[load.index[0], ['grid.P']] = load.values[0]
        load = calc_cost(self.loads[['grid.P']], self.tariff, return_all=True)
        data['reward_feeder_energy_cost'] = \
            load['energy_cost'].iloc[-1] * self.parameter['scaler']['energy_cost']
        data['reward_feeder_demand_cost'] = \
            load['sum_demand_cost'].diff().iloc[-1] * self.parameter['scaler']['demand_cost'] * self.episode_duration
        demand_max = load['power_max'].diff().iloc[-1]
        if init:
            demand_cost = 0
            demand_max = 0
        data['reward_feeder_cost'] = data['reward_feeder_energy_cost'] + data['reward_feeder_demand_cost'] 
        data['reward_feeder_demand'] = demand_max
        #data['reward_feeder_co2'] = data['grid.co2'] * self.parameter['scaler']['emissions']
        now = pd.to_datetime(data.index[0], unit='s', origin=str(self.forecasts['grid'].index[0].year))
        emissions = data['grid.P'] / 1e3 * self.forecasts['grid'].loc[now.replace(minute=0), 'ECo2pW'] # kgCO2 per kW
        data['reward_feeder_co2'] = emissions * self.parameter['scaler']['emissions']
        data['grid.tariff_energy'] = load['$/kWh'].values[-1]
        data['grid.tariff_demand'] = load['$/kW'].values[-1] 
        
        # Reward calculation
        data['reward'] = data[self.parameter['reward_col']].sum(axis=axis)
        
        return data
  
    
class ies_resetprocessor(object):
    def __init__(self, parameter):
        self.parameter = parameter
    
    def do_calc(self, data, parameter, init):
        if self.parameter['cleanup_energyplus']:
            for f in os.listdir(os.getcwd()):
                if f.startswith('Output_EPExport_'):
                    shutil.rmtree(f)
                    
#         if self.parameter['fmu_kind']['external_building']:
#             for i in range(1, self.parameter['feeder_nodes']): 
#                 data[f'grid.Site[{i}].building.T_in'] = 22.5
                    
        return data, parameter
