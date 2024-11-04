import os
import numpy as np

try:
    root = os.path.dirname(os.path.realpath(__file__))
except:
    root = os.getcwd()
    
eplus_fmus = ['refbldg_mediumoffice_new2004_v1-4_7-2-USA_CA_Los.Angeles.Intl.AP.722950_TMY3',
              'refbldg_midriseapartment_new2004_v1-4_7-2-USA_CA_Los.Angeles.Intl.AP.722950_TMY3']
    
def get_simplecoupled_config(seed=1, log_level=0, start_time=0, final_time=0, step_size=60*60*1,
                             step_size_internal=5*60, pv=True, eplus_buildings=True, run_period='summer',
                             feeder_nodes=3, building_ft2=50e3, der_scale=1, battery_scale=1,
                             eplus_emulated=False, use_smartinverter=False, pv_scale=[0.5, 1.0],
                             grid='ieee13', smartbuilding_fmus=False, external_building=False,
                             n_ev_transportation=None, charger_scale=1, n_cha=None, use_ltc=False,
                             ltc_sensLoc=7, use_chargerconnect=False):
    '''
    This function is used to configure the co-simulation and underlaying modules.
    
    Inputs
    ------
    seed (float): Seed for random functions. Pass None to disable. (default = 1)
    log_level (int): Log level of simulators. Pass 0-quiet, 4-debug. (default = 0)
    start_time (float): Start time of the simulation. Jan 1 midnight is 0. (default = 0)
    final_time (float): Final time of the simulation. If set to 0 "run_period" is used. (default = 0)
    step_size (float): Step size of external inputs/outputs. (default = 60*60*1)
    step_size_internal (float): Internal step size of simulators. (default = 5*60)
    pv (bool): Flag to use PV generation. (default = True)
    eplus_buildings (bool): Flag to use EnergyPlus for building simulation. (default = True)
    run_period (str): Selection of pre-defined simulation period. "final_time" must be set to 0. (default = 'summer')
    feeder_nodes (int): Number of feeder nodes to be used. Note node 0 does not have loads. (default = 3)
    building_ft2 (float): Scaling of building size. (default = 50e3)
    der_scale (float): Scaling of PV and battery. (default = 1)
    battery_scale (float): Scaling of battery. (default = 1)
    eplus_emulated (bool): Flag to use emulated energy plus. (default = False)
    use_smartinverter (bool): Flag to use smart inverters. (default = False)
    pv_scale (list): Scaling of PV [min, max]. (default = [0.5, 1.0])
    grid (str): Selection of grid model. (default = 'ieee13')
    smartbuilding_fmus (bool): 
    external_building (bool):
    n_ev_transportation (int): Number of ev/transporation models. (default = None)
    charger_scale (float): Scale of EV chargers. (default = 1)
    
    Returns
    -------
    config (dict): The configuration dictionary.
    '''
    # Basic config
    config = {}
    config['dir_fmu'] = os.path.join(root, '..', 'resources', 'fmus')
    config['dir_weather'] = os.path.join(root, '..', 'resources', 'weather')
    config['path_forecast'] = os.path.join(root, '..', 'resources', 'forecast')
    config['dir_evtransport'] = os.path.join(root, '..', 'resources', 'transportation')
    config['seed'] = seed
    config['log_level'] = log_level # 0-off; 4-info
    config['master'] = 'cs' # selection of master algorithm ['cs', 'me']

    config['instance_id'] = 0
    config['feeder_nodes'] = feeder_nodes # max. 34 nodes for IEEE 34
    config['dynamics'] = 5 * 60 # s
    config['tolerance'] = 1e-5
    config['external_building'] = external_building
    config['smartbuilding_fmus'] = smartbuilding_fmus
    config['n_ev_transportation'] = n_ev_transportation
    config['n_cha'] = n_cha
    if config['smartbuilding_fmus']:
        print('WARNING: "smartbuilding_fmus" must be enbaled. Setting to "False".')
        config['smartbuilding_fmus'] = False
        
    # Grid Setup
    config['grid'] = grid
    config['grid_fmu_param'] = {}
    if grid == 'copperplate':
        feeder_nodes_total = 34
        if use_smartinverter:
            config['grid_fmu'] = 'Copperplate_smartinverter.fmu'
        else:
            config['grid_fmu'] = 'Copperplate.fmu'
    elif grid == 'ieee13':
        feeder_nodes_total = 13
        if config['smartbuilding_fmus']:
            if use_smartinverter:
                config['grid_fmu'] = 'IEEE13_smartinverter.fmu'
            else:
                config['grid_fmu'] = 'IEEE13.fmu'
        else:
            if config['n_ev_transportation']:
                config['grid_fmu'] = 'IEEE13_smartBuilding_ev_external.fmu'
            else:
                config['grid_fmu'] = 'IEEE13_smartBuilding_external.fmu'
    elif grid == 'ieee34':
        feeder_nodes_total = 34
        if config['smartbuilding_fmus']:
            if use_smartinverter:
                config['grid_fmu'] = 'IEEE34_smartinverter.fmu'
            else:
                config['grid_fmu'] = 'IEEE34.fmu'
        else:
            config['grid_fmu'] = 'IEEE34_smartBuilding_external.fmu'
    if use_ltc:
        config['grid_fmu'] = config['grid_fmu'].replace('external.fmu', f'external_ltcN{int(ltc_sensLoc)}.fmu')
        # config['grid_fmu_param']['Grid.sensLoc'] = ltc_sensLoc
        config['grid_fmu_param']['Grid.deadBand'] = 2 * 2 # 4 Volt dead band
    config['use_ltc'] = use_ltc
    config['feeder_nodes_total'] = feeder_nodes_total

    # Building Setup
    if eplus_buildings:
        config['site_fmu'] = 'SmartBuilding_external.fmu'
        config['building_fmus'] = []
        config['building_scale'] = [0]
        np.random.seed(config['seed']+1)
        for i, prefix in enumerate(np.random.choice(eplus_fmus, config['feeder_nodes_total']-1)):
            config['building_fmus'].append(f'{prefix}_{i+1}.fmu')
            config['building_scale'].append(3 if 'midriseapartment' in prefix else 1)
    else:
        print('WARNING: Only EnergyPlus "eplus_buildings=True" supported.')
        config['site_fmu'] = 'SmartBuilding.fmu'
        config['building_fmus'] = []
        config['building_scale'] = [0] + [1] * (config['feeder_nodes_total']-1)
    if type(building_ft2) == type([]):
        np.random.seed(config['seed']+1)
        config['building_ft2'] = np.random.choice(building_ft2, config['feeder_nodes_total'])
        config['building_ft2'][0] = 1e-3
        config['building_ft2'][config['feeder_nodes']:] = 1e-3
    else:
        config['building_ft2'] = [1e-3] + [building_ft2] * (config['feeder_nodes'] - 1) \
            + [1e-3] * (config['feeder_nodes_total'] - config['feeder_nodes'])
    config['der_scale'] = [1e-3] + [der_scale] * (config['feeder_nodes'] - 1) \
        + [1e-3] * (config['feeder_nodes_total'] - config['feeder_nodes'])
    config['battery_scale'] = [1e-3] + [battery_scale] * (config['feeder_nodes'] - 1) \
        + [1e-3] * (config['feeder_nodes_total'] - config['feeder_nodes'])
    config['eplus_emulated'] = f'EplusCsvReader.fmu' if eplus_emulated else False
    config['weather_file'] = ['USA_CA_Los.Angeles.Intl.AP.722950_TMY3.mos'] * config['feeder_nodes_total']
    
    if run_period and final_time > 0:
        print('WANRING: "run_period" overwriting "final_time".')
    if run_period == 'summer':
        start_time = (150)*24*60*60
        final_time = (152)*24*60*60
    elif run_period == 'winter':
        start_time = (39)*24*60*60
        final_time = (41)*24*60*60
    config['run_period'] = run_period
    config['start_time'] = start_time
    config['final_time'] = final_time
    config['step_size'] = step_size
    config['step_size_internal'] = step_size_internal
    config['filter'] = ['P_load*', '*co2', 'V_*', '*cost',
                        'P_set_*', 'T_*_*', '*.SOC', '*.pv.weaBus.HGloHor', '*.pv.weaBus.TDryBul',
                        '*.pv.weaBus.HDirNor', '*.pv.weaBus.HDifHor', 'ieee*.Vpu*', 'Grid.VoltVarWatt*.PLim',
                        'Grid.VoltVarWatt*.QCtrl','Grid.P_*','Site*.P*','Site*.building.P*','Site*.pv.P*',
                        'Grid.ieee13.Vpu*','PPlugCtrl*','SOC*','P*','P_site*','Grid.LTC_ctrl1.*']
    config['control_vars'] = {'site{}.P_set_battery': 'Setpoint for Battery (<0 => discharge; >0 => charge) [W]',
                              'site{}.T_set_cool': 'HVAC cooling setpoint [K]',
                              'site{}.T_set_heat': 'HVAC heating setpoint [K]'}
    config['objective'] = ['grid.Grid.co2', 'grid.Site[{}].cost']
                        
    # EV Transport Setup
    config['charger_scale'] = [1e-3] + [charger_scale] * (config['feeder_nodes'] - 1) \
        + [1e-3] * (config['feeder_nodes_total'] - config['feeder_nodes'])
    if use_chargerconnect:
        config['charger_reconnect'] = np.random.uniform(low=config['step_size']/10,
                                                        high=config['step_size'],
                                                        size=config['feeder_nodes_total']) # stochastic reconnect
    else:
        config['charger_reconnect'] = [1e6]*config['feeder_nodes_total']
        
    if config['n_ev_transportation']:
        config['ev_fmu'] = 'EV_transport.fmu'
        # https://evstatistics.com/2022/04/bev-batteries-average-83-kwh-versus-15-kwh-for-phevs/
        config['ev_fmu_param'] = []
        ev_seed_map = [0, 1, 3, 4, 5]
        for t in range(config['n_ev_transportation']):
            fileName = os.path.join(config['dir_evtransport'], f'test_200-500-13-{ev_seed_map[t]}.txt')
            config['ev_fmu_param'].append({'fileName': fileName,
                                           'startTime': start_time,
                                           'EMax': 78e3,
                                           'Pmax': 7.2e3,
                                           'SOC_start': 0.5})
        config['n_evs'] = 200
        if config['n_cha']:
            if config['n_cha'] != 200:
                print(f'WARNING: The EV model is built for 500 n_cha, not {config["n_cha"]}.')
        else:
            config['n_cha'] = 500
        if config['feeder_nodes'] != 13:
            print(f'WARNING: The EV model is built for 13 nodes, not {config["feeder_nodes"]}.')
    else:
        config['ev_fmu'] = None
        config['ev_fmu_param'] = []
        config['n_evs'] = None
                        
    # PV Setup
    config['use_smartinverter'] = use_smartinverter
    if pv:
        config['PV_scale'] = pv_scale
    else:
        config['PV_scale'] = [0.0, 0.0]
    np.random.seed(config['seed'])
    config['selected_pvs'] = np.random.uniform(low=config['PV_scale'][0],
                                               high=config['PV_scale'][1],
                                               size=config['feeder_nodes_total'])
    config['selected_pvs'] = np.round(config['selected_pvs'], 4)
    if use_smartinverter:
        config['smartinverter_QMax'] = np.round(config['selected_pvs'] * config['building_ft2'] / 50e3 * (50e3 * 0.3), 1) # 50 kW PV * 30 %
        config['smartinverter_QMax'][0] = 0
    else:
        config['smartinverter_QMax'] = [0] * config['feeder_nodes_total']
    return config