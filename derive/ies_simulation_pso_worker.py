import os
import sys
import time
import random
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue

try:
    root = os.path.dirname(os.path.realpath(__file__))
except:
    root = os.getcwd()
    
def setup_and_run(parameter_gym, inputs, fmi_gym, me_handler, agent_ctrl=None, parameter_ctrl={}, q=None):
    env = fmi_gym(parameter_gym, pyfmi=me_handler)
    if agent_ctrl:
        ctrl = agent_ctrl(parameter_gym, parameter_ctrl, env)
    done = False
    state = env.reset()
    i = 0
    while not done:
        if agent_ctrl:
            action = ctrl.forward(state)
        else:
            action = inputs.iloc[i].values
        #rep.log(action, lvl=logging.INFO)
        state, reward, done, info = env.step(action)
        i += 1
    #timing = env.fmu.simulator.timing
    res = env.data.copy(deep=True).iloc[1:]
    res.index = pd.to_datetime(res.index, unit='s', origin=str(parameter_gym['year']))
    env.close()
    del env
    if q:
        q.put(res['reward_feeder_cost'].cumsum().values[-1]*-1)
    else:
        return res

def worker(inputs_all, ies_path='', parameter_gym={}, print_console=True, battery_only=False, cleanup=True,
           res_dir='', controller='direct'):
    
    from pyswarms.utils import Reporter
    import logging
    
    rep = Reporter()

    seed = int(random.random()*1e9)
    #rep.log(seed, lvl=logging.INFO)
    np.random.seed(seed)
    delay = np.random.random()*10
    tmp_dir = f'/tmp/tmp_{seed}'
    #rep.log(delay, lvl=logging.INFO)
    #time.sleep(delay)
    
    os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    
    
    parameter_gym['fmu_kind']['instance_id'] = str(seed)
    #rep.log(parameter_gym['fmu_kind']['instance_id'], lvl=logging.INFO)
    
    horizon = int((parameter_gym['fmu_final_time'] - parameter_gym['fmu_warmup_time']) / parameter_gym['fmu_step_size'])

    #run = 0
    run = seed
    inputs_hourly = None
    res_dir = None
    #print_console = True
    
    sys.path.append(os.path.join(ies_path, 'environment'))
    from fmi_mlc import fmi_gym
    from orchestrator_ies import me_handler
    from controller import get_control, make_heur_table
    
    #rep.log(inputs, lvl=logging.INFO)
    #rep.log(inputs[0], lvl=logging.INFO)
    
    r = []
    for inputs in inputs_all:
        if controller.startswith('heuristic'):
            if controller == 'heuristic_separate':
                agent_ctrl, parameter_ctrl = get_control(name='heuristic-individual')
                parameter_ctrl = {'lookup': [make_heur_table(inputs[:3]),
                                             make_heur_table(inputs[3:])]}
            elif controller == 'heuristic_combined':
                agent_ctrl, parameter_ctrl = get_control(name='heuristic-combined')
                parameter_ctrl = {'lookup': make_heur_table(inputs)}
        else:
            agent_ctrl=None
            parameter_ctrl={}
            inputs = inputs.reshape(horizon, int(len(inputs)/horizon))
            if battery_only:
                inputs = pd.DataFrame(inputs, columns=[c for c in parameter_gym['action_names'] if 'battery' in c])
                for c in parameter_gym['action_names']:
                    if not c in inputs.columns:
                        inputs[c] = 0
                inputs = inputs.loc[:, parameter_gym['action_names']]
            else:
                inputs = pd.DataFrame(inputs, columns=parameter_gym['action_names'])

        #rep.log(inputs, lvl=logging.INFO)

        #print('HELLO')
        
        
#         heur = pd.DataFrame(index=range(24))
#         heur['P_set_battery'] = 0
#         heur['T_set_cool'] = 0
#         heur['T_set_heat'] = 0
#         heur['P_set_battery'].loc[:6] = 60e3
#         heur['P_set_battery'].loc[11:17] = -50e3
#         heur['P_set_battery'].loc[17:] = -25e3
#         #heur['P_set_battery'].loc[17:] = 25e3
#         parameter_heur = {'lookup': heur}


        st = time.time()

        complete = False
        n_try = 0
        
        '''
        while not complete:
            try:
                queue = Queue()
                p = Process(target=setup_and_run, args=(parameter_gym, inputs, fmi_gym, me_handler, queue))
                p.start()
                p.join()
                t = queue.get()
                r.append(t)
                complete = True
            except Exception as e:
                n_try += 1
                time.sleep(delay)
                if n_try > 0:
                    print('ERROR: Run {:07d} did not conclude.'.format(run))
                    print(e)
                    r.append(1e3)
                    complete = True
        '''
        
        while not complete:
            try:
                res = setup_and_run(parameter_gym, inputs, fmi_gym, me_handler, agent_ctrl, parameter_ctrl)
                r.append(res['reward_feeder_cost'].cumsum().values[-1]*-1)
                complete = True
            except Exception as e:
                n_try += 1
                time.sleep(delay)
                if n_try > 0:
                    print('ERROR: Run {:07d} did not conclude.'.format(run))
                    print(e)
                    r.append(1e3)
                    complete = True
        #print('Completed Run {:07d} after {} s.'.format(run, round(time.time()-st,1)))
    
    
    if cleanup:
        os.chdir('/tmp')
        shutil.rmtree(tmp_dir)
    
    return np.array(r)
    
    '''
    try:
        
        # Instantiate simulator
        
        

        
        
        res = setup_and_run(parameter_gym, me_handler, inputs)
        
        

        
        
        
        # Instantiate simulator
        #from simplecoupled import simplecoupled
        #config['instance_id'] = run
        #simulator = simplecoupled(config)

        # Simulate
        #res = simulator.simulate(inputs=inputs)
        #res = res.resample('1H').mean()
        
        #print(res)

        # Store results
        #inputs_hourly.to_csv(os.path.join(res_dir, 'inputs_run{:05d}.csv'.format(run)))
        #res.to_csv(os.path.join(res_dir, 'results_run{:05d}.csv'.format(run)))
        #pd.DataFrame({0:simulator.get_objective()}).transpose().to_csv(os.path.join(res_dir, 'objective_run{:05d}.csv'.format(run)))

        print('Completed Run {:07d} after {} s.'.format(run, round(time.time()-st,1)))
        if print_console:
            os.write(1, 'Completed Run {:07d} after {} s.\n'.format(run, round(time.time()-st,1)).encode())
            
        # Return reward
        r = res['reward_feeder_cost'].cumsum().values[-1]
        #rep.log(r, lvl=logging.INFO)
        #rep.log(np.array([r]), lvl=logging.INFO)
        return np.array([r*-1])

    except Exception as e:
        print('ERROR: Run {:07d} did not conclude.'.format(run))
        print(e)
        return np.array([1e3])
    '''