from neural_model import *
import phyre
import torch
import numpy as np

default_params = {
    'network_type'        :'resnet18',
    'train_batch_size'    : 64,
    'max_train_actions'   : 100000,
    'balance_classes'     : True,   # balancing is critical for the 2B-tasks
    'updates'             : 100000, # number of total updates to perform
    'cosine_scheduler'    : True,
    'learning_rate'       : 3e-4,
    'action_hidden_size'  : 512,
    'report_every'        : 1000,
    'eval_every'          : 100000,
    'eval_size'           : 10000,  # evaluation size of the train & dev data    # check later
    'num_auccess_actions' : 10000,  # number of actions to evaluate in AUCCESS
    'num_auccess_tasks'   : 200,
    
    'rank_size'            : 10000,
    'eval_batch_size'      : 128,
    'max_attempts_per_task': 100
    
}

class DQNAgent():

    def __init__(self, params = default_params):
        self.params = default_params
        self.neural_model = NeuralModel()
    
    def train(self, cache, task_ids, tier, dev_task_ids):
        model = self.neural_model.train(cache, 
                                        task_ids, 
                                        tier, 
                                        dev_task_ids, 
                                        default_params)
        state = dict(model =model, cache = cache)
        return state
    
    def eval(self, state, task_ids, tier):
        model  = state['model']
        cache  = state['cache']
        # NOTE: Current agent is only using the actions that are seen in the training set,
        #       though agent has the ability to rank the actions that are not seen in the training set
        actions = state['cache'].action_array[:self.params['rank_size']]
        
        model.cuda()
        simulator = phyre.initialize_simulator(task_ids, tier)
        observations = simulator.initial_scenes
        evaluator = phyre.Evaluator(task_ids)
        
        for task_index in range(len(task_ids)):
            task_id = simulator.task_ids[task_index]
            observation = observations[task_index]
            scores  = self.neural_model.eval_actions(model,
                                                     actions,
                                                     self.params['eval_batch_size'],
                                                     observation)
            # Rank of the actions in descending order
            action_order = np.argsort(-scores)
            # Result of the actions are already stored in cache
            statuses = cache.load_simulation_states(task_id)
            
            for action_id in action_order:
                if evaluator.get_attempts_for_task(task_index) >= self.params['max_attempts_per_task']:
                    break
                status = phyre.SimulationStatus(statuses[action_id])
                evaluator.maybe_log_attempt(task_index, status)
        return evaluator

if __name__=='__main__':
    agent = DQNAgent()