from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
from my_observation_builder import CustomObservationBuilder
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
import numpy as np
import time

remote_client = FlatlandRemoteClient()

my_observation_builder = DummyObservationBuilder()

def my_controller(step_number, number_of_agents):
    if (step_number == 0):
        time.sleep(100)
    action = dict()
    for ind in range(number_of_agents):
        action[ind] = 2
    return action

evaluation_number = 0
while True:

    evaluation_number += 1
    
    time_start = time.time()
    observation, info = remote_client.env_create(
                    obs_builder_object=my_observation_builder
                )
    env_creation_time = time.time() - time_start
    if not observation:
        break
    
    print("Evaluation Number : {}".format(evaluation_number))

    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    while True:
        time_start = time.time()
        action = my_controller(steps, number_of_agents)
        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)

        time_start = time.time()
        observation, all_rewards, done, info = remote_client.env_step(action)
        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)

        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))
            break
    
    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("="*100)
    print("="*100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("="*100)

print("Evaluation of all environments complete...")
print(remote_client.submit())
