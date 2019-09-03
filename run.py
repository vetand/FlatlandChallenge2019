from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
import heapq
import numpy as np



#####################################################################
# Instantiate a Remote Client
#####################################################################
remote_client = FlatlandRemoteClient()

#####################################################################
# Define your custom controller
#
# which can take an observation, and the number of agents and 
# compute the necessary action for this step for all (or even some)
# of the agents
#####################################################################
def heuristic(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)

class Node:
    
    def __init__(self, env, input_x, input_y, orientation, finish_x, finish_y, input_g):
        self.id = env.width * env.height * orientation + input_y * env.height + input_x
        self.x = input_x
        self.y = input_y
        self.dest = orientation
        self.g_value = input_g
        self.h_value = heuristic(input_x, input_y, finish_x, finish_y)
        self.f_value = self.h_value + self.g_value
        
    def __lt__(self, other):
        if self.f_value == other.f_value:
            return self.g_value < other.g_value
        else:
            return self.f_value < other.f_value

answer_build = False
agent_action = []
current_pos = []

def build(env):
    global agent_action
    last_visit = []
    for x in range(env.height):
        last_visit.append([-1] * env.width)
    
    agent_action = []
    for ind in range(env.get_num_agents()):
        agent_action.append([])
    
    for ind in range(env.get_num_agents()):
        previous = []
        distance = []
        for dest in range(4):
            previous.append([])
            distance.append([])
        for dest in range(4):
            for x in range(env.height):
                previous[dest].append([])
                distance[dest].append([-1] * env.width)
        for dest in range(4):
            for x in range(env.height):
                for y in range(env.width):
                    previous[dest][x].append([-1, -1, -1])

        agent = env.agents[ind]
        OPEN = []
        CLOSED = set()
        USED = set()
        start = Node(env, agent.position[0], agent.position[1], agent.direction, agent.target[0], agent.target[1], 0)
        USED.add(start.id)
        finish_x = agent.target[0]
        finish_y = agent.target[1]
        heapq.heappush(OPEN, start)
        orientation = 0
        while (len(OPEN) != 0):
            current = heapq.heappop(OPEN)
            parent = previous[current.dest][current.x][current.y]
            CLOSED.add(current.id)
            if (current.x == finish_x and current.y == finish_y):
                orientation = current.dest
                break
            position = [current.x, current.y]
            available = env.rail.get_transitions(*position, current.dest)
            candidates = []
            if (available[0] == True):
                candidates.append(Node(env, current.x - 1, current.y, 0, finish_x, finish_y, current.g_value + 1))
            if (available[1] == True):
                candidates.append(Node(env, current.x, current.y + 1, 1, finish_x, finish_y, current.g_value + 1))
            if (available[2] == True):
                candidates.append(Node(env, current.x + 1, current.y, 2, finish_x, finish_y, current.g_value + 1))
            if (available[3] == True):
                candidates.append(Node(env, current.x, current.y - 1, 3, finish_x, finish_y, current.g_value + 1))
            for neighbor in candidates:
                if (neighbor.id not in USED or distance[neighbor.dest][neighbor.x][neighbor.y] > current.g_value + 1):
                    USED.add(neighbor.id)
                    previous[neighbor.dest][neighbor.x][neighbor.y] = [current.dest, current.x, current.y]
                    heapq.heappush(OPEN, neighbor)
                    distance[neighbor.dest][neighbor.x][neighbor.y] = current.g_value + 1
        start_x = agent.position[0]
        start_y = agent.position[1]
        path = []
        while (finish_x != start_x or finish_y != start_y):
            path.append([orientation, finish_x, finish_y])
            parent_node = previous[orientation][finish_x][finish_y]
            finish_x = parent_node[1]
            finish_y = parent_node[2]
            orientation = parent_node[0]
        path.append([orientation, finish_x, finish_y])
        path.reverse()

        time = -1
        for node_num in range(1, len(path)):
            node = path[node_num]
            x = node[1]
            y = node[2]
            dest = node[0]
            next_time = max(time, last_visit[x][y]) + 1
            last_visit[path[node_num - 1][1]][path[node_num - 1][2]] = next_time - 1
            while time < next_time - 1:
                time += 1
                agent_action[ind].append(4)
            time += 1
            pre_dest = path[node_num - 1][0]
            if (abs(pre_dest - dest) % 2 == 0):
                agent_action[ind].append(2)
            elif ((pre_dest + 1) % 4 == dest):
                agent_action[ind].append(3)
            else:
                agent_action[ind].append(1)

def my_controller(env, number_of_agents):
    global answer_build
    global agent_action
    global current_pos
    if (answer_build == False):
        build(env)
        current_pos = [0] * env.get_num_agents()
        answer_build = True
    _action = {}
    for ind in range(env.get_num_agents()):
        if (current_pos[ind] < len(agent_action[ind])):
            _action[ind] = agent_action[ind][current_pos[ind]]
            current_pos[ind] += 1
    return _action

#####################################################################
# Instantiate your custom Observation Builder
# 
# You can build your own Observation Builder by following 
# the example here : 
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14
#####################################################################
my_observation_builder = TreeObsForRailEnv(
                                max_depth=3,
                                predictor=ShortestPathPredictorForRailEnv()
                            )

#####################################################################
# Main evaluation loop
#
# This iterates over an arbitrary number of env evaluations
#####################################################################
evaluation_number = 0
while True:

    answer_build = False
    agent_action = []
    current_pos = []
    evaluation_number += 1
    # Switch to a new evaluation environemnt
    # 
    # a remote_client.env_create is similar to instantiating a 
    # RailEnv and then doing a env.reset()
    # hence it returns the first observation from the 
    # env.reset()
    # 
    # You can also pass your custom observation_builder object
    # to allow you to have as much control as you wish 
    # over the observation of your choice.
    observation = remote_client.env_create(
                    obs_builder_object=my_observation_builder
                )
    if not observation:
        #
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been 
        # evaluated on all the required evaluation environments,
        # and hence its safe to break out of the main evaluation loop
        break
    
    print("Evaluation Number : {}".format(evaluation_number))

    #####################################################################
    # Access to a local copy of the environment
    # 
    #####################################################################
    # Note: You can access a local copy of the environment 
    # by using : 
    #       remote_client.env 
    # 
    # But please ensure to not make any changes (or perform any action) on 
    # the local copy of the env, as then it will diverge from 
    # the state of the remote copy of the env, and the observations and 
    # rewards, etc will behave unexpectedly
    # 
    # You can however probe the local_env instance to get any information
    # you need from the environment. It is a valid RailEnv instance.
    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    # Now we enter into another infinite loop where we 
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    # 
    # An episode is considered done when either all the agents have 
    # reached their target destination
    # or when the number of time steps has exceed max_time_steps, which 
    # is defined by : 
    #
    # max_time_steps = int(1.5 * (env.width + env.height))
    #
    while True:
        #####################################################################
        # Evaluation of a single episode
        #
        #####################################################################
        # Compute the action for this step by using the previously 
        # defined controlle
        action = my_controller(local_env, number_of_agents)

        # Perform the chosen action on the environment.
        # The action gets applied to both the local and the remote copy 
        # of the environment instance, and the observation is what is 
        # returned by the local copy of the env, and the rewards, and done and info
        # are returned by the remote copy of the env
        observation, all_rewards, done, info = remote_client.env_step(action)
        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))
            #
            # When done['__all__'] == True, then the evaluation of this 
            # particular Env instantiation is complete, and we can break out 
            # of this loop, and move onto the next Env evaluation
            break

print("Evaluation of all environments complete...")
########################################################################
# Submit your Results
# 
# Please do not forget to include this call, as this triggers the 
# final computation of the score statistics, video generation, etc
# and is necesaary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())