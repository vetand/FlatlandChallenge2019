from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
import heapq
import numpy as np
from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

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
answer_build = False
agent_action = []
current_pos = []

class Map:
    def __init__(self):
        self.height = -1
        self.width = -1
        self.cellSize = -1
        self.grid = []

    def getMapOptions(self):
        global env

        self.height = env.height
        
        self.width = env.width

        self.cellSize = 1


class Options:
    def __init__(self):
        self.metricType = "manhattan"
        self.searchType = "astar"
        self.hweight = 1
        self.breakingTies = "g-max"
        self.allowSqueeze = False
        self.allowDiagonal = False
        self.cutCorners = False

    def getOptions(self):
        self.metricType = "manhattan"
        self.searchType = "astar"
        self.hweight = 1
        self.breakingTies = "g-max"
        self.allowSqueeze = False
        self.allowDiagonal = False
        self.cutCorners = False


class Agent:
    def __init__(self, agentId):
        self.start_i = -1
        self.start_j = -1
        self.fin_i = -1
        self.fin_j = -1
        self.agentId = agentId

    def getAgent(self):
        global env
        self.start_i = env.agents[self.agentId].position[0]
        self.start_j = env.agents[self.agentId].position[1]


        self.fin_i = env.agents[self.agentId].target[0]
        self.fin_j = env.agents[self.agentId].target[1]

        self.dir = env.agents[self.agentId].direction


class Agents:
    def __init__(self):
        self.allAgents = []
        self.size = 0

    def getAgents(self):
        global env
        self.size = env.get_num_agents()


class Node:
    def __init__(self, i, j, dir):
        self.i = i
        self.j = j
        self.t = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.dir = dir
        self.parent = None

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir

    def __ne__(self, other):
        return not (self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir)

    def __lt__(self, other):
        if (self.i == other.i):
                if (self.j == other.j):
                        return self.dir < other.dir
                return self.j < other.j
        return self.i < other.i

    def __hash__(self):
        return hash((self.i, self.j, self.t, self.dir))


class Entry:
    def __init__(self, priority, value, options):
        self.priority = priority
        self.value = value
        self.breakingTies = options.breakingTies

    def cmp(self, one, two):
        if (one.f == two.f):
            if (one.g == two.g):
                return one < two
            else:
                if (self.breakingTies == "g-min"):
                    return one.g < two.g
                return one.g > two.g
        return one.f < two.f

    def __cmp__(self, other):
        return self.cmp(self.priority, other.priority)

    def __lt__(self, other):
        return self.cmp(self.priority, other.priority)

    def __eq__(self, other):
        return self.priority == other.priority


class ISearch:
    def __init__(self, options):
        global env
        self.lppath = []
        self.hppath = []
        self.printGrid = []
        self.reservations = [set()]
        self.breakingTies = options.breakingTies
        self.sqrtTwo = 1.41421356237
        self.maxTime = 400

    def startAllAgents(self, map, options, agents):
        self.initReservations(map)

        for i in range(agents.size):
            self.startSearch(map, options, agents.allAgents[i])

    def initPrintGrid(self, map):
        self.printGrid.clear()
        for i in range(map.height):
            line = [0] * map.width
            for j in range(map.width):
                val = map.getValue(i, j)
                if val == 0:
                    line[j] = -2
                else:
                    line[j] = -1
            self.printGrid.append(line)

    def initReservations(self, map):
        self.reservations = sets = [set() for i in range(self.maxTime)]

    def checkReservation(self, i, j, t):
        return ((i, j) in self.reservations[t])

    def computeHFromCellToCell(self, i1, j1, i2, j2, options):
        return abs(i1 - i2) + abs(j1 - j2)

    def startSearch(self, map, options, agent):
        prevTime = time.time()

        startNode = Node(agent.start_i, agent.start_j, agent.dir)
        startNode.h = self.computeHFromCellToCell(agent.start_i, agent.start_j, agent.fin_i, agent.fin_j, options)
        startNode.f = startNode.g + startNode.h

        finNode = Node(agent.fin_i, agent.fin_j, agent.dir)

        self.lppath = []
        self.hppath = []

        openHeap = []
        openCopy = dict()
        closed = set()

        numberOfSteps = 0
        pathFound = False

        entry = Entry(startNode, None, options)
        heapq.heappush(openHeap, entry)
        openCopy[(startNode.i, startNode.j, startNode.dir, startNode.t)] = (startNode.h, startNode.f)

        while (not pathFound) and len(openHeap) > 0:
            numberOfSteps += 1

            curNode = (heapq.heappop(openHeap)).priority

            if (curNode.i == finNode.i and curNode.j == finNode.j):
                finNode = curNode
                pathFound = True

            else:
                openCopy.pop((curNode.i, curNode.j, curNode.dir, curNode.t))
                closed.add(curNode)

                successors = self.findSuccessors(curNode, map, options, agent)
                for i in range(len(successors)):
                    scNode = successors[i]
                    foundInClosed = False
                    if (scNode in closed):
                        foundInClosed = True

                    if ((not foundInClosed) and curNode.g + 1 <= scNode.g):

                        scNode.g = curNode.g + 1
                        scNode.h = self.computeHFromCellToCell(scNode.i, scNode.j, agent.fin_i, agent.fin_j, options)
                        scNode.f = scNode.g + scNode.h
                        scNode.t = curNode.t + 1

                        scNode.parent = curNode

                        foundInOpen = False
                        if ((scNode.i, scNode.j, scNode.dir, scNode.t) in openCopy.keys()):
                            check = openCopy.get((scNode.i, scNode.j, scNode.dir, scNode.t))
                            checkH = check[0]
                            checkF = check[1]

                            foundInOpen = True

                        if (foundInOpen == True and checkF > scNode.f):
                            # удалить старую, вставить новую
                            checkNode = Node(scNode.i, scNode.j, scNode.dir)
                            checkNode.t = scNode.t
                            entry = Entry(checkNode, None, options)

                            openHeap.remove(entry)
                            heapq.heapify(openHeap)

                            entry = Entry(scNode, None, options)
                            heapq.heappush(openHeap, entry)

                            openCopy[(scNode.i, scNode.j, scNode.dir, scNode.t)] = (scNode.h, scNode.f)

                        if (foundInOpen == False):
                            entry = Entry(scNode, None, options)
                            heapq.heappush(openHeap, entry)
                            openCopy[(scNode.i, scNode.j, scNode.dir, scNode.t)] = (scNode.h, scNode.f)

        pathLength = 0.0

        if pathFound:
            pathLength = finNode.g
            self.makePrimaryPath(finNode, startNode)
            self.makeFlatlandFriendlyPath(agent)

    def findSuccessors(self, curNode, map, options, agent):
                global env
                position = [curNode.i, curNode.j]
                available = env.rail.get_transitions(*position, curNode.dir)
                inter_answer = []
                if (available[0] == True):
                        inter_answer.append(Node(curNode.i - 1, curNode.j, 0))
                if (available[1] == True):
                        inter_answer.append(Node(curNode.i, curNode.j + 1, 1))
                if (available[2] == True):
                        inter_answer.append(Node(curNode.i + 1, curNode.j, 2))
                if (available[3] == True):
                        inter_answer.append(Node(curNode.i, curNode.j - 1, 3))
                inter_answer.append(Node(curNode.i, curNode.j, curNode.dir))
                successors = []
                for ind in range(len(inter_answer)):
                        inter_answer[ind].g = curNode.g + 1
                        inter_answer[ind].h = self.computeHFromCellToCell(inter_answer[ind].i, inter_answer[ind].j, agent.fin_i, agent.fin_j, options)
                        inter_answer[ind].f = inter_answer[ind].g + inter_answer[ind].h
                        inter_answer[ind].t = curNode.t + 1
                        if (not self.checkReservation(inter_answer[ind].i, inter_answer[ind].j, inter_answer[ind].t)): # место не занято
                                if (self.checkReservation(inter_answer[ind].i, inter_answer[ind].j, inter_answer[ind].t + 1)):
                                    continue
                                if (not (self.checkReservation(inter_answer[ind].i, inter_answer[ind].j, curNode.t) and self.checkReservation(curNode.i, curNode.j, inter_answer[ind].t))): #нет edge conflict
                                        successors.append(inter_answer[ind])
                return successors

    def makePrimaryPath(self, curNode, startNode):
        # агент стоит в финишной вершине до конца
        for i in range(curNode.t, self.maxTime):
            self.reservations[i].add((curNode.i, curNode.j))

        while curNode != startNode:
            self.lppath.append(curNode)
            self.reservations[curNode.t].add((curNode.i, curNode.j))
            curNode = curNode.parent

        self.lppath.append(curNode)
        self.reservations[curNode.t].add((curNode.i, curNode.j))

        self.lppath = self.lppath[::-1]

    def makeFlatlandFriendlyPath(self, agent):
        for ind in range(1, len(self.lppath)):
            if (self.lppath[ind].i == self.lppath[ind - 1].i and self.lppath[ind].j == self.lppath[ind - 1].j):
                agent_action[agent.agentId].append(4)
            elif abs(self.lppath[ind].dir - self.lppath[ind - 1].dir) % 2 == 0:
                agent_action[agent.agentId].append(2)
            elif ((self.lppath[ind - 1].dir + 1) % 4 == self.lppath[ind].dir):
                agent_action[agent.agentId].append(3)
            else:
                agent_action[agent.agentId].append(1)
        
def build():
    global agent_action
    global answer_build
    global current_pos
    answer_build = False
    agent_action = []
    current_pos = []
    global env
    map = Map()
    map.getMapOptions()

    options = Options()
    options.getOptions()

    agents = Agents()
    agents.getAgents()

    for i in range(agents.size):
        agent = Agent(i)
        agent.getAgent()
        agents.allAgents.append(agent)
        agent_action.append([])

    search = ISearch(options)
    search.startAllAgents(map, options, agents)
    
def my_controller(env, number_of_agents):
    global answer_build
    global agent_action
    global current_pos
    if (answer_build == False):
        build()
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