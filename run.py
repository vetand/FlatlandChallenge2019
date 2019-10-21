from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import EnvAgentStatic, EnvAgent, RailAgentStatus
import numpy as np
import heapq
import copy
import time
import numpy as np
from queue import Queue
#import random
#random.seed(47)
EPS = 0.001
INFINITY = 1000000007
ADD_GROUP_NUMBER = 5


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

class Agent: # agent general and instant information
    def __init__(self, agentId):
        self.start_i = -1 # start (from previous reset)
        self.start_j = -1
        self.fin_i = -1 # finish
        self.fin_j = -1
        self.current_pos = 0 # current position of a personal plan
        self.actions = [] # personal plan
        self.obligations = 0 # in case of stochastic event or mistake the system will automatically
                             # perform some steps, so we must take it into account;
                             # right now this is delay time in start cell
        self.agentId = agentId # ID (with the same order, as flatland has)
        self.malfunction = False # if malfunction began just now, so we need a reset
        self.spawned = False
        self.wait_after_spawn = 0

    def getAgent(self, env, type_used):
        if (env.agents[self.agentId].position == None):
            self.start_i = env.agents[self.agentId].initial_position[0]
            self.start_j = env.agents[self.agentId].initial_position[1]
        else:
            self.start_i = env.agents[self.agentId].position[0] # read start, finish, direction from system
            self.start_j = env.agents[self.agentId].position[1]
        
        self.fin_i = env.agents[self.agentId].target[0]
        self.fin_j = env.agents[self.agentId].target[1]

        self.dir = env.agents[self.agentId].direction
        self.stepsToExitCell = int(1 / env.agents[self.agentId].speed_data["speed"] + EPS) # number of steps required to
                                                                                 # move to next cell

class Agents: # agent`s rapport between themselves
    def __init__(self):
        self.allAgents = [] # array of agents
        self.size = 0

    def getAgents(self, env, type_used):
        self.size = env.get_num_agents()
        if (type_used == "first"):
            for ind in range(self.size):
                self.allAgents.append(Agent(ind))
                self.allAgents[ind].getAgent(env, type_used)
        else:
            for ind in range(self.size):
                self.allAgents[ind].getAgent(env, type_used)

class Node: # low-level code: Node class of a search process (no changes)
    def __init__(self, i, j, dir):
        self.i = i
        self.j = j
        self.t = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.dir = dir
        self.spawned = False
        self.parent = None

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.t == other.t and self.spawned == other.spawned

    def __ne__(self, other):
        return not (self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.t == other.t and self.spawned == other.spawned)

    def __lt__(self, other):
        if (self.i == other.i):
                if (self.j == other.j):
                        if (self.dir == other.dir):
                            if (self.t == other.t):
                                return self.spawned < other.spawned
                            return self.t < other.t
                        return self.dir < other.dir
                return self.j < other.j
        return self.i < other.i

    def __hash__(self):
        return hash((self.i, self.j, self.t, self.dir, self.spawned))

class Entry: # low-level code: Entry class for priority queue in A* (no changes)
    def __init__(self, priority, value):
        self.priority = priority
        self.value = value

    def cmp(self, one, two):
        if (one.f == two.f):
            if (one.g == two.g):
                if (one.spawned == two.spawned):
                    return one < two
                return one.spawned < two.spawned
            else:
                return one.g > two.g
        return one.f < two.f

    def __cmp__(self, other):
        return self.cmp(self.priority, other.priority)

    def __lt__(self, other):
        return self.cmp(self.priority, other.priority)

    def __eq__(self, other):
        return self.priority == other.priority

def global_heuristic_simple(x1, y1, x2, y2): # simple Manhattan heuristic
    return abs(x2 - x1) + abs(y2 - y1)

class Node_h: # low-level code: Node class for advanced heuristic finder
    def __init__(self, x, y, dir, time):
        self.i = x
        self.j = y
        self.dir = dir
        self.time = time
        
class global_H: # advanced heuristic - shortest path from this cell to finish with no other agents
    def __init__(self, env):
        self.database = dict()
        self.env = env
        for ind in range(env.get_num_agents()):
            self.start_agent(ind)
        
    def get_neighbors(self, curNode): # actually, the same procedure with Isearch class
        available = self.env.rail.get_transitions(*[curNode.i, curNode.j], curNode.dir)
        answer = []
        if (available[0] == True):
            answer.append(Node_h(curNode.i - 1, curNode.j, 0, curNode.time + 1))
        if (available[1] == True):
            answer.append(Node_h(curNode.i, curNode.j + 1, 1, curNode.time + 1))
        if (available[2] == True):
            answer.append(Node_h(curNode.i + 1, curNode.j, 2, curNode.time + 1))
        if (available[3] == True):
            answer.append(Node_h(curNode.i, curNode.j - 1, 3, curNode.time + 1))
        return answer
        
    def get_dir(self, position): # get a direction (orientation) of a trainstation
        answer = []
        for dest in range(4):
            available = self.env.rail.get_transitions(*position, dest)
            if (sum(available) > 0):
                answer.append(dest)
        return answer
        
    def start_agent(self, number): # perform A* to count heuristic
        correct_dir = self.get_dir(self.env.agents[number].target)
        queue = Queue()
        for dest in correct_dir:
            start = Node_h(self.env.agents[number].target[0], self.env.agents[number].target[1], dest, 0)
            queue.put(start)
        while (not queue.empty()):
            current = queue.get()
            candidates = self.get_neighbors(current)
            for node in candidates:
                if ((number, current.i, current.j, (node.dir + 2) % 4) not in self.database):
                    self.database[(number, current.i, current.j, (node.dir + 2) % 4)] = current.time
                    queue.put(node)
                    
    def get_heuristic(self, agentId, x, y, dir): # output
        if (agentId, x, y, dir) in self.database:
            return self.database[(agentId, x, y, dir)]

class ISearch:
    def __init__(self, env):
        self.lppath = [] # path of low-level nodes
        for ind in range(env.get_num_agents()):
            self.lppath.append([])
        self.reservations = dict() # reservated cells
        self.maxTime = 1000

    def startallAgents(self, env, control_agent, order, moving_numbers): # preparations and performing A* 
                                                         # search for every single agent
        # preparations step : match required reservations
        
        for ind in range(env.get_num_agents()):
            if order[ind] not in moving_numbers:
                continue
            agent = control_agent.allAgents[order[ind]]
            if (agent.spawned == True):
                for step in range(agent.obligations):
                    agent.actions.append(4)
                    self.reservations[(step, agent.start_i, agent.start_j)] = agent.agentId
                for step in range(agent.stepsToExitCell):
                    self.reservations[(step + agent.obligations, agent.start_i, agent.start_j)] = agent.agentId
    
        # path exists is a feedback for high-level class
        path_exists = []
        for i in range(env.get_num_agents()):
            path_exists.append([])

        for i in range(env.get_num_agents()): # execute A* with every single agent with 
            if (order[i] not in moving_numbers):
                path_exists[order[i]] = True
                continue
            agent = control_agent.allAgents[order[i]]
            if (agent.spawned == True and env.agents[agent.agentId].position == None):
                path_exists[agent.agentId] = True
            else:
                path_exists[agent.agentId] = self.startSearch(agent, env)
        return path_exists

    def checkReservation(self, i, j, t): # low-level code: reservations info
        return ((t, i, j) in self.reservations)
    
    def get_occupator(self, i, j, t):
        return self.reservations[(t, i, j)]

    def check_line(self, left_bound, right_bound, i, j):
        for step in range(max(left_bound, 0), right_bound + 1):
            if (self.checkReservation(i, j, step) == True):
                return False
        return True

    def startSearch(self, agent, env):

        # start of A* algorithm
        startNode = Node(agent.start_i, agent.start_j, agent.dir)
        finNode = Node(agent.fin_i, agent.fin_j, agent.dir)
        
        startNode.h = heuristic.get_heuristic(agent.agentId, startNode.i, startNode.j, startNode.dir)
        startNode.f = startNode.g + startNode.h
        startNode.t = agent.obligations
        startNode.spawned = agent.spawned
    
        openHeap = []
        openCopy = dict()
        closed = set()

        pathFound = False

        entry = Entry(startNode, None)
        heapq.heappush(openHeap, entry)
        openCopy[(startNode.i, startNode.j, startNode.dir, startNode.t, startNode.spawned)] = (startNode.h, startNode.f)

        while (not pathFound) and len(openHeap) > 0:

            curNode = (heapq.heappop(openHeap)).priority
            
            if (curNode.t == self.maxTime):
                break

            if (curNode.i == finNode.i and curNode.j == finNode.j):
                finNode = curNode
                pathFound = True
                break

            else:
                openCopy.pop((curNode.i, curNode.j, curNode.dir, curNode.t, curNode.spawned))
                closed.add(curNode)

                successors = self.findSuccessors(curNode, agent, env)
                for i in range(len(successors)):
                    scNode = successors[i]
                    
                    foundInClosed = False
                    if (scNode in closed):
                        foundInClosed = True

                    if ((not foundInClosed) and curNode.g + 1 <= scNode.g):

                        scNode.parent = curNode

                        foundInOpen = False
                        if ((scNode.i, scNode.j, scNode.dir, scNode.t, scNode.spawned) in openCopy.keys()):
                            check = openCopy.get((scNode.i, scNode.j, scNode.dir, scNode.t, scNode.spawned))
                            checkH = check[0]
                            checkF = check[1]

                            foundInOpen = True

                        if (foundInOpen == True and checkF > scNode.f):
                            checkNode = Node(scNode.i, scNode.j, scNode.dir)
                            checkNode.t = scNode.t
                            checkNode.spawned = scNode.spawned
                            entry = Entry(checkNode, None)

                            openHeap.remove(entry)
                            heapq.heapify(openHeap)

                            entry = Entry(scNode, None)
                            heapq.heappush(openHeap, entry)

                            openCopy[(scNode.i, scNode.j, scNode.dir, scNode.t, scNode.spawned)] = (scNode.h, scNode.f)

                        if (foundInOpen == False):
                            entry = Entry(scNode, None)
                            heapq.heappush(openHeap, entry)
                            openCopy[(scNode.i, scNode.j, scNode.dir, scNode.t, scNode.spawned)] = (scNode.h, scNode.f)
        

        if pathFound:
            self.makePrimaryPath(finNode, startNode, agent)
            self.makeFlatlandFriendlyPath(agent)
            return True
        else:
            return False
        
    def correct_point(self, scNode, agent):
        for step in range(agent.stepsToExitCell):
            if self.checkReservation(scNode.i, scNode.j, scNode.t + step) and self.get_occupator(scNode.i, scNode.j, scNode.t + step) != agent.agentId:
                return False
        if (self.checkReservation(scNode.i, scNode.j, scNode.t + agent.stepsToExitCell)):
            current_number = agent.agentId
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t + agent.stepsToExitCell)
            if (current_number > other_number):
                return False
        if (self.checkReservation(scNode.i, scNode.j, scNode.t - 1)):
            current_number = agent.agentId
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t - 1)
            if (current_number < other_number):
                return False
        return True

    def findSuccessors(self, curNode, agent, env): # find neighbors of current cell, which we are able to visit
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
        
        if (curNode.spawned == False):
            not_spawned = Node(curNode.i, curNode.j, curNode.dir)
            not_spawned.g = curNode.g + 1
            not_spawned.t = curNode.t + 1
            not_spawned.h = curNode.h
            not_spawned.f = curNode.f + 1
            not_spawned.spawned = False
            successors.append(not_spawned)
            
            spawned_on_this_turn = copy.deepcopy(not_spawned)
            spawned_on_this_turn.spawned = True
            if (self.correct_point(spawned_on_this_turn, agent) == True):
                successors.append(spawned_on_this_turn)
            return successors
        
        for scNode in inter_answer:
            scNode.g = curNode.g + 1
            scNode.h = heuristic.get_heuristic(agent.agentId, scNode.i, scNode.j, scNode.dir)
            scNode.f = scNode.g + scNode.h
            scNode.spawned = True

            if scNode.i == curNode.i and scNode.j == curNode.j:
                scNode.t = curNode.t + 1
            else:
                scNode.t = curNode.t + agent.stepsToExitCell

            if not self.correct_point(scNode, agent):
                continue

            if (not self.checkReservation(scNode.i, scNode.j, curNode.t) or not self.checkReservation(curNode.i, curNode.j, scNode.t)):
                successors.append(scNode)
                continue
            # we don`t catch edge conflict only if agents have the same speed (so, no changes)
            edge_conflict = ((self.get_occupator(scNode.i, scNode.j, curNode.t) == self.get_occupator(curNode.i, curNode.j, scNode.t)) and 
                            (not self.get_occupator(curNode.i, curNode.j, scNode.t) == self.get_occupator(curNode.i, curNode.j, curNode.t)))
            if (not edge_conflict):
                successors.append(scNode)
        return successors

    def makePrimaryPath(self, curNode, startNode, agent):

        wait_action = False
        while curNode != startNode:
            self.lppath[agent.agentId].append(curNode)
            if not wait_action:
                for step in range(agent.stepsToExitCell):
                    self.reservations[(curNode.t + step, curNode.i, curNode.j)] = agent.agentId
            elif curNode.spawned == True:
                self.reservations[(curNode.t, curNode.i, curNode.j)] = agent.agentId
            if curNode.i == curNode.parent.i and curNode.j == curNode.parent.j:
                wait_action = True
            else:
                wait_action = False
            curNode = curNode.parent

        self.lppath[agent.agentId].append(curNode)
        if not wait_action:
            for step in range(agent.stepsToExitCell):
                self.reservations[(curNode.t + step, curNode.i, curNode.j)] = agent.agentId
        elif curNode.spawned == True:
            self.reservations[(curNode.t, curNode.i, curNode.j)] = agent.agentId

        self.lppath[agent.agentId] = self.lppath[agent.agentId][::-1]

    def makeFlatlandFriendlyPath(self, agent):
        for ind in range(1, len(self.lppath[agent.agentId])):
            if (self.lppath[agent.agentId][ind].i == self.lppath[agent.agentId][ind - 1].i and self.lppath[agent.agentId][ind].j == self.lppath[agent.agentId][ind - 1].j):
                if (self.lppath[agent.agentId][ind - 1].spawned == False and self.lppath[agent.agentId][ind].spawned == True):
                    agent.actions.append(2)
                else:
                    agent.actions.append(4)
            elif abs(self.lppath[agent.agentId][ind].dir - self.lppath[agent.agentId][ind - 1].dir) % 2 == 0:
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(2)
            elif ((self.lppath[agent.agentId][ind - 1].dir + 1) % 4 == self.lppath[agent.agentId][ind].dir):
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(3)
            else:
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(1)
                    
def build_start_order(env, type): # custom desine of start agents order
    answer = []
    if (type == "as usual"):
        for i in range(len(env.agents)):
            answer.append(i)

    if (type == "reversed"):
        for i in range(len(env.agents) - 1, -1, -1):
            answer.append(i)

    if (type == "scientific"):
        queue = []
        for ind in range(len(env.agents)):
            x1, y1 = env.agents[ind].initial_position
            x2, y2 = env.agents[ind].target
            potential = global_heuristic(x1, y1, x2, y2)
            queue.append([potential, ind])
        queue.sort()
        for i in range(len(env.agents)):
            answer.append(queue[i][1])

    if (type == "random"):
        queue = []
        for ind in range(len(env.agents)):
            queue.append(ind)
        random.shuffle(queue)
        for i in range(len(env.agents)):
            answer.append(queue[i][1])

    if (type == "my"):
        queue = []
        for ind in range(len(env.agents)):
            x1, y1 = env.agents[ind].initial_position
            x2, y2 = env.agents[ind].target
            potential = heuristic.get_heuristic(ind, x1, y1, env.agents[ind].direction)
            queue.append([potential, ind])
        queue.sort()
        for i in range(len(env.agents)):
            answer.append(queue[i][1])
            
    if (type == "speed_also"):
        queue_1 = []
        queue_2 = []
        for ind in range(len(env.agents)):
            x1, y1 = env.agents[ind].initial_position
            x2, y2 = env.agents[ind].target
            potential = heuristic.get_heuristic(ind, x1, y1, env.agents[ind].direction) / env.agents[ind].speed_data['speed']
            if (env.agents[ind].malfunction_data["malfunction"] > 1):
                queue_2.append([potential, ind])
            else:
                queue_1.append([potential, ind])
        queue_1.sort()
        queue_2.sort()
        for i in range(len(queue_2)):
            answer.append(queue_2[i][1])
        for i in range(len(queue_1)):
            answer.append(queue_1[i][1])
    return answer


class submission:
    def __init__(self, env): # initialization of a new simulation
        self.env = env
        self.control_agent = Agents()
        self.control_agent.getAgents(env, "first")
        self.answer_build = False
        self.search = ISearch(env)
        self.current_order = build_start_order(env, "speed_also")
        self.stoppers = set()
        self.no_malfunction = False
        self.next_agent_add = INFINITY
        self.moving_numbers = set()
        self.done_numbers = set()
        self.capacity = (self.env.height + self.env.width) // 5  # we don`t want to overcharge the map, so we set the maximum number of trains in elaluation
        self.current_capacity = 0
          
    def flush_actions(self):
        for ind in range(self.env.get_num_agents()):
            self.control_agent.allAgents[ind].current_pos = 0
            self.control_agent.allAgents[ind].actions = []
        
    def build(self): # if we need to build a new paths
        for attempt in range(10): # we can change number of attempts in future
            path_exists = self.build_with_order(self.current_order)
            #print(attempt, self.current_order, path_exists)
            new_order = []
            answer_ready = True
            for ind in range(self.env.get_num_agents()):
                if (path_exists[self.current_order[ind]] == False):
                    new_order.append(self.current_order[ind])
                    answer_ready = False
            if (answer_ready):
                break
            #random.shuffle(new_order)
            for ind in range(self.env.get_num_agents()):
                if (path_exists[self.current_order[ind]] == True):
                    new_order.append(self.current_order[ind])
            self.current_order = copy.deepcopy(new_order) # at the beginning of new order are only agents
                                                          # which didn`t manage to build their paths
        self.answer_build = True
        
    def build_with_order(self, order): # try to build a paths with this agents order
        self.flush_actions()
        self.search = ISearch(self.env)
        path_exists = self.search.startallAgents(self.env, self.control_agent, order, self.moving_numbers)
        for ind in range(self.env.get_num_agents()):
            waiting_time = self.env.agents[ind].malfunction_data['malfunction'] - 1
            self.control_agent.allAgents[ind].obligations = max(waiting_time, 0)
        return path_exists
    
    def check_expected_next(self, agent): # check if the next cell is occupied
        if agent.current_pos + 1 >= len(self.search.lppath[agent.agentId]):
            return True
        nextNode = self.search.lppath[agent.agentId][agent.current_pos + 1]
        position = (nextNode.i, nextNode.j)
        if position in self.stoppers:
            return False
        else:
            return True
        
    def get_nearest_malfunction(self):
        minimum = INFINITY
        for ind in range(self.env.get_num_agents()):
            next_malfunction = self.env.agents[ind].malfunction_data['next_malfunction'] + self.env.agents[ind].malfunction_data['malfunction']
            if self.env.agents[ind].malfunction_data['malfunction_rate'] == 0.0 or self.env.agents[ind].position == None: # this agent can`t be broken
                continue
            minimum = min(minimum, next_malfunction)
        return minimum
    
    def update_malfunctions_before(self): # find agents which won`t be able to complete the next turn
        steps_remain = self.get_nearest_malfunction()
        self.next_agent_add -= 1
        for ind in range(self.env.get_num_agents()):
            if min(steps_remain, self.next_agent_add) <= self.control_agent.allAgents[ind].stepsToExitCell and self.env.agents[ind].speed_data['position_fraction'] == 0.0 and not self.no_malfunction:
                self.stoppers.add(self.env.agents[ind].position)
        # maybe other agents need to stop
        for i in range(10): # I hope there won`t be more than 10 trains in a row
            for ind in range(self.env.get_num_agents()):
                agent = self.control_agent.allAgents[ind]
                if not self.check_expected_next(agent):
                    self.stoppers.add(self.env.agents[ind].position)
                    
    def print_step(self):
        _action = {}
        for ind in range(self.env.get_num_agents()):
            if (ind not in self.moving_numbers):
                continue
            agent = self.control_agent.allAgents[ind]
            position = self.env.agents[ind].position
            if agent.current_pos < len(agent.actions):
                if position not in self.stoppers:
                    if (agent.actions[agent.current_pos] != 4):
                        agent.spawned = True
                    _action[ind] = agent.actions[agent.current_pos]
                    self.control_agent.allAgents[ind].current_pos += 1
                else:
                    _action[ind] = 4 # the next agent`s cell is occupied or malfunction
        return _action
    
    def update_malfunctions_after(self): # we should decide if we need a re-plan due to a new malfunction
                                         # and update some info
        self.stoppers = set()
        need_reset = (self.next_agent_add == 0)
        for ind in range(self.env.get_num_agents()):
            if (ind not in self.moving_numbers):
                continue
            agent = self.control_agent.allAgents[ind]
            if self.env.agents[ind].malfunction_data['malfunction'] > 1 and not agent.malfunction:
                agent.malfunction = True
                need_reset = True
            if self.env.agents[ind].malfunction_data['malfunction'] <= 1:
                agent.malfunction = False
            if self.env.agents[ind].position == None:
                agent_position = self.env.agents[ind].initial_position
            else:
                agent_position = self.env.agents[ind].position
            current_position = Node(agent_position[0], 
                                    agent_position[1],
                                    self.env.agents[ind].direction )
            waiting_time = self.env.agents[ind].malfunction_data['malfunction'] - 1
            self.control_agent.allAgents[ind].obligations = max(waiting_time, 0)
        if need_reset:
            self.reset()
            
    def reset(self):
        self.control_agent.getAgents(self.env, "second")
        self.build()
        self.next_agent_add = INFINITY
        
    def add_agents(self):
        if (self.next_agent_add <= 3):
            return
        usual_adding = (self.capacity - len(self.moving_numbers) >= ADD_GROUP_NUMBER)
        last_adding = (len(self.moving_numbers) + len(self.done_numbers) + ADD_GROUP_NUMBER >= self.env.get_num_agents()) and (self.env.get_num_agents() - len(self.done_numbers) <= self.capacity) 
        if (not usual_adding and not last_adding) or (len(self.moving_numbers) + len(self.done_numbers) == self.env.get_num_agents()):
            return
        for ind in range(self.env.get_num_agents()):
            if (self.current_capacity == self.capacity):
                break
            if (self.current_order[ind] not in self.moving_numbers and self.current_order[ind] not in self.done_numbers):
                self.moving_numbers.add(self.current_order[ind])
                self.current_capacity += 1
        self.next_agent_add = 3
        
    def remove_agents(self):
        for ind in range(self.env.get_num_agents()):
            if (ind in self.moving_numbers and self.env.agents[ind].status == RailAgentStatus.DONE_REMOVED):
                self.current_capacity -= 1
                self.done_numbers.add(ind)
                self.moving_numbers.remove(ind)
            
            
def my_controller(env, path_finder):
    path_finder.remove_agents()
    path_finder.add_agents()
    path_finder.update_malfunctions_after()
    if (path_finder.answer_build == False):
        path_finder.build()
    path_finder.update_malfunctions_before()
    return path_finder.print_step()

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
    observation, info = remote_client.env_create(
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

    heuristic = global_H(local_env)
    path_finder = submission(local_env)

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
        action = my_controller(local_env, path_finder)

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