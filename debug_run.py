import heapq
import copy
import time
import numpy as np
from queue import Queue
import subprocess
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

EPS = 0.0001
INFINITY = 1000000007
SAFE_LAYER = 4
START_TIME_LIMIT = 150
REPLAN_LIMIT = 150
MAX_TIME_ONE = 300000

def getStepsToExitCell(v):
    return int(1 / v + EPS)

class Agent: # agent general and instant information
    def __init__(self, agentId, env):
        self.start_i = -1 # start (from previous reset)
        self.start_j = -1
        self.fin_i = -1 # finish
        self.fin_j = -1
        self.current_pos = 0 # current position of a personal plan
        self.actions = [] # personal plan
        self.obligations = None
        self.agentId = agentId # ID (with the same order, as flatland has)
        self.spawned = False
        self.malfunctioning = False

    def getAgent(self, env):
        if (env.agents[self.agentId].position == None):
            self.start_i = env.agents[self.agentId].initial_position[0]
            self.start_j = env.agents[self.agentId].initial_position[1]
        else:
            self.start_i = env.agents[self.agentId].position[0] # read start, finish, direction from system
            self.start_j = env.agents[self.agentId].position[1]
        
        self.fin_i = env.agents[self.agentId].target[0]
        self.fin_j = env.agents[self.agentId].target[1]

        self.dir = env.agents[self.agentId].direction
        self.stepsToExitCell = getStepsToExitCell(env.agents[self.agentId].speed_data["speed"]) # number of steps required to
                                                                                 # move to next cell

class Agents:
    def __init__(self):
        self.allAgents = [] # array of agents
        self.size = 0

    def getAgents(self, env):
        self.size = env.get_num_agents()
        if (self.allAgents == []):
            for ind in range(self.size):
                self.allAgents.append(Agent(ind, env))
                self.allAgents[ind].getAgent(env)
        else:
            for ind in range(self.size):
                self.allAgents[ind].getAgent(env)
                
    def reset_agent(self, number, new_i, new_j):
        self.allAgents[number].actions = []
        self.allAgents[number].current_pos = 0
        self.allAgents[number].start_i = new_i
        self.allAgents[number].start_j = new_j

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
        return self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.spawned == other.spawned

    def __ne__(self, other):
        return not (self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.spawned == other.spawned)

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
        
class Global_H: # advanced heuristic - shortest path from this cell to finish with no other agents
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
        else:
            return INFINITY
                

class ISearch:
    def __init__(self, env):
        self.lppath = [] # path of low-level nodes
        for ind in range(env.get_num_agents()):
            self.lppath.append([])
        self.reservations = dict() # reservated cells
        self.maxTime = 5000
        self.additional_reserve = 20

    def startallAgents(self, env, control_agent, order, time_limit, current_step): # preparations and performing A* on the first turn

        # path exists is a feedback for high-level class
        path_exists = []
        for i in range(env.get_num_agents()):
            path_exists.append(False)       

        start_time = time.time()
        for i in range(len(order)): # execute A* with every single agent with 
            agent = control_agent.allAgents[order[i]]
            path_exists[agent.agentId] = self.startSearch(agent, env, current_step)
            if (int(time.time()) - start_time > time_limit):
                break
        return path_exists

    def checkReservation(self, i, j, t): # low-level code: reservations info
        return (t, i, j) in self.reservations
    
    def get_occupator(self, i, j, t):
        if (t, i, j) in self.reservations:
            return self.reservations[(t, i, j)]
        else:
            return None

    def startSearch(self, agent, env, current_step):

        # start of A* algorithm
        startNode = agent.obligations
        finNode = Node(agent.fin_i, agent.fin_j, agent.dir)
    
        openHeap = []
        openCopy = dict()
        closed = set()

        pathFound = False

        entry = Entry(startNode, None)
        heapq.heappush(openHeap, entry)
        openCopy[(startNode.i, startNode.j, startNode.dir, startNode.t, startNode.spawned)] = (startNode.h, startNode.f)
        start_search_time = time.time()

        while (not pathFound) and len(openHeap) > 0:

            curNode = (heapq.heappop(openHeap)).priority
            
            if (curNode.t >= self.maxTime or curNode.h == INFINITY or time.time() - start_search_time >= MAX_TIME_ONE):
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
        for step in range(-SAFE_LAYER, agent.stepsToExitCell + SAFE_LAYER):
            if self.checkReservation(scNode.i, scNode.j, max(scNode.t + step, 0)) and self.get_occupator(scNode.i, scNode.j, max(scNode.t + step, 0)) != agent.agentId:
                return False
        if (self.checkReservation(scNode.i, scNode.j, scNode.t + SAFE_LAYER + agent.stepsToExitCell)):
            current_number = agent.agentId
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t + SAFE_LAYER + agent.stepsToExitCell)
            if (current_number > other_number):
                return False
        if (self.checkReservation(scNode.i, scNode.j, scNode.t - SAFE_LAYER)):
            current_number = agent.agentId
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t - SAFE_LAYER)
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
        
        if (curNode.spawned == False): # if the agents is not spawned there are two opportunities - to remain unspawned and to enter the simulation
            not_spawned = Node(curNode.i, curNode.j, curNode.dir)
            not_spawned.g = curNode.g + 1
            not_spawned.t = curNode.t + 1
            not_spawned.h = curNode.h
            not_spawned.f = curNode.f + 1
            not_spawned.spawned = False
            successors.append(not_spawned)
            
            spawned_on_this_turn = copy.deepcopy(not_spawned)
            spawned_on_this_turn.spawned = True
            spawned_on_this_turn.h -= 1
            spawned_on_this_turn.f -= 1
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

            if (not self.checkReservation(scNode.i, scNode.j, curNode.t) or not self.checkReservation(curNode.i, curNode.j, scNode.t)): # definitely no edge conflict
                successors.append(scNode)
                continue

            # we don`t catch edge conflict only if agents have the same speed (so, no changes)
            edge_conflict = ((self.get_occupator(scNode.i, scNode.j, curNode.t) == self.get_occupator(curNode.i, curNode.j, scNode.t)) and
                            (self.get_occupator(scNode.i, scNode.j, curNode.t) != agent.agentId))
            if (not edge_conflict):
                successors.append(scNode)
        return successors
    
    def delete_path(self, number):
        for ind in range(1, len(self.lppath[number])):
            curNode = self.lppath[number][ind]
            for step in range(SAFE_LAYER):
                if (curNode.t + step, curNode.i, curNode.j) in self.reservations and self.reservations[(curNode.t + step, curNode.i, curNode.j)] == number:
                    del self.reservations[(curNode.t + step, curNode.i, curNode.j)]
        self.lppath[number] = []
        
    def replan_agent(self, agent, env, current_step, calculated, start_replanning_time):
        self.delete_path(agent.agentId)
        if (agent.spawned == False):
            for step in range(current_step, agent.obligations.t):
                agent.actions.append(4)
            path_exists = self.startSearch(agent, env, current_step)
            return []
        passers_by = []
        for step in range(current_step, agent.obligations.t):
            if self.checkReservation(agent.start_i, agent.start_j, step) and self.get_occupator(agent.start_i, agent.start_j, step) != agent.agentId:
                passers_by.append(self.get_occupator(agent.start_i, agent.start_j, step))
                self.delete_path(passers_by[-1])
            self.reservations[(step, agent.start_i, agent.start_j)] = agent.agentId
            agent.actions.append(4)
            
        if (calculated >= 2):
            for step in range(agent.obligations.t, agent.obligations.t + self.additional_reserve * (calculated - 1)):
                if self.checkReservation(agent.start_i, agent.start_j, step) and self.get_occupator(agent.start_i, agent.start_j, step) != agent.agentId:
                    passers_by.append(self.get_occupator(agent.start_i, agent.start_j, step))
                    self.delete_path(passers_by[-1])
                self.reservations[(step, agent.obligations.i, agent.obligations.j)] = agent.agentId
            agent.obligations.t = agent.obligations.t + self.additional_reserve * (calculated - 1) // 2
            for step in range(self.additional_reserve * (calculated - 1) // 2):
                agent.actions.append(4)

        for step in range(agent.stepsToExitCell + SAFE_LAYER):
            if self.checkReservation(agent.obligations.i, agent.obligations.j, step + agent.obligations.t) and self.get_occupator(agent.obligations.i, agent.obligations.j, step + agent.obligations.t) != agent.agentId:
                passers_by.append(self.get_occupator(agent.obligations.i, agent.obligations.j, step + agent.obligations.t))
                self.delete_path(passers_by[-1])

        path_exists = self.startSearch(agent, env, current_step)
        while path_exists == False:
            if (time.time() - start_replanning_time >= REPLAN_LIMIT):
                break
            agent_dead = False
            for step in range(agent.obligations.t + agent.stepsToExitCell, self.maxTime):
                if self.checkReservation(agent.obligations.i, agent.obligations.j, step) and self.get_occupator(agent.obligations.i, agent.obligations.j, step) != agent.agentId:
                    passers_by.append(self.get_occupator(agent.obligations.i, agent.obligations.j, step))
                    self.delete_path(passers_by[-1])
                    break
                if step == self.maxTime - 1:
                    agent_dead = True
            if (agent_dead):
                break
            path_exists = self.startSearch(agent, env, current_step)
        return passers_by

    def makePrimaryPath(self, curNode, startNode, agent): # path of nodes

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
                    
def build_start_order(env): # custom desine of start agents order, there is only one worthwhile variant
            
    answer = []
    queue = []
    for speed_value in range(5):
        queue.append([])
    for ind in range(len(env.agents)):
        x1, y1 = env.agents[ind].initial_position
        x2, y2 = env.agents[ind].target
        potential = heuristic.get_heuristic(ind, x1, y1, env.agents[ind].direction)
        queue[getStepsToExitCell(env.agents[ind].speed_data['speed'])].append([potential, ind])
    #queue[1], queue[2], queue[3], queue[4] = queue[4], queue[3], queue[2], queue[1]
    for speed_value in range(1, 5):
        queue[speed_value].sort()
    for speed_value in range(1, 5):
        for ind in range(len(queue[speed_value])):
            answer.append(queue[speed_value][ind][1])

    return answer


class Solver:
    def __init__(self, env): # initialization of a new simulation
        self.env = env
        self.control_agent = Agents()
        self.control_agent.getAgents(env)
        self.answer_build = False
        self.search = ISearch(env)
        self.current_step = 0
        self.maxStep = 8 * (env.width + env.height + 20)
        self.prev_action = [2] * self.env.get_num_agents()
        self.current_order = build_start_order(self.env)
        self.overall_time = 0
        self.calculated = [0] * self.env.get_num_agents()
    
    def make_obligation(self, number): # in fact this is a start Node (which the agent is obligated to reach before it starts to make any decisions)
        if (self.env.agents[number].position != None):
            start_i, start_j = self.env.agents[number].position
        else:
            start_i, start_j = self.env.agents[number].initial_position
        direction = self.env.agents[number].direction
        self.control_agent.allAgents[number].obligations = Node(start_i, start_j, self.env.agents[number].direction)
        agent = self.control_agent.allAgents[number]
        self.control_agent.allAgents[number].obligations.h = heuristic.get_heuristic(agent.agentId, start_i, start_j, direction) + (not agent.spawned)
        self.control_agent.allAgents[number].obligations.spawned = agent.spawned
        self.control_agent.allAgents[number].obligations.f = self.control_agent.allAgents[number].obligations.h
        if (self.env.agents[number].speed_data['position_fraction'] == 0.0):
            self.control_agent.allAgents[number].obligations.t = self.current_step + self.env.agents[number].malfunction_data['malfunction']
        else:
            current_direction = self.env.agents[number].direction
            if (self.prev_action[number] == 1):
                current_direction -= 1
            elif (self.prev_action[number] == 3):
                current_direction += 1
            current_direction %= 4
            self.control_agent.allAgents[number].obligations.dir = current_direction
            if (current_direction == 0 and self.control_agent.allAgents[number].obligations.i > 0):
                self.control_agent.allAgents[number].obligations.i -= 1
            elif (current_direction == 1 and self.control_agent.allAgents[number].obligations.j < self.env.width - 1):
                self.control_agent.allAgents[number].obligations.j += 1
            elif (current_direction == 2 and self.control_agent.allAgents[number].obligations.i < self.env.height - 1):
                self.control_agent.allAgents[number].obligations.i += 1
            elif (current_direction == 3 and self.control_agent.allAgents[number].obligations.j > 0):
                self.control_agent.allAgents[number].obligations.j -= 1
            remain = self.env.agents[number].malfunction_data['malfunction'] + int((1 - self.env.agents[number].speed_data['position_fraction'] + EPS) / self.env.agents[number].speed_data['speed'])
            self.control_agent.allAgents[number].obligations.t = self.current_step + remain
            
    def set_obligations(self):
        for ind in range(self.env.get_num_agents()):
            self.make_obligation(ind)
            if (self.env.agents[ind].position != None):
                self.control_agent.allAgents[ind].start_i, self.control_agent.allAgents[ind].start_j = self.env.agents[ind].position
            else:
                self.control_agent.allAgents[ind].start_i, self.control_agent.allAgents[ind].start_j = self.env.agents[ind].initial_position
        
    def build_on_the_start(self):
        self.set_obligations()
        path_exists = self.search.startallAgents(self.env, self.control_agent, self.current_order, START_TIME_LIMIT, self.current_step)
        new_order = []
        for ind in range(len(self.current_order)):
            if (path_exists[self.current_order[ind]] == True):
                new_order.append(self.current_order[ind])
        self.current_order = copy.deepcopy(new_order)
        self.answer_build = True

    def print_step(self):
        _action = {}
        for ind in range(self.env.get_num_agents()):
            agent = self.control_agent.allAgents[ind]
            position = self.env.agents[ind].position
            if agent.current_pos < len(agent.actions):
                if (agent.actions[agent.current_pos] != 4):
                    self.prev_action[ind] = agent.actions[agent.current_pos]
                    agent.spawned = True
                _action[ind] = agent.actions[agent.current_pos]
                self.control_agent.allAgents[ind].current_pos += 1
        self.current_step += 1
        return _action
    
    def update_malfunctions(self):
        self.calculated = [0] * self.env.get_num_agents()
        for ind in range(self.env.get_num_agents()):
            if (self.env.agents[ind].malfunction_data['malfunction'] > 1 and self.control_agent.allAgents[ind].malfunctioning == False):
                replanning_queue = []
                replanning_queue.append(ind)
                start_replanning_time = time.time()
                pos = 0
                while pos < len(replanning_queue):
                    if time.time() - start_replanning_time >= REPLAN_LIMIT:
                        break
                    current = replanning_queue[pos]
                    if (self.env.agents[current].position == None):
                        if self.control_agent.allAgents[current].spawned == True:
                            pos += 1
                            continue
                        self.control_agent.reset_agent(current, self.env.agents[current].initial_position[0], self.env.agents[current].initial_position[1])
                    else:
                        self.control_agent.reset_agent(current, self.env.agents[current].position[0], self.env.agents[current].position[1])
                    self.make_obligation(current)
                    additional = self.search.replan_agent(self.control_agent.allAgents[current], self.env, self.current_step, self.calculated[current], start_replanning_time)
                    self.calculated[current] += 1
                    for i in range(len(additional)):
                        replanning_queue.append(additional[i])
                    pos += 1
        for ind in range(self.env.get_num_agents()):
            self.control_agent.allAgents[ind].malfunctioning = (self.env.agents[ind].malfunction_data['malfunction'] > 1)

def my_controller(env, path_finder):
    if (path_finder.answer_build == False and path_finder.current_step == 1):
        path_finder.build_on_the_start()
    elif path_finder.current_step != 0 and path_finder.overall_time <= 800:
        path_finder.update_malfunctions()
    return path_finder.print_step()

width = 40  # With of map
height = 40  # Height of map
nr_trains = 75  # Number of trains that have an assigned task in the env
cities_in_map = 4  # Number of cities where agents can start or end
seed = 34  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )

# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.


stochastic_data = MalfunctionParameters(malfunction_rate=400,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )


# Custom observation builder without predictor
observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
# observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
local_env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              remove_agents_at_target=True)
local_env.reset()


# Initiate the renderer
env_renderer = RenderTool(local_env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800)  # Adjust these parameters to fit your resolution

print("Start episode...")
# Reset environment and get initial observations for all agents
start_reset = time.time()
obs = local_env.reset()
end_reset = time.time()

print(end_reset - start_reset)
print(local_env.get_num_agents(), )
# Reset the rendering sytem
env_renderer.reset()
env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
heuristic = Global_H(local_env)
path_finder = Solver(local_env)

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository

c = int(time.time()) % 100
file = open("input.in", "w")
file.write(str(c) + "\n")
file.close()
subprocess.run("1.exe")
file = open("output.out", "r")
info = file.readline().split()
summary, source = int(info[0]), int(info[1])
print(summary, source)

score = 0
start_time = time.time()
# Run episode
frame_step = 0
for step in range(16000):
    print("========== step number ", step, " ==========", sep = "")
        
    # Chose an action for each agent in the environment
    action_dict = my_controller(local_env, path_finder)

    next_obs, all_rewards, done, info = local_env.step(action_dict)
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    frame_step += 1
    # Update replay buffer and train agent
    for a in range(local_env.get_num_agents()):
        score += all_rewards[a]

    obs = next_obs.copy()
    if done['__all__']:
        break


print('Episode: Steps {}\t Score = {}'.format(step, score))