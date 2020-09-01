import numpy as np
import math
from enum import IntEnum
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
import itertools


class AgentStatus(IntEnum):
    OUTSIDE = 0
    MOVING = 1
    STATIONARY = 2
    DONE = 3


class Node:
  def __init__(self, state, parent=None, actions=None):
    self.state = state
    self.parent = parent
    self.actions = actions
    self.f = math.inf
    self.g = math.inf
    self.h = math.inf
  def __eq__(self, other):
    return self.state == other.state

def check_action(agent, action, state, env: RailEnv):

  x = agent['x']
  y = agent['y']
  direction = agent['direction']
  status = agent['status']

  if status == AgentStatus.DONE:
    if action == 0:
      return  {
      'x': x,
      'y': y,
      'direction': direction,
      'status': status}
    else:
      return False

  if status == AgentStatus.OUTSIDE:
    if action == RailEnvActions.DO_NOTHING:
      return  {
      'x': x, 
      'y': y,
      'direction': direction,
      'status': status}
    elif action == RailEnvActions.MOVE_FORWARD: 
      if is_cell_free((y,x), state):
        return {
      'x': x,
      'y': y,
      'direction': direction,
      'status': AgentStatus.STATIONARY}
      else:
        return False
    else:
      return False

  if action == RailEnvActions.DO_NOTHING:
    if status == AgentStatus.MOVING:
      return False
    else:
      return  {
      'x': x,
      'y': y,
      'direction': direction,
      'status': status}

  if action == RailEnvActions.STOP_MOVING:
    if status == AgentStatus.STATIONARY:
      return False
    else:
      return {
        'x': x,
        'y': y,
        'direction': direction,
        'status': AgentStatus.STATIONARY}

  transition_valid = None
  possible_transitions = env.rail.get_transitions(y, x, direction)
  num_transitions = np.count_nonzero(possible_transitions)

  new_direction = direction
  if action == RailEnvActions.MOVE_LEFT:
      new_direction = direction - 1
      if num_transitions <= 1:
          transition_valid = False

  elif action == RailEnvActions.MOVE_RIGHT:
      new_direction = direction + 1
      if num_transitions <= 1:
          transition_valid = False

  new_direction %= 4

  if action == RailEnvActions.MOVE_FORWARD and num_transitions == 1:
      # - dead-end, straight line or curved line;
      # new_direction will be the only valid transition
      # - take only available transition
      new_direction = np.argmax(possible_transitions)
      transition_valid = True

  new_position = get_new_position((y, x), new_direction)

  new_cell_valid = (
      np.array_equal(  # Check the new position is still in the grid
          new_position,
          np.clip(new_position, [0, 0], [env.rail.height - 1, env.rail.width - 1]))
      and  # check the new position has some transitions (ie is not an empty cell)
      env.rail.get_full_transitions(*new_position) > 0)

  if transition_valid is None:
              transition_valid = env.rail.get_transition(
                  (y,x, direction),
                  new_direction)

  if new_cell_valid:
      # Check the new position is not the same as any of the existing agent positions
      # (including itself, for simplicity, since it is moving)
      cell_free = is_cell_free(new_position, state)
  else:
      # if new cell is outside of scene -> cell_free is False
      cell_free = False
  if cell_free and transition_valid:
    return  {
    'x': new_position[1],
    'y': new_position[0],
    'direction': new_direction,
    'status': AgentStatus.MOVING}
  else:
    return False

def is_cell_free(pos, state):
  x, y = pos[1], pos[0]
  for a in state:
    if a['x'] == x and a['y'] == y and a['status'] != AgentStatus.DONE and a['status'] != AgentStatus.OUTSIDE:
      return False
  return True

def get_new_position(position, movement):
    """ Utility function that converts a compass movement over a 2D grid to new positions (r, c). """
    if movement == 0:
        return (position[0] - 1, position[1])
    elif movement == 1:
        return (position[0], position[1] + 1)
    elif movement == 2:
        return (position[0] + 1, position[1])
    elif movement == 3:
        return (position[0], position[1] - 1)

def get_neighbours(node: Node, env: RailEnv):
  n = len(env.get_agent_handles())
  possibilities = []
  goal = get_trains_goal_state(env)

  def get_neighbours_for_agent(state, actions, i):
    if i == n:
      possibilities.append((state, actions))
      return
    agent = node.state[i]
    for action in range(0, 5):
      ok = check_action(agent, action, state, env)
      if not ok:
        continue
      else:
        if get_manhattan_distance(ok, goal[i]) == 0:
          ok['status'] = AgentStatus.DONE
        s = state[:]
        a = actions[:]
        s[i] = ok
        a[i] = action
        get_neighbours_for_agent(s, a, i+1)

  get_neighbours_for_agent(node.state, [None]*n, 0)
  neighbours = []
  for states, state_actions in possibilities:
    neighbours.append(Node(states, node, state_actions))
  return neighbours

def get_manhattan_distance(posA, posB):
  return (abs(posA['x'] - posB['x']) + abs(posA['y'] - posB['y']))

def get_max_manhattan_distance(stateA, stateB):
  return max(get_manhattan_distance(stateA[i], stateB[i]) for i in range(len(stateA)))

def get_trains_initial_state(env: RailEnv): 
  return [ 
    {'x': a.initial_position[1],
    'y': a.initial_position[0],
    'direction': a.direction,
    'status': AgentStatus.OUTSIDE
   } for a in env.agents]

def get_trains_goal_state(env: RailEnv):
  return [
    {'x': a.target[1],
    'y': a.target[0],
    'direction': None,
    'status': AgentStatus.DONE
   } for a in env.agents]


def search(env: RailEnv):
  '''
  Runs multi-agent A* search.
  Uses the maximum manhattan distance across all agents as the heuristic.
  The tie-breaker used is the h-value of the node, as we want to search the nodes with 
  a lower h-value (closer to the goal) first when we encounter multiple with the same f-value.
  '''
  schedule = []

  initial_node = Node(get_trains_initial_state(env))
  initial_node.g = 0
  final_node = Node(get_trains_goal_state(env))
  current_node = None
  open = []
  open.append(initial_node)
  closed = []
  solved = False

  while len(open):
    open.sort(key=lambda x: (x.f, x.h))
    current_node = open.pop(0)
    if current_node.h == 0:
      solved = True
      break
    closed.append(current_node)

    for neighbour in get_neighbours(current_node, env):
      if neighbour in closed:
        continue
      g = current_node.g + 1
      h = get_max_manhattan_distance(neighbour.state, final_node.state)
      neighbour.g = g
      neighbour.h = h
      neighbour.f = g+h

      if neighbour not in open:
        open.append(neighbour)
      else:
        for i, node in enumerate(open):
          if node == neighbour and node.g > neighbour.g:
            open[i] = neighbour
  if solved:
    while current_node.parent:
      schedule.append(dict(zip(env.get_agent_handles(),current_node.actions)))
      current_node = current_node.parent
  
  return schedule[::-1]