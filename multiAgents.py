# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()

        food_list = newFood.asList() # converting the remaining food positions into a list
        food_length = len(food_list) # finding how many food pellets are still left

        # only calculating food distance if there is still food remaining
        if food_length > 0: 
            food_distances = [util.manhattanDistance(newPos, food) for food in food_list] # finding the Manhattan distance from PacMan to every food pellet
            closest_food = min(food_distances) # getting the distance to the closest food pellet

            food_score = 1.5 / (closest_food + 1) # giving a larger bonus when the closest food is nearby
            score += food_score # adding the food bonus to the total score
        
        ghost_states_length = len(newGhostStates) # getting the number of ghosts that need to be considered

        # checking PacMan's distance from each ghost
        for i in range(ghost_states_length): 
            current_ghost_position = newGhostStates[i].getPosition() # getting the current ghost's position
            ghost_distance = util.manhattanDistance(newPos, current_ghost_position) # calculating the Manhattan distance between PacMan and the ghost

            current_scared_ghost = newScaredTimes[i] # getting the amount of scared time remaining for this ghost
            
            # if the ghost is scared then it is safe for PacMan to approach it
            if current_scared_ghost > 0: 
                scared_ghost_score = 2.5 / (ghost_distance + 1) # giving a larger reward when PacMan is closer to a scared ghost
                score += scared_ghost_score # adding the scared ghost reward to the total score
            else: 
                if ghost_distance <= 1: # giving a large penalty if PacMan gets directly next to a not scared ghost
                    score -= 10 
                else: 
                    score -= 1.0 / ghost_distance # giving a smaller penalty when the ghost is farther away (gets smaller as the distance from the ghost increases)

        return score # returning the final score for the action

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent and AlphaBetaAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        num_agents = gameState.getNumAgents()

        # recursively finding the minimax value of a game state
        def minimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth: # stop searching if the game is over or the depth limit is reached
                return self.evaluationFunction(state)

            next_agent = (agentIndex + 1) % num_agents # moving to the next agent after the current agent takes a turn
            next_depth = depth # keeping the same depth while the ghosts are moving

            if next_agent == 0: # increasing the depth when all ghosts have moved and we return to PacMan
                next_depth = next_depth + 1

            values = [] # storing the minimax value produced by each action

            # trying every legal action for the current agent
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action) # generating the state after the current agent takes action
                value = minimax(successor,next_agent,next_depth) # recursively finding the value of the resulting state
                values.append(value) # saving the value so the current agent can choose from them

            if agentIndex == 0: # PacMan trying to maximize the score
                return max(values)
            else: # ghosts trying to minimize the score
                return min(values)

        chosen_action = None # keeping track of PacMan best action
        chosen_value = -float('inf') # starting at negative infinity

        # trying each action PacMan can make from the starting state
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action) # generating the state after PacMan takes the action
            value = minimax(successor, 1, 0) # the ghost moves next while we are still at depth 0

            # keeping action if it has the highest minimax value so far
            if value > chosen_value:
                chosen_value = value
                chosen_action = action
        print("chosen_value: ", chosen_value) # printing the value to be checked against the expected values
        return chosen_action # returning PacMan best action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()

        def alpha_beta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth: # stop searching if the game is over or the depth limit is reached
                return self.evaluationFunction(state)
            
            next_agent = (agentIndex + 1) % num_agents # moving to the next agent after the current agent takes a turn
            next_depth = depth # keeping the same depth while the ghosts are moving
            
            if next_agent == 0: # increasing the depth when all ghosts have moved and we return to PacMan
                next_depth = next_depth + 1

            if agentIndex == 0: # Best value this node has received from its children
                best_value = -float('inf')
            else:
                best_value = float('inf')

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                child_value = alpha_beta(successor, next_agent, next_depth, alpha, beta)

                if agentIndex == 0:
                    best_value = max(best_value, child_value)
                    alpha = max(alpha, best_value)
                else:
                    best_value = min(best_value, child_value)
                    beta = min(beta, best_value)

                if alpha > beta: # prune the remaining children
                    break

            return best_value
        alpha = -float('inf')
        beta = float('inf')
        chosen_action = None
        chosen_value = -float('inf')
        for action in gameState.getLegalActions(0):
            # Pacman has moved, so ghost 1 is next
            value = alpha_beta(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if value > chosen_value:
                chosen_value = value
                chosen_action = action

            alpha = max(alpha, chosen_value)
        return chosen_action

