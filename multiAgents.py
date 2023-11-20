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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsuls = successorGameState.getCapsules()
        "*** YOUR CODE HERE ***"
        import math
        value = successorGameState.getScore()
        if len(newFood) != 0 :
            closestFood = max([-manhattanDistance(newPos, food) for food in newFood])
            value += closestFood
        if len(newCapsuls) != 0 :
            closestCapsul = max([1.5 * -manhattanDistance(newPos, capsul) for capsul in newCapsuls])
            value += closestCapsul
        closestGhost = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates if ghostState.scaredTimer == 0], default=0)
        ghostNum = len([ghost for ghost in newGhostStates if ghost.scaredTimer == 0])
        if closestGhost < 2 and ghostNum > 0:
            return -math.inf
        value += closestGhost
        value += 100 * -len(newCapsuls)
        value += 100 * -ghostNum
        value += 200 * - len(newFood)
        return value

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
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

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
        "*** YOUR CODE HERE ***"
        bestMove = self.getMaxValue(gameState, 0, self.depth)
        return bestMove[0]
    
    def getMinValue(self, gameState: GameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextFunc = self.getMaxValue if nextAgent == 0 else self.getMinValue
        if nextAgent == 0:
            depth -= 1
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        values = [nextFunc(gamestate, nextAgent, depth)[1] for gamestate in successors]
        bestValue = min(values)
        return (legalMoves[values.index(bestValue)], bestValue) 
    
    def getMaxValue(self, gameState: GameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        values = [self.getMinValue(gamestate, agentIndex + 1, depth)[1] for gamestate in successors]
        bestValue = max(values)
        return (legalMoves[values.index(bestValue)], bestValue)        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        bestMove = self.getMaxValue(gameState, 0, self.depth, -math.inf, math.inf)
        return bestMove[0]

    def getMinValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        import math
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextFunc = self.getMaxValue if nextAgent == 0 else self.getMinValue
        if nextAgent == 0:
            depth -= 1
        legalMoves = gameState.getLegalActions(agentIndex)
        bestValue = math.inf
        bestAction = 0 
        for action in legalMoves:
            newState = gameState.generateSuccessor(agentIndex, action)
            newValue = nextFunc(newState, nextAgent, depth, alpha, beta)[1]
            if bestValue > newValue:
                bestValue = newValue
                bestAction = action
            if newValue < alpha:
                return (bestAction, bestValue)
            beta = min(beta, newValue)
        return (bestAction, bestValue)
    
    def getMaxValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        import math
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        legalMoves = gameState.getLegalActions(agentIndex)
        bestValue = -math.inf
        bestAction = 0 
        for action in legalMoves:
            newState = gameState.generateSuccessor(agentIndex, action)
            newValue = self.getMinValue(newState, agentIndex + 1, depth, alpha, beta)[1]
            if bestValue < newValue:
                bestValue = newValue
                bestAction = action
            if newValue > beta:
                return (bestAction, bestValue)
            alpha = max(alpha, newValue)
        return (bestAction, bestValue)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestMove = self.getMaxValue(gameState, 0, self.depth)
        return bestMove[0]

    def getExpectedValue(self, gameState: GameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextFunc = self.getMaxValue if nextAgent == 0 else self.getExpectedValue
        if nextAgent == 0:
            depth -= 1
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        values = [nextFunc(gamestate, nextAgent, depth)[1] for gamestate in successors]
        bestValue = sum(values) / len(values)
        return (None, bestValue) 
    
    def getMaxValue(self, gameState: GameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        values = [self.getExpectedValue(gamestate, agentIndex + 1, depth)[1] for gamestate in successors]
        bestValue = max(values)
        return (legalMoves[values.index(bestValue)], bestValue) 
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsuls = successorGameState.getCapsules()
    import math
    value = successorGameState.getScore()
    if len(newFood) != 0 :
        closestFood = max([-manhattanDistance(newPos, food) for food in newFood])
        value += closestFood
    if len(newCapsuls) != 0 :
        closestCapsul = max([1.5 * -manhattanDistance(newPos, capsul) for capsul in newCapsuls])
        value += closestCapsul
    closestGhost = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates if ghostState.scaredTimer == 0], default=0)
    ghostNum = len([ghost for ghost in newGhostStates if ghost.scaredTimer == 0])
    if closestGhost < 2 and ghostNum > 0:
        return -math.inf
    value += closestGhost
    value += 100 * -len(newCapsuls)
    value += 100 * -ghostNum
    value += 200 * - len(newFood)
    return value

# Abbreviation
better = betterEvaluationFunction
