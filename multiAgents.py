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

        "*** YOUR CODE HERE ***"
        #start at original score
        score = successorGameState.getScore()

        #food incentive
        foodList = newFood.asList()
        if len(foodList) > 0:
            #list of distances to each food
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            minDistance = min(foodDistances)
            #incentivizes closer food
            score += 10.0 / (minDistance + 1) #avoiding division by 0


        #avoid ghosts 
        for ghostState in newGhostStates: #checking every ghost
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)

            if ghostDistance < 2:
                score -= 1000 #means ghost is right next to pacman


        #avoids pacman just staying in the corner
        if action == Directions.STOP:
            score -= 10

        return score


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
        result = self.minimax(gameState, 0, 0)
        return result[1] #action

    def minimax(self, gameState, agentIndex, depth):
        #base case
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0: #pacmans turn bc pacman is agent 0
            return self.maxValue(gameState, agentIndex, depth)
        else: #otherwise it's the ghosts' turn
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        maxScore = -9999999
        maxAction = None
        
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            #recursive call to check the ghosts move (agent 1)
            score, _ = self.minimax(successor, 1, depth)
            
            #update optimal pacman move
            if score > maxScore:
                maxScore, maxAction = score, action
                
        return maxScore, maxAction

    def minValue(self, gameState, agentIndex, depth):
        minScore = 9999999
        minAction = None
        
        #go to next ghost
        nextAgent = agentIndex + 1
        nextDepth = depth
        
        #once all ghosts checked, the next turn is pacman's
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 #no more ghosts left
            nextDepth += 1 #once all ghosts checked go down a level

        for action in gameState.getLegalActions(agentIndex):
            #getting successor state for each action
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successor, nextAgent, nextDepth)
            
            #update optimal ghost move
            if score < minScore:
                minScore, minAction = score, action
                
        return minScore, minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #init alpha and beta
        alpha = -9999999
        beta = 9999999
        result = self.minimax(gameState, 0, 0, alpha, beta)
        return result[1] #action

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        #base case
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0: #pacmans turn bc pacman is agent 0
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else: #otherwise it's the ghosts' turn
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        maxScore = -9999999
        maxAction = None
        
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            #recursive call to check the ghosts move (agent 1)
            score, _ = self.minimax(successor, 1, depth, alpha, beta)
            
            #update optimal pacman move
            if score > maxScore:
                maxScore, maxAction = score, action
            
            #PRUNING: If maxScore > beta, min will never pick it anyways
            if maxScore > beta:
                return maxScore, maxAction
            #update alpha
            alpha = max(alpha, maxScore)
                
        return maxScore, maxAction

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        minScore = 9999999
        minAction = None
        
        #go to next ghost
        nextAgent = agentIndex + 1
        nextDepth = depth
        
        #once all ghosts checked, the next turn is pacman's
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 #no more ghosts left
            nextDepth += 1 #once all ghosts checked go down a level

        for action in gameState.getLegalActions(agentIndex):
            #getting successor state for each action
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successor, nextAgent, nextDepth, alpha, beta)
            
            #update optimal ghost move
            if score < minScore:
                minScore, minAction = score, action
            
            #PRUNING: If minScore < alpha, max won't pick it anyways
            if minScore < alpha:
                return minScore, minAction
            #update beta
            beta = min(beta, minScore)
                
        return minScore, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    current_state = GameState

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # start expectimax from root (pacman = agent 0, depth 0)
        # value() returns (score, action), return action
        result = self.value(gameState, 0, 0)
        return result[1]  # best action found
    

    def value(self, gameState: GameState, agentIndex, depth):
        # base case: stop searching if game is over OR we hit max depth
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            # evaluation function just returns a score for this state
            return self.evaluationFunction(gameState), None

        # pacman = max node
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            # ghosts = chance nodes (uniform random)
            return self.exp_value(gameState, agentIndex, depth)


    def max_value(self, gameState: GameState, agentIndex, depth):
        # get legal moves for pacman
        legal_actions = gameState.getLegalActions(agentIndex)

        # remove STOP
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        max_score = float("-inf")

        # pick a default action so something is returned
        best_action = legal_actions[0] if legal_actions else Directions.STOP

        # try each possible action
        for action in legal_actions:
            succesor = gameState.generateSuccessor(agentIndex, action)

            # move to next agent (ghost)
            nextAgent = agentIndex + 1
            nextDepth = depth

            # if all agents moved, wrap back to pacman and increase depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            # recursively evaluate this branch
            score, _ = self.value(succesor, nextAgent, nextDepth)

            # keep track of best scoring action
            if score > max_score:
                max_score = score
                best_action = action

        # return best score + action that gave it
        return max_score, best_action
    

    def exp_value(self, gameState: GameState, agentIndex, depth):
        # ghosts act randomly, so we compute expected value

        legal_action = gameState.getLegalActions(agentIndex)
        score = 0

        actions = gameState.getLegalActions(agentIndex)

        # if ghost has no moves just evaluate state
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None

        # uniform probability over actions
        prob = 1.0 / len(actions)

        # go through each possible ghost move
        for action in legal_action:
            succesor = gameState.generateSuccessor(agentIndex, action)

            # move to next agent
            nextAgent = agentIndex + 1
            nextDepth = depth

            # if last ghost, go back to pacman + increase depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            # recursive call to evaluate outcome
            curr_score, curr_action = self.value(succesor, nextAgent, nextDepth)

            # expected value = sum(prob * value)
            score += prob * curr_score

        return score, None
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
