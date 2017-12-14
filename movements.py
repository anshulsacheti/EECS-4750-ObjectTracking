
import numpy as np
import time
from enum import Enum

class Directions(Enum):
	UP	= 0
	DOWN	= 1
	LEFT	= 2
	RIGHT	= 3
	NONE	= 4

# Example
# frameCompare([0,1,0,2,0,3,1,3,2,3,3,3,2,3,1,3,0,3,0,2,0,1])
# Ouput 
# [<Directions.DOWN: 1>, <Directions.DOWN: 1>, <Directions.RIGHT: 3>, <Directions.RIGHT: 3>, 
# <Directions.RIGHT: 3>, <Directions.LEFT: 2>, <Directions.LEFT: 2>, <Directions.LEFT: 2>, 
# <Directions.UP: 0>, <Directions.UP: 0>]

def frameCompare(originPoints):

  movements = []
  movementsInt = []

  for i in range(0, len(originPoints) - 2, 2):
    xMovement = originPoints[i + 2] - originPoints[i]
    yMovement = originPoints[i + 3] - originPoints[i + 1]
    
    if (xMovement > 0):
      movements.append(Directions.RIGHT)
      movementsInt.append(3)
    elif (xMovement < 0):
      movements.append(Directions.LEFT)
      movementsInt.append(2)
    elif (yMovement < 0):
      movements.append(Directions.UP)
      movementsInt.append(0)
    elif (yMovement > 0):
      movements.append(Directions.DOWN)
      movementsInt.append(1)
    else:
      movements.append(Directions.NONE)
      movementsInt.append(4)

      
  return movements, movementsInt
  
 
