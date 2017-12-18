
import numpy as np
import time
from enum import Enum

class Directions(Enum):
	"""
	Converts frame origin coordinates to direction moved frame to frame
	"""
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
  """
  Inputs:
  	originPoints - list of object origins
  Returns:
    list of directions object moved frame to frame, list of numerical representation of those directions
  """
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



# DATABASE OF MOVEMENTS

MOVEMENT_DB = []
MOVEMENT_DB_NAME = []

MOVEMENT_DB.append([0.167, 0.167, 0.33, 0.33, 3])
MOVEMENT_DB_NAME.append("TRIANGLE")

MOVEMENT_DB.append([0.25, 0.25, 0.25, 0.25, 4])
MOVEMENT_DB_NAME.append("SQUARE")

MOVEMENT_DB.append([0.30, 0.30, 0.20, 0.20, 5])
MOVEMENT_DB_NAME.append("PENTAGON")

MOVEMENT_DB.append([0.5, 0.001, 0.5, 0.001, 4])
MOVEMENT_DB_NAME.append("ZIG-ZAG-UP-LEFT-4-SIDES")

MOVEMENT_DB.append([0.5, 0.001, 0.001, 0.5, 4])
MOVEMENT_DB_NAME.append("ZIG-ZAG-UP-RIGHT-4-SIDES")

MOVEMENT_DB.append([0.001, 0.5, 0.5, 0.001, 4])
MOVEMENT_DB_NAME.append("ZIG-ZAG-DOWN-LEFT-4-SIDES")

MOVEMENT_DB.append([0.001, 0.5, 0.001, 0.5, 4])
MOVEMENT_DB_NAME.append("ZIG-ZAG-DOWN-RIGHT-4-SIDES")

MOVEMENT_DB.append([1, 0.001, 0.001, 0.001, 1])
MOVEMENT_DB_NAME.append("LINE-UP")

MOVEMENT_DB.append([0.001, 1, 0.001, 0.001, 1])
MOVEMENT_DB_NAME.append("LINE-DOWN")

MOVEMENT_DB.append([0.001, 0.001, 1, 0.001, 1])
MOVEMENT_DB_NAME.append("LINE-LEFT")

MOVEMENT_DB.append([0.001, 0.001, 0.001, 1, 1])
MOVEMENT_DB_NAME.append("LINE-RIGHT")
