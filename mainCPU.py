import numpy as np
import time
# from pycuda import compiler, gpuarray
import frameGenerator
# import kernel
import movements
from enum import IntEnum
import argparse

# MAIN

# Generate frames with moving object

# Build HASH
  # Find the origins for object in each frame
  # Determine the difference in movement between frames
  # Classify the direction of movements (UP, DOWN, LEFT, RIGHT) between each frame
  # Count the number of each directional movement
  # Find the slopes of all adjacent points in the movement history
  # Use slope history to determine the number of sides in the object's path history
  # Use the directional number counts and the sides to build hash classifier

# Pass into LSH
  # Find nearness value to each item in movement database using the hash above
  # Determine the highest scored value, ie movement determined to be most similar to the hash
  # Retrieve the movement from the database

# Print





# frames = frameGenerator.gen( frame_size = [256, 256], num_of_frames = 2, move_set = ["right", "up"],
#                        color_scale = 256, size_of_object = 15, movement_distance = 10)


# Generate frames to be analyzed
def createFrames(frame_size, num_of_frames, move_set, color_scale, size_of_object, movement_distance):

    # GENERATE FRAMES TO BE ANALYZED
    frames = []
    goldenCoord = []

    # while (not frames):
    #     frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set =
    #                            ["right", "right", "right", "right","right", "right","right","right","right","right","right","right","right","right","right","right","right","right",
    #                             "up", "up", "up", "up","up","up","up","up","up","up","up","up","up","up","up","up","up","up",
    #                             "left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left",
    #                             "down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down"],
    #                            color_scale = 256, size_of_object = 5, movement_distance = 4)

    while (not frames):
        frames, goldenCoord = frameGenerator.gen( frame_size = frame_size, num_of_frames = num_of_frames, move_set = move_set,
                                                    color_scale = color_scale, size_of_object = size_of_object, movement_distance = movement_distance)

    # TESTS

    #while (not frames):
    #    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set = 'square'
    #                           color_scale = 256, size_of_object = 5, movement_distance = 4)

    #while (not frames):
    #    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set = 'triangle'
    #                           color_scale = 256, size_of_object = 5, movement_distance = 4)

    #while (not frames):
    #    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set = 'line'
    #                           color_scale = 256, size_of_object = 5, movement_distance = 4)

    #while (not frames):
    #    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set = 'zig-zag'
    #                           color_scale = 256, size_of_object = 5, movement_distance = 4)

    #while (not frames):
    #    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set = 'pentagon'
    #                           color_scale = 256, size_of_object = 5, movement_distance = 4)

    # TESTS
    return [frames, goldenCoord]

# Find origins of object in frames
def findOrigins(frames, goldenCoord):

    img_height = frames[0].shape[0]
    img_width = frames[0].shape[1]

    frameOrigins = np.zeros(len(frames)*2)

    # Match format of calculation as on GPU
    for frameCount,frame in enumerate(frames):
        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):

                # Initializations
                pixelColor = frame[r,c]
                row_plus1 = row_plus2 = col_plus1 = col_plus2 = False
                row_minus1 = col_minus1 = True

                # Next rows in same col have same color and are in bounds
                if ((r + 1) < img_height and frame[r+1,c] == pixelColor):
                    row_plus1 = True
                if ((r + 2) < img_height and frame[r+2,c] == pixelColor):
                    row_plus2 = True

                # Next cols in same row have same color and are in bounds
                if ((c + 1) < img_width and frame[r,c+1] == pixelColor):
                    col_plus1 = True
                if ((c + 2) < img_width and frame[r,c+2] == pixelColor):
                    col_plus2 = True

                # Confirm not in the middle of an edge and on a vertex
                if ((r - 1) >= 0 and frame[r-1,c] == pixelColor):
                    row_minus1 = False
                if ((c - 1) >= 0 and frame[r,c-1] == pixelColor):
                    col_minus1 = False

                # print("row: %i col: %i pixelColor: %i, row_plus1: %i, row_plus2: %i, col_plus1: %i, col_plus2: %i, row_minus1: %i, col_minus1: %i" % (r, c, pixelColor, row_plus1, row_plus2, col_plus1, col_plus2, row_minus1, col_minus1))
                # If vertex, do all calculations for frame
                if (row_plus1 and row_plus2 and col_plus1 and col_plus2 and row_minus1 and col_minus1):

                    # print("Found corner. r:%i c:%i pixelColor: %i, frame: %i" % (r, c, pixelColor, frame))
                    frameOrigins[frameCount*2] = c;
                    frameOrigins[frameCount*2+1] = r;

    print("Generated origins == golden: %s" % (np.allclose(frameOrigins, goldenCoord)))
    return frameOrigins

# Get slopes between jumps to determine the number of sides of the path
def neighborSlope(fMI, jump, goldenCoord):
    """
    jump: Amount to jump forward over range when calculating slope
    fMI: list of frame movement ints
    """

    slopesBetweenJumps = np.zeros(int(len(fMI)/2) - jump)

    for i in range(int(len(fMI)/2) - jump):
        x_slopes = (fMI[(i+jump)*2] - fMI[(i*2)])
        if x_slopes == 0:
            slopesBetweenJumps[i] = np.inf
        else:
            slopesBetweenJumps[i] = (fMI[(i+jump)*2 + 1] - fMI[(i*2) + 1]) /  x_slopes

    return slopesBetweenJumps

# counts slope change to count sides
def countSlopeChange(slopesBetweenJumps):

    currSlope = 10000
    sides = 0
    for x in slopesBetweenJumps:
       if(abs(x - currSlope) > 10):
          sides +=1
          currSlope = x
    return sides

# Another algo to count slope change to count sides
def countSlopeChange2(slopesBetweenJumps):

    countSides = 0
    currSlope2 = 10000
    runningTotal = 0

    for x in slopesBetweenJumps:

       if(x == currSlope2):
          runningTotal += 1
       else:
          currSlope2 = x
          runningTotal = 0

       if(runningTotal == 3):
          countSides += 1

    return countSides

# Runs a locality sensitive hash variant to calculate the overall movement of an input object
def DistanceCompare(MOVEMENT_HASH, MOVEMENT_DB, MOVEMENT_DB_NAME):

    SCORED_CLOSENESS = np.zeros(len(MOVEMENT_DB))

    # Iterate over every movement
    for i in range(len(SCORED_CLOSENESS)):
      # Break the HASH down to analyze
      UP = MOVEMENT_HASH[0]
      DOWN = MOVEMENT_HASH[1]
      LEFT = MOVEMENT_HASH[2]
      RIGHT = MOVEMENT_HASH[3]
      CORNERS = MOVEMENT_HASH[4]
      MOVES_TOTAL = UP + DOWN + LEFT + RIGHT

      # Get the DB hash to compare
      UP_DB = MOVEMENT_DB[i][0]
      DOWN_DB = MOVEMENT_DB[i][1]
      LEFT_DB = MOVEMENT_DB[i][2]
      RIGHT_DB = MOVEMENT_DB[i][3]
      CORNERS_DB = MOVEMENT_DB[i][4]
      MOVES_TOTAL_DB = UP_DB + DOWN_DB + LEFT_DB + RIGHT_DB

      # print("UP_DB: %f, DOWN_DB: %f, LEFT_DB: %f, RIGHT_DB: %f, CORNERS_DB: %f" % (UP_DB, DOWN_DB, LEFT_DB, RIGHT_DB, CORNERS_DB))

      # Calculate the closeness to the DB hash
      DISTANCE_COEFF = 0.0

      if CORNERS == CORNERS_DB:
        DISTANCE_COEFF += 60

      DISTANCE_COEFF += 20 * (1- np.abs((1.*UP_DB/MOVES_TOTAL_DB) - (1.*UP/MOVES_TOTAL)))
      DISTANCE_COEFF += 20 * (1- np.abs((1.*DOWN_DB/MOVES_TOTAL_DB) - (1.*DOWN/MOVES_TOTAL)))
      DISTANCE_COEFF += 20 * (1- np.abs((1.*LEFT_DB/MOVES_TOTAL_DB) - (1.*LEFT/MOVES_TOTAL)))
      DISTANCE_COEFF += 20 * (1- np.abs((1.*RIGHT_DB/MOVES_TOTAL_DB) - (1.*RIGHT/MOVES_TOTAL)))

      # Store to be compared
      SCORED_CLOSENESS[i] = DISTANCE_COEFF

    closestDist = np.max(SCORED_CLOSENESS)

    i = np.where(SCORED_CLOSENESS==closestDist)[0][0]

    SHAPE_MATCH = MOVEMENT_DB_NAME[i]

    return [closestDist, i, SHAPE_MATCH]

def calculateShape(frames, goldenCoord):
    """
    Calculate final path representation of input
    """

    # Get origin of object in each frame
    origins = findOrigins(frames, goldenCoord)

    # print(origins)
    # Calculate directional movement of object between frames
    frameMovements, frameMovementsInts = movements.frameCompare(origins)
    print(frameMovements)
    print(frameMovementsInts)
    # print (frameMovements)
    # print (frameMovementsInts)

    # Movement hash
    # Build up hash using character list from histogram output
    MOVEMENT_HASH = []
    MOVEMENT_HASH += np.histogram(frameMovementsInts, bins=[0,1,2,3,4])[0].tolist()

    print(MOVEMENT_HASH)
    # Get slopes between jumps to determine the number of sides of the path
    JUMP = 3
    slopesBetweenJumps = neighborSlope(frameMovementsInts, JUMP, goldenCoord)
    print(slopesBetweenJumps)
    # print(slopesBetweenJumps)

    sides = countSlopeChange(slopesBetweenJumps)
    print(sides)

    # Build up more of the hash
    MOVEMENT_HASH.append(sides)
    print(MOVEMENT_HASH)

    # Slope difference is being used so commented this out
    # countSides = countSlopeChange2(slopesBetweenJumps)
    # print(countSides)
    # MOVEMENT_HASH.append(countSides)

    closestDist, i, SHAPE_MATCH = DistanceCompare(MOVEMENT_HASH, np.asarray(movements.MOVEMENT_DB), movements.MOVEMENT_DB_NAME)

    print("BEST SCORE: ")
    print(closestDist)

    print("INDEX OF BEST SCORE: ")
    print(i)

    print("FINAL PATH SHAPE MATCH: ")
    print(SHAPE_MATCH)

    # return [closestDist, i, SHAPE_MATCH]

def main(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--INPUT_SIZE_HEIGHT', type=int)
    parser.add_argument('--INPUT_SIZE_WIDTH', type=int)
    parser.add_argument('--num_of_frames', type=int)
    parser.add_argument('--move_set', type=str)
    parser.add_argument('--color_scale', type=int)
    parser.add_argument('--size_of_object', type=int)
    parser.add_argument('--movement_distance', type=int)

    args = parser.parse_args()

    frames, goldenCoord = createFrames(frame_size = [args.INPUT_SIZE_HEIGHT, args.INPUT_SIZE_WIDTH], num_of_frames = args.num_of_frames, move_set = [args.move_set],
                           color_scale = args.color_scale, size_of_object = args.size_of_object, movement_distance = args.movement_distance)

    calculateShape(frames, goldenCoord)

    return [frames, goldenCoord]

if __name__=='__main__':

    import sys
    main(sys.argv[1:])
