import numpy as np
import time
from pycuda import compiler, gpuarray
import frameGenerator
import kernel
import movements
from enum import IntEnum


######################################################################################################################################
def Kernel_Wrapper (frames, goldenCoord):

    input_matrix = np.vstack(frames).astype(np.int32)

    print("Reference origins: %s" % goldenCoord)
    x = np.zeros(len(goldenCoord)).astype(np.int32)
    for i in range(1,len(goldenCoord),2):
        x[i] = goldenCoord[i-1]
        x[i-1] = goldenCoord[i] + i/2*10
    print("Y val ref: %s" % x)
    print ("INPUT TO GPU")
    for frame in frames:
        print(frame)

    # Setup Params
    FRAME_BATCH_SIZE = len(frames)
    BLOCK_WIDTH = 16
    BLOCK_HEIGHT = 16
    GRID_WIDTH = (INPUT_SIZE_WIDTH - 1) // BLOCK_WIDTH + 1
    GRID_HEIGHT = (INPUT_SIZE_HEIGHT*len(frames) -1) // BLOCK_HEIGHT + 1
    zeroed = np.zeros((FRAME_BATCH_SIZE)*2, dtype=np.int32)

    # Setup Data Structures
    a_gpu = gpuarray.to_gpu(input_matrix.flatten())
    frameOrigin_gpu = gpuarray.to_gpu(zeroed)
    c_gpu = gpuarray.to_gpu(zeroed)
    block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)
    gdim = (GRID_WIDTH, GRID_HEIGHT, 1)

    CUDARunTime = 0
    start = time.time()

    # switch out kernels
    FindObj_CUDA = kernel.genModulo().get_function("FindObj")
    FindObj_CUDA(a_gpu, frameOrigin_gpu, c_gpu, np.int32(INPUT_SIZE_HEIGHT), np.int32(INPUT_SIZE_WIDTH), block = block, grid = gdim)

    CUDARunTime = time.time() - start

    print ("CUDA RUNTIME")
    print (CUDARunTime)

    print ("Frame Origins")
    print (frameOrigin_gpu.get())

    x = np.zeros(len(goldenCoord)).astype(np.int32)
    for i in range(1,len(goldenCoord),2):
        x[i-1] = goldenCoord[i-1]
        x[i] = goldenCoord[i] + i/2*10
    print("golden val ref: %s" % goldenCoord)
    return frameOrigin_gpu.get()

######################################################################################################################################

# MAIN

# frames = frameGenerator.gen( frame_size = [256, 256], num_of_frames = 2, move_set = ["right", "up"],
#                        color_scale = 256, size_of_object = 15, movement_distance = 10)


frames = []

goldenCoord = []

INPUT_SIZE_WIDTH = 500
INPUT_SIZE_HEIGHT = 500
while (not frames):
    frames, goldenCoord = frameGenerator.gen( frame_size = [INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT], num_of_frames = 72, move_set =
                           ["right", "right", "right", "right","right", "right","right","right","right","right","right","right","right","right","right","right","right","right",
                            "up", "up", "up", "up","up","up","up","up","up","up","up","up","up","up","up","up","up","up",
                            "left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left","left",
                            "down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down","down"],
                           color_scale = 256, size_of_object = 5, movement_distance = 4)

origin = Kernel_Wrapper(frames, goldenCoord)
frameMovements, frameMovementsInts = movements.frameCompare(origin)

print (frameMovements)
print (frameMovementsInts)



MOVEMENT_HASH = []


#Count the movements using histogram
a_gpu = gpuarray.to_gpu(np.asarray(frameMovementsInts, dtype=np.int32))
b_gpu = gpuarray.zeros((4), np.int32)
Hist_Moves = kernel.genHist().get_function("histOpt")
Hist_Moves(a_gpu, b_gpu, block=(4,1,1))
print("Histogram Movements")
print(b_gpu.get())



#Not Used as part of core algo
#Get slopes from the start point
zeroed = np.zeros(len(goldenCoord)/2 - 1, dtype = np.float32)
a_gpu = gpuarray.to_gpu(np.asarray(goldenCoord, dtype=np.float32))
slopesFromStart = gpuarray.to_gpu(zeroed)
FindSlopes = kernel.MovementAnalysis().get_function("SlopeHistory")
FindSlopes(a_gpu, slopesFromStart, block=((len(zeroed) + 1,1,1)))

print(slopesFromStart.get())

#Not Used as part of core algo
#Get differences of slopes
zeroed = np.zeros(len(goldenCoord)/2-1, dtype = np.float32)
differencesOfSlopes = gpuarray.to_gpu(zeroed)
FindDiffOfSlopes = kernel.MovementAnalysis().get_function("DifferencesOfSlopes")
FindDiffOfSlopes(slopesFromStart, differencesOfSlopes, block=((len(zeroed)+1,1,1)))

print(differencesOfSlopes)








#Get slopes between jumps
JUMP = 3
zeroed = np.zeros(len(goldenCoord)/2 - JUMP, dtype = np.float32)
slopesBetweenJumps = gpuarray.to_gpu(zeroed)
NeighborSlope = kernel.MovementAnalysis().get_function("NeighborSlope")
NeighborSlope(a_gpu, slopesBetweenJumps, np.int32(JUMP), block=((len(zeroed),1,1)))

print(slopesBetweenJumps)

# still need algo to count the Sides...





# DATABASE OF MOVEMENTS

MOVEMENT_DB = []
MOVEMENT_DB_NAME = []

MOVEMENT_DB.append([0.25, 0.25, 0.25, 0.25, 3])
MOVEMENT_DB_NAME.append("TRIANGLE")

MOVEMENT_DB.append([0.25, 0.25, 0.25, 0.25, 4])
MOVEMENT_DB_NAME.append("SQUARE")

MOVEMENT_DB.append([0.25, 0.25, 0.25, 0.25, 5])
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


testInput = [10,0,0,10,4]
GPU_IN = gpuarray.to_gpu(np.asarray(testInput, dtype=np.float32))

zeroed = np.zeros(len(MOVEMENT_DB), dtype = np.float32)
SCORED_CLOSENESS = gpuarray.to_gpu(zeroed)
MOVEMENT_DB_GPU = gpuarray.to_gpu(np.asarray(MOVEMENT_DB, dtype=np.float32).flatten())
DistanceCompare = kernel.MovementAnalysis().get_function("DistanceCompare")
DistanceCompare(GPU_IN, MOVEMENT_DB_GPU, SCORED_CLOSENESS, block=(len(zeroed),1,1))


closestDist = np.max(SCORED_CLOSENESS.get())

i = 0
for x in SCORED_CLOSENESS.get():
    if x == closestDist:
        break
    i += 1

SHAPE_MATCH = MOVEMENT_DB_NAME[i]

print(closestDist)
print(i)
print(SHAPE_MATCH)
