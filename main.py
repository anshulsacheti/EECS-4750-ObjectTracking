import numpy as np
import time
from scipy import signal
from pycuda import driver, compiler, gpuarray, tools
import frameGenerator
import kernel
import movements


######################################################################################################################################
def Kernel_Wrapper (frames):
    
    input_matrix = np.vstack(frames)
    
    print ("INPUT TO GPU")
    print (input_matrix)

    # Setup Params
    FRAME_BATCH_SIZE = len(frames)
    BLOCK_WIDTH = 16
    BLOCK_HEIGHT = 16
    GRID_WIDTH = (INPUT_SIZE_WIDTH - 1) // BLOCK_WIDTH + 1
    GRID_HEIGHT = (INPUT_SIZE_HEIGHT -1) // BLOCK_HEIGHT + 1
    zeroed = np.zeros((FRAME_BATCH_SIZE), dtype=np.int32)

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

    return frameOrigin_gpu.get()

######################################################################################################################################





# MAIN

# frames = frameGenerator.gen( frame_size = [256, 256], num_of_frames = 2, move_set = ["right", "up"],
#                        color_scale = 256, size_of_object = 15, movement_distance = 10)


frames = []
while (not frames):
    frames = frameGenerator.gen( frame_size = [10, 10], num_of_frames = 5, move_set = ["right", "up", "right", "down", "left"],
                           color_scale = 256, size_of_object = 2, movement_distance = 2)

origin = Kernel_Wrapper(frames)
frameMovements = movements.frameCompare(origin)
