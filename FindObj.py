import numpy as np
import time
from scipy import signal
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

# Setup Kernel
kernels = compiler.SourceModule("""
#include <stdio.h>

__global__ void FindObj(int* a, int* b, int* c, int height, int width, int objRadius)
{
   // Setup Indexing
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int row_o = blockIdx.y*blockDim.y + ty;
   int col_o = blockIdx.x*blockDim.x + tx;

   // printf("%i %i \\n", tx, ty);
   // printf("row_o:%i col_o:%i \\n", row_o, col_o);
   // printf("blockIdx.y:%i blockDim.y:%i \\n", blockIdx.y, blockDim.y);
   // printf("blockIdx.x:%i blockDim.x:%i \\n", blockIdx.x, blockDim.x);

   if(row_o < height && col_o < width)
   {
	// printf("row_o:%i col_o:%i a[row_o*width + col_o]: %i \\n",row_o, col_o, a[row_o*width + col_o]);

	int pixelColor = a[row_o*width + col_o];

        // Lots of different ways to detect obj
	// Four points

        // printf("row_o:%i col_o:%i a[row_o*width + col_o]: %i \\n",row_o, col_o, a[row_o*width + col_o]);

	if((row_o + objRadius <= height  && a[(row_o + objRadius)*width + col_o] == pixelColor) &&
           (row_o - objRadius >= 0       && a[(row_o - objRadius)*width + col_o] == pixelColor) &&
           (col_o + objRadius <= width   && a[row_o*width + col_o + objRadius]   == pixelColor) &&
           (col_o - objRadius >= 0       && a[row_o*width + col_o - objRadius]   == pixelColor)
        )
        {
	  b[0] = row_o;
	  c[0] = col_o;
	}

   }

}

""")

findObj_CUDA = kernels.get_function("FindObj")


def RUN_TEST (INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH, OBJ_RADIUS):

    input_matrix =  np.random.randint(10,size=(INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH)).astype(np.int32)

    input_matrix[32,33] = 1
    input_matrix[33,33] = 1
    input_matrix[33,32] = 1
    input_matrix[33,34] = 1
    input_matrix[34,33] = 1
    print (input_matrix)

    # Setup Params
    FRAME_BATCH_SIZE = 10
    BLOCK_WIDTH = 16
    GRID_WIDTH = (INPUT_SIZE_WIDTH - 1) // BLOCK_WIDTH + 1
    GRID_HEIGHT = (INPUT_SIZE_HEIGHT -1) // BLOCK_WIDTH + 1
    zeroed = np.zeros((FRAME_BATCH_SIZE), dtype=np.int32)

    # Setup Data Structures
    a_gpu = gpuarray.to_gpu(input_matrix.flatten())
    b_gpu = gpuarray.to_gpu(zeroed)
    c_gpu = gpuarray.to_gpu(zeroed)
    block = (BLOCK_WIDTH, BLOCK_WIDTH, 1)
    gdim = (GRID_WIDTH, GRID_HEIGHT, 1)

    CUDARunTime = 0
    start = time.time()
    findObj_CUDA(a_gpu, b_gpu, c_gpu, np.int32(INPUT_SIZE_HEIGHT), np.int32(INPUT_SIZE_WIDTH), np.int32(OBJ_RADIUS), block = block, grid = gdim)
    CUDARunTime = time.time() - start
    print (CUDARunTime)

    print (b_gpu.get())
    print (c_gpu.get())

RUN_TEST(35, 35, 1)



