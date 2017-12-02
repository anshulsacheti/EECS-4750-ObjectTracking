import numpy as np
import time
from scipy import signal
from pycuda import driver, compiler, gpuarray, tools
import frameGenerator

import pycuda.autoinit

# Setup Kernel
kernels = compiler.SourceModule("""
#include <stdio.h>

__global__ void FindObj(int* a, int* b, int* c, int img_height, int img_width)
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

   if(row_o < img_height && col_o < img_width)
   {
      // printf("row_o:%i col_o:%i a[row_o*img_width + col_o]: %i \\n",row_o, col_o, a[row_o*img_width + col_o]);
      int pixelColor = a[row_o*img_width + col_o];

      // printf("row_o:%i col_o:%i a[row_o*img_width + col_o]: %i \\n",row_o, col_o, a[row_o*img_width + col_o]);

      // Lots of different ways to detect obj

      // Find 4 point cross
      /*
      if((row_o + 1 <= img_height  && a[(row_o + 1)*img_width + col_o] == pixelColor) &&
               (row_o - 1 >= 0       && a[(row_o - 1)*img_width + col_o] == pixelColor) &&
               (col_o + 1 <= img_width   && a[row_o*img_width + col_o + 1]   == pixelColor) &&
               (col_o - 1 >= 0       && a[row_o*img_width + col_o - 1]   == pixelColor)
            )
            {
        b[0] = row_o;
        c[0] = col_o;
      }
      */

      // Find Corner
      // TODO: FIX WHEN SQUARE IS ON THE EDGE OF IMAGE...

      point x,y
      point x-1,y if exists is different, point x,y-1 if exists is different

      Equal, and legal location
      (row_o + 1)*img_width < img_height && equal
      (row_o + 2)*img_width < img_height && equal

      col_o + 1 < img_width
      col_o + 2 < img_width

      Unequal
      row_o + 1*img_width is < img_height && unequal
      col_o - 1 is >= 0 && unequal

      if(
         (row_o + 2 <= img_height  && a[(row_o + 1)*img_width + col_o] == pixelColor && a[(row_o + 2)*img_width + col_o] == pixelColor) &&
         (row_o - 2 >= 0       && a[(row_o - 1)*img_width + col_o] != pixelColor && a[(row_o - 2)*img_width + col_o] != pixelColor) &&
         (col_o + 2 <= img_width   && a[row_o*img_width + col_o + 1]   == pixelColor && a[row_o*img_width + col_o + 2]   == pixelColor) &&
         (col_o - 2 >= 0       && a[row_o*img_width + col_o - 1]   != pixelColor && a[row_o*img_width + col_o - 2]   != pixelColor))
        {
          b[0] = row_o;
          c[0] = col_o;
        }

    //Gather all criteria for corner
    //Check all criteria for validity
    //Modulo absolute thread location by frame size and store x val/y val to according locations in global array
    //After all global array values set, compare set of 2 x/y pairs to determine movement and set movement indicator in other global array
    //Return final global array
    }
}

""")

findObj_CUDA = kernels.get_function("FindObj")


def RUN_TEST (INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH):

    input_matrix =  np.random.randint(10,size=(INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH)).astype(np.int32)

    frames = frameGenerator.gen( frame_size = [256, 256], num_of_frames = 2, move_set = ["right", "up"],
                        color_scale = 256, size_of_object = 15, movement_distance = 10)

    SQUARE_WIDTH = 3
    ORIGIN = 10
    for x in range(0, SQUARE_WIDTH):
	  for y in range(0, SQUARE_WIDTH):
		input_matrix[x + ORIGIN, y + ORIGIN] = 1

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
    findObj_CUDA(a_gpu, b_gpu, c_gpu, np.int32(INPUT_SIZE_HEIGHT), np.int32(INPUT_SIZE_WIDTH), block = block, grid = gdim)
    CUDARunTime = time.time() - start
    print (CUDARunTime)

    print (b_gpu.get())
    print (c_gpu.get())

RUN_TEST(20, 20)
