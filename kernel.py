import numpy as np
import time
from scipy import signal
from pycuda import driver, compiler, gpuarray, tools
import frameGenerator

import pycuda.autoinit

def gen():
    """
    Returns: GPU kernel
    """

    # Setup Kernel
    kernels = compiler.SourceModule("""
    #include <stdio.h>

    __global__ void FindObj(int* a, int* frameOrigin, int* c, int img_height, int img_width)
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

       //if(row_o < img_height && col_o < img_width)
       //{
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

         /* point x,y
          point x-1,y if exists is different, point x,y-1 if exists is different

          Equal, and legal location
          (row_o + 1) < img_height && equal
          (row_o + 2) < img_height && equal

          col_o + 1 < img_width
          col_o + 2 < img_width

          Unequal
          row_o + 1 < img_height && unequal
          col_o - 1 >= 0 && unequal

          if(
             (row_o + 2 <= img_height  && a[(row_o + 1)*img_width + col_o] == pixelColor && a[(row_o + 2)*img_width + col_o] == pixelColor) &&
             (row_o - 2 >= 0       && a[(row_o - 1)*img_width + col_o] != pixelColor && a[(row_o - 2)*img_width + col_o] != pixelColor) &&
             (col_o + 2 <= img_width   && a[row_o*img_width + col_o + 1]   == pixelColor && a[row_o*img_width + col_o + 2]   == pixelColor) &&
             (col_o - 2 >= 0       && a[row_o*img_width + col_o - 1]   != pixelColor && a[row_o*img_width + col_o - 2]   != pixelColor))
            {
              b[0] = row_o;
              c[0] = col_o;
            }*/

        //Gather all criteria for corner
        bool row_plus1 = false; bool row_plus2 = false;
        bool col_plus1 = false; bool col_plus2 = false;
        bool row_minus1 = true; bool col_minus1 = true;

        //Next rows in same column have same color and are in bounds
        if((row_o + 1) < img_height && a[(row_o + 1)*img_width + col_o] == pixelColor){row_plus1 = true;}
        if((row_o + 2) < img_height && a[(row_o + 2)*img_width + col_o] == pixelColor){row_plus2 = true;}

        //Next cols in same row have same color and are in bounds
        if((col_o + 1) < img_width && a[row_o*img_width + col_o + 1] == pixelColor){col_plus1 = true;}
        if((col_o + 2) < img_width && a[row_o*img_width + col_o + 2] == pixelColor){col_plus2 = true;}

        //Confirm not in the middle of an edge and on a vertex
        if((row_o - 1) >= 0 && a[(row_o - 1)*img_width + col_o] == pixelColor){row_minus1 = false;}
        if((col_o - 1) >= 0 && a[row_o*img_width + col_o - 1] == pixelColor){col_minus1 = false;}

        //If vertex, do all calculations for global array
        if(row_plus1 && row_plus2 && col_plus1 && col_plus2 && row_minus1 && col_minus1){
            int frame = (int) ((row_o*img_width + col_o)/(img_width*img_height-1));
            frameOrigin[frame] = col_o;
            frameOrigin[frame+1] = row_o  %  img_height;
        }

        //Check all criteria for validity
        //Modulo absolute thread location by frame size and store x val/y val to according locations in global array
        //After all global array values set, compare set of 2 x/y pairs to determine movement and set movement indicator in other global array
        //Return final global array
        }
    }

    """)

    return kernels
