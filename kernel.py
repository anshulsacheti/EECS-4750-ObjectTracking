import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import frameGenerator

import pycuda.autoinit

def genModulo():
    """
    Calculates corners of objects across all frames
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

       int pixelColor = a[row_o*img_width + col_o];

        //Gather all criteria for corner
        bool row_plus1 = false; bool row_plus2 = false;
        bool col_plus1 = false; bool col_plus2 = false;
        bool row_minus1 = true; bool col_minus1 = true;

        //Next rows in same column have same color and are in bounds
        if(((row_o + 1) %  img_height) < img_height && a[(row_o + 1)*img_width + col_o] == pixelColor){row_plus1 = true;}
        if(((row_o + 2) %  img_height) < img_height && a[(row_o + 2)*img_width + col_o] == pixelColor){row_plus2 = true;}

        //Next cols in same row have same color and are in bounds
        if((col_o + 1) < img_width && a[row_o*img_width + col_o + 1] == pixelColor){col_plus1 = true;}
        if((col_o + 2) < img_width && a[row_o*img_width + col_o + 2] == pixelColor){col_plus2 = true;}

        //Confirm not in the middle of an edge and on a vertex
        if((row_o - 1) >= 0 && a[(row_o - 1)*img_width + col_o] == pixelColor){row_minus1 = false;}
        if((col_o - 1) >= 0 && a[row_o*img_width + col_o - 1] == pixelColor){col_minus1 = false;}

        //printf("row: %i col: %i pixelColor: %i, row_plus1: %i, row_plus2: %i, col_plus1: %i, col_plus2: %i, row_minus1: %i, col_minus1: %i \\n",row_o, col_o, pixelColor, row_plus1, row_plus2, col_plus1, col_plus2, row_minus1, col_minus1);
        //If vertex, do all calculations for global array
        if(row_plus1 && row_plus2 && col_plus1 && col_plus2 && row_minus1 && col_minus1){
            int frame = (int) ((row_o*img_width + col_o)/(img_width*img_height-1));
            //printf("Found corner. row_o:%i col_o:%i pixelColor: %i, frame: %i\\n",row_o, col_o, pixelColor, frame);
            frameOrigin[frame*2] = col_o;
            frameOrigin[frame*2+1] = row_o  %  img_height;
        }

        //Check all criteria for validity
        //Modulo absolute thread location by frame size and store x val/y val to according locations in global array
        //After all global array values set, compare set of 2 x/y pairs to determine movement and set movement indicator in other global array
        //Return final global array
    }

    """)

    return kernels

def genFullGPU():
    """
    Returns GPU kernel with x/y pair comparison for frames
    """

    # Setup Kernel
    kernels = compiler.SourceModule("""
    #include <stdio.h>

    __global__ void FindObj(int* a, int* frameOrigin, int* frameDiff, int img_height, int img_width)
    {
       // Setup Indexing
       int tx = threadIdx.x;
       int ty = threadIdx.y;
       int row_o = blockIdx.y*blockDim.y + ty;
       int col_o = blockIdx.x*blockDim.x + tx;

       //Update frameOrigin before use to know when it's updated
       //Assumes all frames worked on simultaneously
       if((row_o*img_width + col_o)<20){
            int frame = (int) ((row_o*img_width + col_o)/(img_width*img_height-1));
            frameOrigin[frame] = -1;
            frameOrigin[frame+1] = -1;
       }
       __syncthreads();

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

            while(!(frame % 2) && frameOrigin[frame+2]==-1 && frameOrigin[frame+3]==-1) {
                bool x = true;
            }

            //All even frames calculate diff between them and next frame
            if(!(frame % 2)){

                //Calculate what distance was moved
                if(frameOrigin[frame]>frameOrigin[frame+2]){
                    frameDiff[frame]=2;
                }
                else {
                    if(frameOrigin[frame]<frameOrigin[frame+2]){
                        frameDiff[frame]=4;
                    }
                }
                if(frameOrigin[frame+1]>frameOrigin[frame+3]){
                    frameDiff[frame]=6;
                }
                else {
                    if(frameOrigin[frame+1]<frameOrigin[frame+3]){
                        frameDiff[frame]=8;
                    }
                }
            }
        }

        //Check all criteria for validity
        //Modulo absolute thread location by frame size and store x val/y val to according locations in global array
        //After all global array values set, compare set of 2 x/y pairs to determine movement and set movement indicator in other global array
        //Return final global array
    }

    """)

def genHist():
    """
    Generate histogram
    Input:
        variable histogramValues: 1-d array with values for histogram
    Return/Output: [hist]
    """

    kernels = compiler.SourceModule("""
    #include <stdio.h>
    #define BINS 4

    __global__ void histOpt(int* histInput, int* histOutput) {

        __shared__ int localHist[BINS];

        int bx = blockIdx.x;
        int tx = threadIdx.x;
        int x = bx * blockDim.x + tx;

        //Initialize bins to 0
        if (tx < BINS) {
            localHist[tx] = 0;
        }
        __syncthreads();

        //Calculate local
        //printf("histInput[%d] = %d, Bin = %d\\n", x, histInput[x], loc);
        atomicAdd( &(localHist[histInput[x]]), 1);
        __syncthreads();

        //Store to global
        if (tx < BINS) {
            //printf("Thread: %d, histOutput[%d] = %d, localHist[tx] = %d\\n", x, tx, histOutput[tx], localHist[tx]);
            atomicAdd( &(histOutput[tx]), localHist[tx] );
            //printf("Thread: %d, histOutput[%d] = %d, localHist[tx] = %d\\n", x, tx, histOutput[tx], localHist[tx]);
        }

    }
    """)

    return kernels

def MovementAnalysis():
    """
    Kernels to analyze complex movements
    """

    kernels = compiler.SourceModule("""
    #include <stdio.h>

    __global__ void SlopeHistory(float* a, float* b)
    {
       int tx = threadIdx.x;
       // Establish the initial point
       float initialPointX = a[0];
       float initialPointY = a[1];
       //printf("tx: %i, a[tx]: %f \\n", tx, a[tx]);

       if(tx >= 1)
       {
           // Get the point which will build the slope from the initial point
           float historyPointX = a[tx * 2];
           float historyPointY = a[(tx * 2) + 1];
           //printf("tx: %f, historyPointX: %f, historyPointY: %f, initialPointX: %f, initialPointY: %f \\n", tx, historyPointX, historyPointY, initialPointX, initialPointY);

           // Build the list of all historical slopes with respect to the start point
           if(historyPointX - initialPointX > 0)
           {
               b[tx - 1] = (historyPointY - initialPointY) / (historyPointX - initialPointX);
           }
           else
           {
               b[tx - 1] = 1000000;
           }

       }

    }

    __global__ void NeighborSlope(float* a, float* b, int jump)
   {

      // Finds all sequential slopes
      int tx = threadIdx.x;
      //printf("tx: %i, a[tx]: %f \\n", tx, a[tx]);
      //printf("tx: %i, a[tx]: %f, a[(tx+jump)*2 +1)]: %f, (a[(tx+jump)*2]: %f \\n", tx, a[tx], a[(tx+jump)*2 +1], a[(tx+jump)*2]);
      b[tx] = (a[(tx+jump)*2 + 1] - a[(tx*2) + 1]) /  (a[(tx+jump)*2] - a[(tx*2)]);
   }

   __global__ void DistanceCompare(float* a, float* b, float* c)
   {


      int tx = threadIdx.x;

      // Break the HASH down to analyze
      float UP = a[0];
      float DOWN = a[1];
      float LEFT = a[2];
      float RIGHT = a[3];
      float CORNERS = a[4];
      float MOVES_TOTAL = UP + DOWN + LEFT + RIGHT;

      // Get the DB hash to compare
      float UP_DB = b[tx*5];
      float DOWN_DB = b[tx*5 + 1];
      float LEFT_DB = b[tx*5 + 2];
      float RIGHT_DB = b[tx*5 + 3];
      float CORNERS_DB = b[tx*5 + 4];
      float MOVES_TOTAL_DB = UP_DB + DOWN_DB + LEFT_DB + RIGHT_DB;

      // printf("UP_DB: %f, DOWN_DB: %f, LEFT_DB: %f, RIGHT_DB: %f, CORNERS_DB: %f \\n", UP_DB, DOWN_DB, LEFT_DB, RIGHT_DB, CORNERS_DB);

      // Calculate the closeness to the DB hash
      float DISTANCE_COEFF = 0.0;

      if(CORNERS == CORNERS_DB)
      {
        DISTANCE_COEFF += 60;
      }

      DISTANCE_COEFF += 20 * (1- fabsf((UP_DB/MOVES_TOTAL_DB) - (UP/MOVES_TOTAL)));
      DISTANCE_COEFF += 20 * (1- fabsf((DOWN_DB/MOVES_TOTAL_DB) - (DOWN/MOVES_TOTAL)));
      DISTANCE_COEFF += 20 * (1- fabsf((LEFT_DB/MOVES_TOTAL_DB) - (LEFT/MOVES_TOTAL)));
      DISTANCE_COEFF += 20 * (1- fabsf((RIGHT_DB/MOVES_TOTAL_DB) - (RIGHT/MOVES_TOTAL)));

      // Store to be compared
      c[tx] = DISTANCE_COEFF;

   }


   __global__ void DifferencesOfSlopes(float* a, float* b)
   {
      int tx = threadIdx.x;
      b[tx] = a[(tx + 1)] - a[tx];
   }


    """)

    return kernels
