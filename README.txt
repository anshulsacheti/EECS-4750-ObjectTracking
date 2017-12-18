Please provide a README file in your submission. The following should be included in the README file:
Descriptions of the project, sub-modules, and files
Instructions to compile and run the program
Expected output

Our project aims to parallelize object recognition to track an object over
hundreds or thousands of video frames and identify the object’s path. The goal is
to take advantage of GPU parallelism and CUDA to examine individual pixels for
object identification and tracking.

The project contains the following files:

mainWrapper.py

  Functions/Methods:
    main -  The wrapper around our code base. It generates frames that are
            then passed to mainCPU and mainGPU to calculate the expected shape
            of the object path. These function calls are timed and outputs stored
            and plotted respectively for tracking error percentages and run times.

frameGenerator.py
  Functions/Methods:
    main -  Just calls the function gen()
    gen  -  Generates a set of frames with a wide variety of parms to determine
            how to generate them. Such as: number of frames, size of the object,
            and movement distance.

mainCPU.py
  Functions/Methods:
    main              - Calls mainCmdParsing.main_cmdLine to parse args and then
                        calls calculateShape to determine object path shape
    findOrigins       - Passes frames to CPU for origin calculation
    neighborSlope     - Calculates the slope of the object movement across frames
    CountSlopeChange  - Counts the number of slope changes and corresponds that
                        to sides
    distanceCompare   - Runs a locality sensitive hash variant to calculate the
                        overall path of an input object
    calculateShape    - Calculate final path representation of input

mainGPU.py
  Functions/Methods:
    main              - Calls mainCmdParsing.main_cmdLine to parse args and then
                        calls calculateShape to determine object path shape
    kernel_Wrapper    - Passes frames to GPU for origin calculation
    calculateShape    - Calculates the shape of object path using frames and
                        golden coordinates

kernels.py
  Functions/Methods:
    genModulo         - Kernel to calculate corners of objects across all frames
    genFullGPU        - Unused
    genHist           - Kernel to generate histogram of actions taken in each
                        direction
    MovementAnalysis  - Kernels to analyze complex movements, such as slope
                        and object path shape calculations
movements.py
  Functions/Methods:
    frameCompare      - Generates list of directions moved frame to frame for an
                        object

  Classes:
    Directions        - Converts frame origin coordinates to direction moved frame
                        to frame

  Global Vars:
    MOVEMENT_DB       - Database of numerical representation of movements
    MOVEMENT_DB_NAME  - Database of shapes corresponding to a given set of movements

mainCmdParsing.py
  Functions/Methods:
    main_cmdLine      - Parse args from cmd line and return frames with given args
    createFrames      - Calls frameGenerator.gen with frame parms

To run:
python mainWrapper.py
sbatch --gres=gpu:1 --wrap="/opt/PYTHON/bin/python mainWrapper.py"

Expected Output:
Because this is a stochastic model and we randomly generate input data, the actual
output numbers will vary but if mainWrapper.py is run then you should see something
like the following:

For paths creating LINE, cpuErrorPercentage: 33.333333, gpuErrorPercentage: 26.666667,
cpu-gpu Mismatch ErrorPercentage: 0.000000
For paths creating TRIANGLE, cpuErrorPercentage: 26.666667, gpuErrorPercentage: 20.000000,
cpu-gpu Mismatch ErrorPercentage: 0.000000
For paths creating SQUARE, cpuErrorPercentage: 0.000000, gpuErrorPercentage: 6.666667,
cpu-gpu Mismatch ErrorPercentage: 0.000000
For paths creating PENTAGON, cpuErrorPercentage: 6.666667, gpuErrorPercentage: 6.666667,
cpu-gpu Mismatch ErrorPercentage: 0.000000

NOTE: This output was run using a 320x320 frame with a total of 72 frames generated
per iteration in mainWrapper. This took approximately 25min to run most of it due
to how we generating frames. Since the frames are completely random, we don't know
whether a frame is illegal before trying to move an object within it in real time
which has some overhead. Moreover, to properly represent the object path shapes
a certain granularity is required and we found this to be around 70 frames.
