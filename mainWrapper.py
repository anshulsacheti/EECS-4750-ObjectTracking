import numpy as np
import time
import frameGenerator
import movements
from enum import IntEnum
import argparse
import mainGPU
import kernel
from pycuda import compiler, gpuarray
import mainCPU
import mainCmdParsing
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    move_path = ['LINE', 'ZIG-ZAG', 'TRIANGLE', 'SQUARE', 'PENTAGON']

    # Iterate over each path type 1000 times
    # Calculate how often each gets the answer correct
    # Measure runtime
    cpuRuntime = np.zeros(len(move_path))
    gpuRuntime = np.zeros(len(move_path))

    for pathIter,path in enumerate(move_path):

        # Initialize
        cpuErrorRate = 0
        gpuErrorRate = 0
        mismatchErrorRate = 0
        iterations = 100


        for i in range(iterations):

            # Runtime concerns on CPU implementation mean that larger frames take significantly longer
            # To address this we have a smaller number of frames, and the size of frames is smaller as well
            frames, goldenCoord = mainCmdParsing.createFrames(frame_size = [256, 256], num_of_frames = 25, move_set = [path], color_scale = 256, size_of_object = 5, movement_distance = 4)

            start = time.time()
            cpuShape = mainCPU.calculateShape(frames=frames, goldenCoord=goldenCoord)
            end = time.time() - start
            cpuRuntime[pathIter] += end

            start = time.time()
            gpuShape = mainGPU.calculateShape(frames=frames, goldenCoord=goldenCoord)
            end = time.time() - start
            gpuRuntime[pathIter] += end

            if cpuShape.find(path):
                cpuErrorRate+=1
            if gpuShape.find(path):
                gpuErrorRate+=1
            if gpuShape.find(cpuShape):
                mismatchErrorRate+=1

            if i % 20 == 0:
                print("On iter %d of %s" % (i, path))
            # print("Shapes - CPU: %s, gpuShape: %s" % (cpuShape, gpuShape))
            # print("Current error rates: %f %f %f" % (cpuErrorRate, gpuErrorRate, mismatchErrorRate))

        # Get averages
        cpuErrorPercent = (1.*cpuErrorRate/iterations)*100
        gpuErrorPercent = (1.*gpuErrorRate/iterations)*100
        mismatchErrorPercent = (1.*mismatchErrorRate/iterations)*100

        gpuRuntime[pathIter] /= iterations
        cpuRuntime[pathIter] /= iterations

        print("For paths creating %s, cpuErrorPercentage: %f, gpuErrorPercentage: %f, cpu-gpu Mismatch ErrorPercentage: %f" % (path, cpuErrorPercent, gpuErrorPercent, mismatchErrorPercent))


    # Plot
    plt.gcf()

    for iteration,path in enumerate(move_path):
      plt.plot(iteration, cpuRuntime[iteration], marker='o', markersize=3, color='red', label="CPU_"+path)
      plt.plot(iteration, gpuRuntime[iteration], marker='o', markersize=3, color='green', label="GPU_"+path)


    plt.legend(loc='best')
    plt.xticks(range(len(move_path)),move_path)
    plt.xlabel("Path Type")
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs CUDA GPU RunTime")
    plt.gca().set_ylim([0,max(max(cpuRuntime),max(gpuRuntime))])
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_gpuCUDA_runtime_plot.png',bbox_inches='tight')
    plt.close()
