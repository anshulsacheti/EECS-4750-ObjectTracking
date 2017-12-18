
import numpy as np
np.warnings.filterwarnings('ignore')
import time
import frameGenerator
import movements
from enum import IntEnum
import argparse

def createFrames(frame_size, num_of_frames, move_set, color_scale, size_of_object, movement_distance):
    """
    Generates frames to be analyzed

    Inputs:
        frame_size
        num_of_frames
        move_set: The movements such as up/down/left/right frame to frame or the overall shape they should take such as 'triangle'
        color_scale: The range of values for each pixel
        size_of_object:
        movement_distance: How much to move in a given direction frame to frame
    Returns:
        frames: set of frames representing object movement over time
        goldenCoord: golden origin coordinates
    """
    frames = []
    goldenCoord = []

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


def main_cmdLine(args):
    """
    Parse args from cmd line and return frames with given args
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--INPUT_SIZE_HEIGHT', type=int)
    parser.add_argument('--INPUT_SIZE_WIDTH', type=int)
    parser.add_argument('--num_of_frames', type=int)
    parser.add_argument('--move_set', type=str)
    parser.add_argument('--color_scale', type=int)
    parser.add_argument('--size_of_object', type=int)
    parser.add_argument('--movement_distance', type=int)
    parser.add_argument('--printVal',default = False, type=bool)

    args = parser.parse_args()

    frames, goldenCoord =  createFrames(frame_size = [args.INPUT_SIZE_HEIGHT, args.INPUT_SIZE_WIDTH], num_of_frames = args.num_of_frames, move_set = [args.move_set],
                           color_scale = args.color_scale, size_of_object = args.size_of_object, movement_distance = args.movement_distance)

    return [frames, goldenCoord]
