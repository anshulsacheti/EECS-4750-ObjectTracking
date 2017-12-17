import numpy as np
import pdb
from PIL import Image

def gen( frame_size = [256, 256], num_of_frames = 24, move_set = ["right", "up"],
                    color_scale = 256, size_of_object = 8, movement_distance = 4):
    """
    Generates sets of frames that represent frame to frame movement for each frame set
    Adds object to each frame with location adjusted against previous frame movement

    Returns: set of frames representative of single movement

    Inputs:
        frame_size:
        num_of_frames:
        move_set:
        color_scale:
        size_of_object:
        movement_distance:
    Outputs: List of numpy n-d arrays. Zero index is initial frame. All subsequent frames follow each movement from move_set
    """
    # pdb.set_trace()

    # Evaluate inputs correct
    if not(move_set[0] in ['LINE','ZIG-ZAG','TRIANGLE', 'SQUARE', 'PENTAGON']) and \
        (num_of_frames != len(move_set)):
        raise InputError('Number of frames generated doesn\'t match number of movements')
        return

    if len(frame_size) < 2:
        raise InputError('Frame Size is not at least 2d')
        return

    # Overwriting move_set with auto generated actions that represent:
    # ['line','zig_zag','triangle', 'square', 'pentagon']
    if move_set[0] in ['LINE', 'ZIG-ZAG', 'TRIANGLE', 'SQUARE', 'PENTAGON']:

        if move_set[0] == 'LINE':
            direction = np.random.choice(['up','down','left','right'])
            move_set = [direction]*num_of_frames

        elif move_set[0] == 'ZIG-ZAG':

            # Make even
            if num_of_frames % 2:
                num_of_frames -= 1

            v_direction = np.random.choice(['up','down'])
            h_direction = np.random.choice(['left','right'])

            num_of_frames_div2 = int(num_of_frames/2)

            if np.random.uniform()>0.5:
                move_set = [v_direction, h_direction]*num_of_frames_div2
            else:
                move_set = [h_direction, v_direction]*num_of_frames_div2

        elif move_set[0] == 'TRIANGLE':

            # Make divisble by 4
            if num_of_frames % 4:
                num_of_frames = (num_of_frames/4) * 4

            move_set = []

            if np.random.uniform()>0.5:
                v_direction = 'up'
                v_next_direction = 'down'
            else:
                v_direction = 'down'
                v_next_direction = 'up'

            if np.random.uniform()>0.5:
                h_direction = 'left'
                h_next_direction = 'right'
            else:
                h_direction = 'right'
                h_next_direction = 'left'

            num_of_frames_div4 = int(num_of_frames/4)
            num_of_frames_div2 = int(num_of_frames/2)

            # Different combinations form different triangles
            if np.random.uniform()>0.5:
                move_set = [v_direction, h_direction]*num_of_frames_div4
                move_set.extend([v_next_direction, h_direction]*num_of_frames_div4)
                move_set.extend([h_next_direction]*num_of_frames_div2)
            else:
                move_set = [h_direction, v_direction]*num_of_frames_div4
                move_set.extend([h_next_direction,v_direction]*num_of_frames_div4)
                move_set.extend([v_next_direction]*num_of_frames_div2)

        elif move_set[0] == 'SQUARE':

            # Make divisble by 4
            if num_of_frames % 4:
                num_of_frames = (num_of_frames/4) * 4

            move_set = []
            num_of_frames_div4 = int(num_of_frames/4)

            move_set.append(['up']*num_of_frames_div4)
            move_set.append(['left']*num_of_frames_div4)
            move_set.append(['down']*num_of_frames_div4)
            move_set.append(['right']*num_of_frames_div4)

            # Put different sets of directions together
            move_set = np.roll(move_set,np.random.randint(4))
            move_set = [dir for set in move_set for dir in set]

        elif move_set[0] == 'PENTAGON':

            # Make divisble by 6 based on pentagon we're trying to draw
            if num_of_frames % 6:
                num_of_frames = (num_of_frames/6) * 6

            move_set = []
            num_of_frames_div3 = int(num_of_frames/3)
            num_of_frames_div6 = int(num_of_frames/6)
            num_of_frames_div12 = int(num_of_frames/12)

            move_set.append(['up']*num_of_frames_div6)
            move_set.append(['right','up']*num_of_frames_div12)
            move_set.append(['right','down']*num_of_frames_div12)
            move_set.append(['down']*num_of_frames_div6)
            move_set.append(['left']*num_of_frames_div3)

            # Put different sets of directions together
            move_set = np.roll(move_set,np.random.randint(5))
            move_set = [dir for set in move_set for dir in set]

    # Generate initial frame
    frame0 = np.random.randint(0,high=color_scale-1,size=frame_size)
    frame1 = frame0.copy()

    # Generate Random n-d location for square object
    num_of_dims = len(frame_size)
    object_idx = []
    for dim in range(num_of_dims):
        object_idx.append(np.random.randint(0,high= ( frame_size[dim] - size_of_object)))

    # Update frame0 with initial object placement
    object_x = object_y = object_z = 0
    object_val = np.random.randint(0,high=color_scale)

    # Golden reference coordinates for origin of object
    goldenCoord = []
    if num_of_dims == 2:
        object_y = object_idx[0]
        object_x = object_idx[1]
        frame1[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object)] = object_val
        goldenCoord = [object_x, object_y]

    elif num_of_dims == 3:
        object_y = object_idx[0]
        object_x = object_idx[1]
        object_z = object_idx[2]
        frame1[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object),object_z:(object_z+size_of_object)] = object_val
        goldenCoord = [object_x, object_y]

    # Create frames with each new movement
    frame_set = [frame1]
    for frameIdx in range(num_of_frames):
        movement = move_set[frameIdx]

        copy = frame0.copy()

        # Adjust object origin
        if movement == 'left':
            object_x -= movement_distance
        elif movement == 'right':
            object_x += movement_distance
        elif movement == 'down':
            object_y += movement_distance
        elif movement == 'up':
            object_y -= movement_distance

        # Determine if movement causes object to move out of bounds, ignore this frame set
        if object_x < 0 or object_x+size_of_object >= frame_size[0] or object_y < 0 or object_y+size_of_object >= frame_size[1]:
            # print("Got out of bounds")
            return [[],[]]

        if num_of_dims == 2:
            copy[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object)] = object_val
            goldenCoord.extend([object_x, object_y])

        elif num_of_dims == 3:
            copy[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object),object_z:(object_z+size_of_object)] = object_val
            goldenCoord.extend([object_x, object_y, object_z])

        frame_set.append(copy)

    # for i,frame in enumerate(frame_set):
    #     frame = frame.astype(np.uint8)
    #     tmp = Image.fromarray(frame)
    #     tmp.save('frame' + str(i) + '.png')
    return [frame_set, goldenCoord]
if __name__ == '__main__':
    gen()
