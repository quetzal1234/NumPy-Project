"""
590PR Spring 2020.
Instructor: John Weible  jweible@illinois.edu
Assignment on Numpy: "High-Tech Sculptures"

See assignment instructions in the README.md document AND in the
TO DO comments below.
"""
import glob
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from typing import List



def get_orientations_possible(block: np.ndarray) -> List[List[dict]]:
    """Given a 3D numpy array, look at its shape to determine how many ways it
    can be rotated in each axis to end up with a (theoretically) different array
    that still has the SAME shape.

    if all three dimensions are different sizes, then we have 3 more
    orientations, excluding the original, which are all 180-degree rotations.

    if just two dimensions match size, we have 7 plus original. 90-degree
    rotations are around the unique-length axis.

    if all three dimensions match (a cube), then we have 23 plus original.

    :param block: a numpy array of 3 dimensions.
    :return: a list of the ways we can rotate the block. Each is a list of dicts containing parameters for np.rot90()

    >>> a = np.arange(64, dtype=int).reshape(4, 4, 4)  # a cube
    >>> rotations = get_orientations_possible(a)
    >>> len(rotations)
    23
    >>> rotations  # doctest: +ELLIPSIS
    [[{'k': 1, 'axes': (0, 1)}], ... [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}]]
    >>> a = a.reshape(2, 4, 8)
    >>> len(get_orientations_possible(a))
    3
    >>> a = a.reshape(16, 2, 2)
    >>> len(get_orientations_possible(a))
    7
    >>> get_orientations_possible(np.array([[1, 2], [3, 4]]))
    Traceback (most recent call last):
    ValueError: array parameter block must have exactly 3 dimensions.
    >>> marble_block_1 = np.load(file='data/marble_block_1.npy')
    >>> len(get_orientations_possible(marble_block_1))
    7
    """

    if len(block.shape) != 3:
        raise ValueError('array parameter block must have exactly 3 dimensions.')

    # Create list of the 23 possible 90-degree rotation combinations -- params to call rot90():
    poss = [
        [{'k': 1, 'axes': (0, 1)}],  # 1-axis rotations:
        [{'k': 2, 'axes': (0, 1)}],
        [{'k': 3, 'axes': (0, 1)}],
        [{'k': 1, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}],
        [{'k': 2, 'axes': (1, 2)}],
        [{'k': 3, 'axes': (1, 2)}],
        [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],  # 2-axis rotations:
        [{'k': 1, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
        ]

    # consider the 3-tuple shape of axes numbered 0, 1, 2 to represent (height, depth, width)
    (height, depth, width) = block.shape

    if height == depth == width:
        return poss  # return all possibilities, it's a cube
    rotations = []
    for combo in poss:
        r = block  # start with a view of block, unmodified for comparison
        for r90 in combo:  # apply all the rotations given in this combination
            r = np.rot90(r, k=r90['k'], axes=r90['axes'])
        if r.shape == block.shape:
            rotations.append(combo)
    return rotations


def carve_sculpture_from_density_block(shape: np.ndarray, block: np.ndarray) -> np.ndarray:
    """The shape array guides our carving. It indicates which parts of the
    material block to keep (the 1 values) and which to carve away (the 0 values),
    producing a new array that defines a sculpture and its varying densities.
    It must have NaN values everywhere that was 'carved' away.

    :param shape: array to guide carving into some 3D shape
    :param block: array describing densities throughout the raw material block
    :return: array of densities in the resulting sculpture, in same orientation.
    :raises: ValueError if the input arrays don't match in size and shape.

    >>> s = np.array([[1,0], [1,0],[1,0]])
    >>> b = np.array([[3.4, 4.2], [5.2,6.2], [7.2,8.2]])
    >>> carve_sculpture_from_density_block(s,b)
    array([[3.4, nan],
           [5.2, nan],
           [7.2, nan]])

    >>> s = np.array([[1,0], [1,0], [1,0]])
    >>> b = np.array([[3.2,4.2],[5.2,4.2]])
    >>> carve_sculpture_from_density_block(s,b)
    Traceback (most recent call last):
    ValueError: Shape and Block don't match in size and shape!
    """
    if shape.shape != block.shape:
        raise ValueError ("Shape and Block don't match in size and shape!")
    else:
        return np.where(shape == 1, block, np.nan)

def is_stable(sculpture: np.ndarray) -> bool:
    """Given a 'sculpted' NDarray, where number values represent densities and
    NaN values represent the areas carved away, determine if, in the orientation
    given, whether it will sit stably upon its base.

    :param sculpture: NDarray representing a sculpture of variable density material.

    >>> sculpture = np.array([[[3.4, 0, 4.5], [4.5, 0, 4.0]], [[5.6, 7.8, 0], [5.4, 3.2, 2.2]]])
    >>> is_stable(sculpture)
    True

    >>> marble_block_1 = np.load(file='data/marble_block_1.npy')
    >>> is_stable(marble_block_1)
    True
    """
    no_nan_sculpture = np.nan_to_num(sculpture, nan=0)
    base = no_nan_sculpture[-1]
    feet = np.nonzero(base)
    feet_coords = np.stack(feet, axis=-1)
    masscenter = center_of_mass(no_nan_sculpture)
    masscenter = np.array(masscenter[1:])
    coods_w_mass = np.append(feet_coords, [masscenter], axis=0)
    hull = ConvexHull(feet_coords)
    hull_mass = ConvexHull(coods_w_mass)
    if hull.area == hull_mass.area:
        return True
    else:
        return False


def analyze_sculptures(block_filenames: list, shape_filenames: list):
    """Given all the filenames of blocks and sculpture shapes to carve,
    analyze them in all usable block rotations to show their resulting
    densities and stabilities.  See the README.md file for an example
    output format.

    :param block_filenames:
    :param shape_filenames:
    :return:

    >>> block_filename = ["data/marble_block_1.npy"]
    >>> shape_filename = ["data/shape_1.npy"]
    >>> analyze_sculptures(block_filename, shape_filename) #write output file

    >>> block_filename = ["data/marble_block_2.npy"]
    >>> shape_filename = ["data/shape_2.npy"]
    >>> analyze_sculptures(block_filename, shape_filename) #write output file
    """

    outfile = open("output.txt", "w")
    for shape_file in shape_filenames:
        outfile.write(shape_file)
        outfile.write('\n')
        shape = np.load(shape_file)
        for block_file in block_filenames:
            outfile.write(block_file)
            outfile.write('\n')
            block = np.load(block_file)
            rotations = get_orientations_possible(block)
            for rotation in rotations:
                r = block  # start with a view of block unmodified for comparison
                for r90 in rotation:  # apply all the rotations given in this combination
                    r = np.rot90(r, k=r90['k'], axes=r90['axes'])
                    outfile.write('Rotation: {} axes {} \t'.format(r90['k'], r90['axes']))
                sculpture = carve_sculpture_from_density_block(shape, r)
                density = (np.nanmean(sculpture.astype('float32')))
                outfile.write('\tmean density: {:.2f} \t'.format(density))
                stable = is_stable(sculpture)
                if stable == True:
                    outfile.write('Stability: Stable \t \n')
                else:
                    outfile.write('Stability: Unstable \t \n')

    outfile.close()

def are_rotations_unique(list_of_rotations: List[List[dict]], verbose=False) -> bool:
    """Given a list of list of 3D rotation combinations suitable for using with np.rot90()
    and as returned from the get_orientations_possible() function, determine whether any
    of the rotations are equivalent, and discard the duplicates.

    The purpose is to detect situations where a combination of rotations would produce either
    the original unmodified array or the same orientation as any previous one in the list.

    NOTE: This function is already complete! It is provided as an example of rotation
    calculations, good Doctests, and it could be useful to you as part of your solution.

    :param list_of_rotations: a list, such as returned by get_orientations_possible()
    :param verbose: if True, will print details to console, otherwise silent.
    :return: True, if all listed rotation combinations produce distinct orientations.

    >>> x = [[{'k': 4, 'axes': (0, 1)}]]  # 4x90 degrees is a full rotation
    >>> are_rotations_unique(x)
    False
    >>> x = [[{'k': 2, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 1)}]]  # also a full rotation
    >>> are_rotations_unique(x)
    False
    >>> y1 = [[{'k': 3, 'axes': (1, 2)}], [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (2, 0)}]]
    >>> are_rotations_unique(y1)
    True
    >>> y2 = y1 + [[{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (1, 0)}]]  # equiv. to earlier
    >>> are_rotations_unique(y2, verbose=True)
    combination #1: [{'k': 3, 'axes': (1, 2)}] ok.
    combination #2: [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (2, 0)}] ok.
    combination #3: [{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (1, 0)}] not unique.
    it results in the same array as combination 2
    False
    """
    # create a small cube to try all the input rotations. It has unique values so that
    #  no distinct rotations could create an equivalent array by accident.
    cube = np.arange(0, 27).reshape((3, 3, 3))

    # Note: In the code below, the arrays must be appended to the orientations_seen
    #  list in string form, because Numpy would otherwise misunderstand the intention
    #  of the if ... in orientations_seen expression.

    orientations_seen = [cube.tostring()]  # record the original

    count = 0
    for combo in list_of_rotations:
        count += 1
        if verbose:
            print('combination #{}: {}'.format(count, combo), end='')

        r = cube  # start with a view of cube unmodified for comparison
        for r90 in combo:  # apply all the rotations given in this combination
            r = np.rot90(r, k=r90['k'], axes=r90['axes'])
        if r.tostring() in orientations_seen:
            if verbose:
                print(' not unique.')
                if r.tostring() == cube.tostring():
                    print('it results in the original 3d array.')
                else:
                    print('it results in the same array as combination',
                          orientations_seen.index(r.tostring()))
            return False
        else:
            if verbose:
                print(' ok.')
        orientations_seen.append(r.tostring())
    return True


if __name__ == '__main__':
    block_list = glob.glob('data/marble*.npy')

    shape_list = glob.glob('data/shape*.npy')

    analyze_sculptures(block_list, shape_list)


