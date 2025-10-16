# Owen Gonzales
# Last updated: 22 Jul 2024

# This file contains two functions that allow for parallelization of code

import numpy as np
import psutil
from multiprocess import Pool

allcores = psutil.cpu_count(logical=False)

def Map(func, data, ncores, *args, **kwargs):

    '''
    This function is a parallelized version of the built-in map() function. It speeds up the mapping process by creating parallel streams
        which can be run at the same time, then attached back together.

    Parameters:
        func: function to be applied to each part of the split dataset
        data: dataset to be split across multiple cores
        ncores: number of cores to split dataset between
        *args: any other arguments to be passed to func() ***CANNOT BE ITERATED OVER***
        concat: if True, returns one array attatched at the 0th axis. If False, returns a list the same length as ncores
        **kwargs: any other keyword arguments to be passed to func() ***CANNOT BE ITERATED OVER***
            
    Returns:
        Returns an identical result to func(data) if concat=True. If concat=False, returns a list of length(ncores) where each
        element of the list is the portion of the calculation that each core performed

    Example:
        dataset = np.arange(1e6)
        output = map(lambda x: x*x, dataset, 8)
        print(output)
    >>> [0, 1, 4, ... 999997^2, 999998^2, 999999^2]
    Note that in this example, the output is the same as the split() function. The difference is split() would calculate [0, 1, ... 124999]**2 
    for the first chunk of the split data while map() would calculate [0**2, 1**2, ... 124999**2].
    '''
    
    allcores = psutil.cpu_count(logical=False)

    if not isinstance(ncores, int):
        raise TypeError('! Error: Number of cores must be an integer')
    if (ncores <= 0) | (ncores > allcores):
        raise Exception('! Error: Number of cores must be greater than 0 and less than the total number of cores available')

    workers = Pool(ncores)
    if (len(args) > 0) or (len(kwargs) > 0):
        def wrapper(data):
            return func(data, *args, **kwargs)
        mapdatalist = workers.map(wrapper, data)
    else:
        mapdatalist = workers.map(func, data)
    workers.close()
    workers.join()

    return mapdatalist


def Split(func, data, ncores, *args, concat=True, **kwargs):
    '''
    This function uses the multiprocess package to split calculations across multiple cores.
    It takes as input an array, a function to apply to that array, and the number of available cores.
    When run, it splits the dataset into as many different peices as there are specified cores. It then - similarly
        to a map function - applies the function to each chunk of the dataset, except in parallel. After applying
        the function to each chunk of the split data, it concatenates them back together along the 0th axis. Unlike map()
        above, this will return a numpy array as it is designed to be run on large numpy arrays.
    Note that this function will not provide a parallel benefit for most applications! Only for when your array is so large
        that it makes sense to lessen the memory demand per core.

    Parameters:
        func: function to be applied to each part of the split dataset
        data: dataset to be split across multiple cores
        ncores: number of cores to split dataset between
        *args: any other arguments to be passed to func() ***CANNOT BE ITERATED OVER***
        concat: if True, returns one array attatched at the 0th axis. If False, returns a list the same length as ncores
        **kwargs: any other keyword arguments to be passed to func() ***CANNOT BE ITERATED OVER***
    
    Returns:
        Returns an identical result to func(data) if concat=True. If concat=False, returns a list of length(ncores) where each
        element of the list is the portion of the calculation that each core performed

    Example:
        dataset = np.arange(1e6)
        output = split_processes(lambda x: x*x, dataset, 8)
        print(output)
    >>> [0, 1, 4, ... 999997^2, 999998^2, 999999^2]
    '''

    allcores = psutil.cpu_count(logical=False)
    
    if not isinstance(ncores, int):
        raise TypeError('! Error: Number of cores must be an integer')
    if (ncores <= 0) | (ncores > allcores):
        raise Exception('! Error: Number of cores must be greater than 0 and less than the total number of cores available')
    
    data = np.array_split(data, ncores)
    
    # If we are passing in a function which has some default kwargs, and we are updating them from their default values in this function call
    # we need to manually change them then change them back
    workers = Pool(ncores)
    if (len(args) > 0) or (len(kwargs) > 0):
        def wrapper(data):
            return func(data, *args, **kwargs)
        mapdatalist = workers.map(wrapper, data)
    else:
        mapdatalist = workers.map(func, data)

    workers.close()
    workers.join()

    if concat:
        return np.concatenate(mapdatalist)
    else:
        return mapdatalist
    
    
### Test case for parallelization
if __name__ == '__main__':
    import time

    def square_and_sleep(x):
        time.sleep(1)
        return x**2
    
    def timeit(func, data):
        starttime = time.time()
        output = func(data)
        elapsed_time = time.time() - starttime
        return output, elapsed_time
    
    ncores = 8

    # Test for split_processes()
    nums = np.arange(ncores*5)
    print(f'\nnums: {list(nums)}')
    print('square_and_sleep: squares a number and sleeps one second')
    print('timeit: times how long a function takes to run')
    print('split_processes: when applying a function to a large amount of data, breaks it into smaller pieces which are run parallel')
    print('map_processes: when applying a function iteratively to items in some iterable, creates multiple parallel loops')

    output, elapsed_time = timeit(square_and_sleep, nums)
    print('\nResults of square_and_sleep(nums):')
    print(list(output))
    print(f'Time taken: {elapsed_time}')

    output, elapsed_time = timeit(lambda x: Split(square_and_sleep, x, ncores), nums)
    print('\nResults of split_processes(square_and_sleep, nums, ncores):')
    print(list(output))
    print(f'Time taken: {elapsed_time}')
    print(f'Can also output as: {Split(square_and_sleep, nums, ncores, concat=False)}')

    print('\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n')

    # Test for map_processes()
    nums = np.arange(ncores*2)

    output, elapsed_time = timeit(lambda x: list(map(square_and_sleep, x)), nums)
    print('Results of list(map(square_and_sleep, nums)):')
    print(output)
    print(f'Time taken: {elapsed_time}')

    output, elapsed_time = timeit(lambda x: Map(square_and_sleep, x, ncores), nums)
    print('\nResults of map_processes(square_and_sleep, nums, ncores):')
    print(list(output))
    print(f'Time taken: {elapsed_time}')