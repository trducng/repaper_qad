import numpy as np
import numpy.random as random
from numba import jit


def bestIntSplit(ratio, total):
    """Divides a total into integer shares that best reflects ratio
    Args:
      share      - [1 X N ] - Percentage in each pile
      total      - [int   ] - Integer total to split

    Returns:
      intSplit   - [1 x N ] - Number in each pile
  """
    # Handle poorly defined ratio
    if sum(ratio) is not 1:
        ratio = np.asarray(ratio) / sum(ratio)

    # Get share in real and integer values
    floatSplit = np.multiply(ratio, total)
    intSplit = np.floor(floatSplit)
    remainder = int(total - sum(intSplit))

    # Rank piles by most cheated by rounding
    deserving = np.argsort(-(floatSplit - intSplit), axis=0)

    # Distribute remained to most deserving
    intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1
    return intSplit

# @jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4 * acc / nsamples


if __name__ == '__main__':
    # result = bestIntSplit([1/1, 1/3.5, 1/3.5, 1/3.5, 1/3.5], 16)
    # print(result)
    # print(sum(result), len(result))

    import timeit
    t = timeit.timeit('monte_carlo_pi(10000)', number=1000, globals=globals())
    print(t)
