"""
Provides a function to enable summing of primes.
"""


def get_sum(maximum):
    """
    Add all the prime numbers below the argument max_val.
    """
    if not isinstance(maximum, int):
        maximum = int(maximum)

    primes = [2, 3]
    temp = []
    i = 1
    while i < (maximum + 1) / 6:
        temp.append(6 * i - 1)
        temp.append(6 * i + 1)
        i += 1

    for item in temp:
        state = 1
        for prime in primes[:]:
            if item % prime == 0:
                # number not prime
                break

        if state:
            primes.append(item)

    return sum(primes)
