import os
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

import linecache
import tracemalloc


def display_top(snapshot, key_type='lineno', limit=3):
    """Profiler"""
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def gblur(img, K):
    """Gaussian blur with zero padding."""
    M, N = img.shape
    new_img = np.zeros(img.shape)
    K |= 1  # Make kernel size odd
    bn = binom.pmf(np.arange(0, K), K - 1, 0.5)  # Binomial coefficients
    kern = np.reshape(bn, [K, 1]) * bn
    for k in range(K):
        for l in range(K):
            i_shift = k - (K >> 1)
            j_shift = l - (K >> 1)
            i_p = max(i_shift, 0)
            i_n = max(-i_shift, 0)
            j_p = max(j_shift, 0)
            j_n = max(-j_shift, 0)
            new_img[i_n:M - i_p, j_n:N - j_p] += kern[k, l] * \
                                                 img[i_p:M - i_n, j_p:N - j_n]
    return new_img


def np_avg(a, axis=-1, prepend=None, append=None):
    """
    Calculate the average of adjoining entries along the given axis.
    (Modified on np.diff)

    The first difference is given by ``out[i] = (a[i+1] + a[i]) / 2``
    along the given axis.

    Parameters
    ----------
    a : array_like
        Input array
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

        .. versionadded:: 1.16.0

    Returns
    -------
    avg : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.
    """

    if a.dtype not in [np.float16, np.float32, np.float64]:
        raise TypeError("avg only applies to floating point numbers")
    a = np.core.umath.asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("avg requires input that is at least one dimensional")
    axis = np.core.multiarray.normalize_axis_index(axis, nd)

    combined = []
    if prepend is not None:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    slice1 = tuple(slice(1, None) if i == axis else slice(None) for i in range(nd))
    slice2 = tuple(slice(None, -1) if i == axis else slice(None) for i in range(nd))

    return .5 * (a[slice1] + a[slice2])


def minmod(a, b):
    return .5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def reinit(u0, h, V0, dt=None, steps=None, tol=1e-10):
    """Compute an initial value by expanding or shrinking a initial region,
    which is represented by a signed distance function, to desired volume.

    1. Use reinitialization method (Sussman & Fatemi, 1996) to obtain the signed
        distance $u(x)$. (Implementation from Bin Dong)
    2. Use Newton's iteration to obtain the appropriate threshold $c$ such that
                    phi=\tanh((c-u)/eps)
        has the desired volume where eps is the desired interface width."""
    # Use reinitialization procedure to get signed distance function
    # Solve the hyperbolic equation u_t + sgn(u)|∇u|=sgn(u0)
    u = np.copy(u0)
    u = gblur(u, 7)
    if dt is None:
        dt = 0.5 * h
    if steps is None:
        steps = max(10, int(2. / h))
    for _ in range(steps):
        # 1st difference (extend by copying boundary row)
        Dxu = np.diff(u, axis=0, prepend=u[0:1, :], append=u[-1:, :])
        Dyu = np.diff(u, axis=1, prepend=u[:, 0:1], append=u[:, -1:])
        G_p = np.sqrt(np.maximum(np.maximum(Dxu[:-1, :], 0) ** 2, np.minimum(Dxu[1:, :], 0) ** 2)
                      + np.maximum(np.maximum(Dyu[:, :-1], 0) ** 2, np.minimum(Dyu[:, 1:], 0) ** 2))
        G_n = np.sqrt(np.maximum(np.minimum(Dxu[:-1, :], 0) ** 2, np.maximum(Dxu[1:, :], 0) ** 2)
                      + np.maximum(np.minimum(Dyu[:, :-1], 0) ** 2, np.maximum(Dyu[:, 1:], 0) ** 2))
        u -= dt / h * ((u > 0) * (G_p - h) * u / np.sqrt(u ** 2 + G_p ** 2 + 1e-9)
                       + (u < 0) * (G_n - h) * u / np.sqrt(u ** 2 + G_n ** 2 + 1e-9))
    # return u
    # Get phase field
    eps = 5 * h
    # vol = lambda c: np.sum(np.tanh((c - u) / eps)) * h ** 2
    # vol_d = lambda c: np.sum((1 / np.cosh((c - u) / eps)) ** 2) * h ** 2 / eps
    c = np.average(u)
    for i in range(3):
        # Starting with eps too small can cause overflow.
        # Use continuation method to decrease it slowly
        for _ in range(steps):
            phi = np.tanh((c - u) / eps)
            V = np.sum(phi + 1) * h ** 2
            # print(vol(c),vol_d(c))
            if abs(V - 2 * V0) < tol:
                break
            if abs(V - 2 * V0) > 0.5:
                a = 0.1  # Damped step length
            else:
                a = 1  # Full step length
            c -= a * (V - 2 * V0) / (np.sum(1 / np.cosh((c - u) / eps) ** 2) * h ** 2 / eps)
        # c-=(vol(c)-V0)/vol_d(c)
        eps /= 2
    return .5 * (1 + phi)


def zigzag_traverse(*ranges):
    """Traverse multi-d parameters in given range in a zig-zag manner which
    ensures successive parameter tuples are close together (only differ by 1
    component)."""

    def is_end(i):
        """Whether the i-th index has reached end"""
        nonlocal ind, sign, lens
        if sign[i] > 0:
            return ind[i] >= lens[i] - 1
        else:
            return ind[i] <= 0

    D = len(ranges)  # dimensionality
    ind = [0] * D  # multi-index
    sign = [1] * D  # sign of increment (+1 or -1)
    lens = [len(v) for v in ranges]  # maximum indices
    while True:
        yield tuple(ranges[i][ind[i]] for i in range(D))
        # Change from last to first
        i = D - 1
        while is_end(i) and i >= 0:  # Find index to increment
            i -= 1
        if i == -1:  # All indices reach end
            break
        # Roll over and continue
        ind[i] += sign[i]
        # Reverse increment direction for following indices
        for j in range(i + 1, D):
            sign[j] *= -1


if __name__ == "__main__":
    np.random.seed(2010)
    A = np.random.randn(3, 5, 7, 2)
    for i in range(4):
        B = np_avg(A, axis=i, prepend=0, append=0)
        print(B.shape)
        print(np.sum(B), np.sum(A))

    N = 128
    xx, yy = np.meshgrid(np.arange(1, N) / N - 0.5, np.arange(1, N) / N - 0.5)
    z = xx ** 2 + 2 * yy ** 2 + (np.exp(4 * xx) - 2 * xx + 1)
    # plt.colorbar()
    u0 = np.sign(z - 2)
    u = reinit(u0, 1. / N, np.sum(u0 < 0) / N ** 2, steps=200)
    img = plt.imshow(u)
    plt.colorbar(img)
    plt.contour(u, levels=np.linspace(0, 1, 10), colors="red")
    plt.contour(z, colors="white")
    # plt.show()
    for vals in zigzag_traverse([1, 2, 3, 4, 5],
                                [.1, .2, .5, .7],
                                ['a', 'b', 'c'],
                                [100, 200]):
        print(vals)
