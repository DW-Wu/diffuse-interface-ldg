import os, sys, json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ran
from numpy.linalg import norm
import scipy.optimize as optimize
from scipy.sparse.linalg import LinearOperator, gmres

from utils import *


class LCConfig(dict):
    """LC+PhF model configuration object. Same as python dictionary"""

    def __init__(self, **kwargs):
        p_default = {'B': 6400, 'C': 3500, 'L': 4e-11,
                     'N': 128, 'lam': 1e-6, 'epsr': 0.005, 'vol': 0.04,
                     'wa': 4e7, 'wv': 1e14, 'wphi': 2e8}
        p_default.update(kwargs)
        super().__init__(**p_default)


def save_lc_config(fn, x):
    """Save configuration as json"""
    if not fn.endswith('.json'):
        fn += '.json'
    with open(fn, 'w') as f:
        f.writelines(json.dumps(dict(x), separators=(',', ': '), indent=2))


def load_lc_config(fn):
    """Load configuration from json"""
    with open(fn, 'rb') as f:
        D = dict(json.loads(f.read()))
    return LCConfig(**D)


class LCState:
    """State variable of LdG liquid crystal and phase field
    We use taggered grid model, P is (N+1)*(N+1)*(2*2) and φ is N*N
    Both are equipped with 0 Dirichlet BC's, so actual sizes are P: (N-1)*(N-1)*2, φ: (N-2)*(N-2)."""

    def __init__(self, N, x=None):
        # Basic parameters
        self.N = N
        self.h = 1. / N
        if x is None:
            self.x = np.zeros(3 * N**2 - 8 * N + 6)
        else:
            self.x = x[:]  # Point to existing array
        # State variables pointing to the same data
        self.p11 = np.reshape(self.x[0:(N - 1)**2], [N - 1, N - 1])
        self.p12 = np.reshape(
            self.x[(N - 1)**2:2 * (N - 1)**2], [N - 1, N - 1])
        self.phi = np.reshape(self.x[2 * (N - 1)**2:], [N - 2, N - 2])


def view_as_lc(x0: np.ndarray, N):
    """Convert (or `view` in numpy terminology) arrays to LCState"""
    return LCState(N, x0)


def load_lc(filename, N):
    """Load state from .npy file and convert to LCState class"""
    return view_as_lc(np.load(filename), N)


class LCFunc(object):
    """Functionals of the LC state

    Use the porous medium approach, i.e. the LdG energy density is modified to
            ½*|∇Q|^2 + λ^2 F_b(Q) + α_ε(φ)*|Q|^2,
    where α_ε(φ) equals 0 when φ=1, and is very large when φ=0.
    For our purpose, we take α(φ)=w_v(1-φ)."""

    def __init__(self, **kwargs):
        # Lengths
        # Characteristic length (nano-scale)
        lam = kwargs.get("char_len", 1e-8)
        capil = kwargs.get("capil_rel", 0.01)  # Capillary width relative to λ
        # Coefficients of Landau-de Gennes energy
        B, C = kwargs.get("Landau_coeff", (6400, 3500))
        # Order parameter s_+ of homogeneous state, measure of uniaxiality
        # Use the special case A=-B^2/3C, s_+=B/C
        self.sp = B / C
        # Coefficient of elastic energy
        L = kwargs.get("elastic_coeff", 4e-11)
        # Penalty factors
        siga = kwargs.get("w_anch", 1000)  # For anchoring
        sigv = kwargs.get("w_void", 100)  # For void
        wphi = kwargs.get("w_per", 1)  # For Cahn-Hilliard functional
        # Coefficients (a lot of them...)
        self.reset_params(B, C, L,
                          lam, capil, siga, sigv, wphi)

    class LCFuncAux(object):
        """Auxiliary values during evaluation"""

        def __init__(self, X: LCState, wb1: float):
            # Auxiliary quantities, stores temporary values to simplify computation
            # We have to pre-allocate space to improve speed
            N = X.N
            self.trP2 = np.zeros([N - 1, N - 1])
            self.phi_avg = np.zeros([N - 1, N - 1])
            self.Dphi_x = np.zeros([N - 1, N - 1])
            self.Dphi_y = np.zeros([N - 1, N - 1])
            self.Dphi2 = np.zeros([N - 1, N - 1])
            self.dFb_dP11 = np.zeros([N - 1, N - 1])
            self.dFb_dP12 = np.zeros([N - 1, N - 1])
            self.update(X, wb1)

        def update(self, X: LCState, wb1):
            """X: LC state variable
            wb1: Coefficient of tr(P^2) in bulk energy"""
            self.trP2[:] = 2. * (X.p11**2 + X.p12**2)  # (N-1)*(N-1)
            self.phi_avg[:] = np_avg(np_avg(X.phi, axis=0, prepend=0, append=0),
                                     axis=1, prepend=0, append=0)  # (N-1)*(N-1)
            self.Dphi_x[:] = np_avg(np.diff(X.phi, axis=0, prepend=0, append=0),
                                    axis=1, prepend=0, append=0)  # (N-1)*(N-1)
            self.Dphi_y[:] = np_avg(np.diff(X.phi, axis=1, prepend=0, append=0),
                                    axis=0, prepend=0, append=0)  # (N-1)*(N-1)
            self.Dphi2[:] = self.Dphi_x**2 + self.Dphi_y**2
            # Derivative of bulk energy
            self.dFb_dP11[:] = 4 * (wb1 + self.trP2) * X.p11
            self.dFb_dP12[:] = 4 * (wb1 + self.trP2) * X.p12

    def reset_params(self, B, C, L, lam, epsr, siga, sigv, wphi):
        self.sp = B / C
        self.wb = lam**2 * .5 * C / L  # bulk energy F_b
        self.wb1 = -.5 * (B / C)**2  # tr(P^2) in bulk
        self.wv = sigv * lam**2  # 1/2*|P^2| in void
        self.wa = siga * epsr * lam  # |...|^2 in anchor
        self.wp1 = 2 * wphi * lam * epsr  # 1/2*|∇φ|^2 in perimeter
        self.wp0 = wphi * lam / epsr  # W(φ) in perimeter

    def reset_conf(self, conf: LCConfig):
        self.reset_params(conf['B'], conf['C'], conf['L'], conf['lam'],
                          conf['epsr'], conf['wa'], conf['wv'], conf['wphi'])

    def volume(self, X: LCState):
        return X.h**2 * np.sum(X.phi)

    def energy(self, X: LCState, part="all"):
        # Get energy of LC state
        h = X.h
        aux = self.LCFuncAux(X, self.wb1)
        E = 0
        if part == 0 or part == "all":
            # Bulk energy (WITHOUT φ)
            E += h**2 * self.wb * \
                np.sum(((self.wb1 + .5 * aux.trP2) * aux.trP2))
            # Elastic energy
            E += np.sum(np.diff(X.p11, axis=0, prepend=0, append=0)**2) \
                + np.sum(np.diff(X.p11, axis=1, prepend=0, append=0)**2) \
                + np.sum(np.diff(X.p12, axis=0, prepend=0, append=0)**2) \
                + np.sum(np.diff(X.p12, axis=1, prepend=0, append=0)**2)
        if part == 3 or part == "all":
            # Perimeter
            E += self.wp0 * h**2 * np.sum(X.phi**2 * (1 - X.phi)**2)
            E += .5 * self.wp1 * (np.sum(np.diff(X.phi, axis=0, prepend=0, append=0)**2)
                                  + np.sum(np.diff(X.phi, axis=1, prepend=0, append=0)**2))
        if part == 2 or part == "all":
            # Anchor
            E += self.wa * np.sum((X.p11 * aux.Dphi_x + X.p12 * aux.Dphi_y + .5 * self.sp * aux.Dphi_x)**2
                                  + (X.p12 * aux.Dphi_x - X.p11 * aux.Dphi_y + .5 * self.sp * aux.Dphi_y)**2)
        if part == 1 or part == "all":
            # Void (Note: 1-φ changed to (1-φ)^2)
            E += .5 * self.wv * h**2 * \
                np.sum((1 - aux.phi_avg)**2 * aux.trP2)
        return E

    def energy_vec(self, x, N, **kwargs):
        """Evaluate function on vector input"""
        x_lc = view_as_lc(x, N)  # Change array type
        return self.energy(x_lc, **kwargs)

    def grad(self, X: LCState, aux: LCFuncAux = None, part="all"):
        """Full gradient of the energy functional, same size as input"""
        if aux is not None:
            aux.update(X, self.wb1)
        else:
            aux = self.LCFuncAux(X, self.wb1)
        N = X.N
        h = X.h

        # Gradient is of same dimension as state variable
        G = LCState(N)
        if part == 0 or part == "all":
            # Bulk energy
            # h**2 * self.wb * np.sum(((self.wb1 + .5 * aux.trP2) * aux.trP2))
            G.p11[:] += self.wb * h**2 * aux.dFb_dP11
            G.p12[:] += self.wb * h**2 * aux.dFb_dP12
            # Elastic energy
            G.p11[:] += -2 * (np.diff(X.p11, n=2, axis=0, prepend=0, append=0)
                              + np.diff(X.p11, n=2, axis=1, prepend=0, append=0))
            G.p12[:] += -2 * (np.diff(X.p12, n=2, axis=0, prepend=0, append=0)
                              + np.diff(X.p12, n=2, axis=1, prepend=0, append=0))
        if part == 3 or part == "all":
            # Perimeter (Double well & diffusion)
            G.phi[:] += self.wp0 * 2 * h**2 * X.phi * (1. - X.phi) * (1. - 2 * X.phi)
            G.phi[:] -= self.wp1 * (np.diff(X.phi, n=2, axis=0, prepend=0, append=0)
                                    + np.diff(X.phi, n=2, axis=1, prepend=0, append=0))
        if part == 2 or part == "all":
            # Anchoring penalty
            G.p11[:] += self.wa * (2 * aux.Dphi2 * X.p11 +
                                   self.sp * (aux.Dphi_x**2 - aux.Dphi_y**2))
            G.p12[:] += self.wa * (2 * aux.Dphi2 * X.p12 +
                                   2 * self.sp * aux.Dphi_x * aux.Dphi_y)
            # Some laborious computation...
            pinkie = aux.trP2 + .5 * self.sp**2
            foo = (pinkie + 2 * self.sp * X.p11) * \
                aux.Dphi_x + 2 * self.sp * X.p12 * aux.Dphi_y
            bar = (pinkie - 2 * self.sp * X.p11) * \
                aux.Dphi_y + 2 * self.sp * X.p12 * aux.Dphi_x
            G.phi[:] -= self.wa * (np_avg(np.diff(foo, axis=0), axis=1)
                                   + np_avg(np.diff(bar, axis=1), axis=0))
        if part == 1 or part == "all":
            # Void penalty (note: 1-φ changed to (1-φ)^2)
            G.p11[:] += 2 * self.wv * h**2 * (1. - aux.phi_avg)**2 * X.p11
            G.p12[:] += 2 * self.wv * h**2 * (1. - aux.phi_avg)**2 * X.p12
            G.phi[:] += self.wv * h**2 * \
                np_avg(np_avg((aux.phi_avg-1)*aux.trP2, axis=0), axis=1)
        # Project phi gradient
        G.phi[:] -= np.average(G.phi)
        return G.x

    def grad_vec(self, x, N, **kwargs):
        """Evaluate gradient on vector input"""
        x_lc = view_as_lc(x, N)
        return self.grad(x_lc, **kwargs)

    def grad_P_vec(self, x, N, **kwargs):
        G = self.grad_vec(x, N, **kwargs)
        G[2 * (N - 1)**2:] = 0.
        return G

    def grad_descent(self, X0: LCState, maxiter, eta, tol=1e-8, bb=False, verbose=False, inspect=False):
        X = LCState(X0.N)
        X.x[:] = X0.x
        if inspect:
            fvec = np.zeros(maxiter)
        flag = 0
        aux = self.LCFuncAux(X0, self.wb1)
        for k in range(maxiter):
            G = self.grad(X, aux)
            if (gnorm := norm(G)) < tol:
                if verbose:
                    print("Iter over @ No. ", k, ", |g| =", gnorm)
                flag = 1
                break
            a = eta
            if bb and k > 0:
                y = G - Gp
                if k & 1:
                    a = np.dot(s, s) / np.dot(y, s)
                else:
                    a = np.dot(s, y) / np.dot(y, y)
                # if a < -1e-2:
                #     a = eta  # Keep step length positive
            # print(k,"::", a, gnorm)
            # Descent
            X.x[:] -= a * G
            # Force volume constraint
            # X.phi[1:N - 1, 1:N - 1] -= (np.sum(X.phi) - V * N**2) / (N - 2)**2
            if np.any(np.isnan(X.x)) and verbose:
                print("NAN at step", k)
                flag = -1
                break
            if bb:
                s = -a * G
                Gp = np.copy(G)
            if inspect:
                fvec[k] = self.energy(X)
        if flag == 0 and verbose:
            print("Iteration failed to converge, |g| =", gnorm)
        if inspect:
            return X, flag, fvec[0:k]
        return X, flag

    def hess(self, X0: LCState):
        """Hessian at given state as object derived from LinearOperator
        The class is a closure"""

        X0 = view_as_lc(np.copy(X0.x), X0.N)  # Copy to local storage

        class LCHess(LinearOperator):
            def __init__(self, F: LCFunc):
                super().__init__(dtype=float, shape=X0.x.shape * 2)
                self.X = view_as_lc(np.copy(X0.x), X0.N)
                self.aux = LCFunc.LCFuncAux(X0, F.wb1)
                # Copy fields from F
                self.wb = F.wb
                self.wb1 = F.wb1
                self.wa = F.wa
                self.wp1 = F.wp1
                self.wp0 = F.wp0
                self.wv = F.wv
                self.sp = F.sp

            def update(self, X: LCState):
                """Substitute with another state in-place. Save malloc time."""
                self.X.x[:] = X.x
                self.aux.update(X, self.wb1)

            def _matvec(self, v, out=None, part="all"):
                # assert v.shape[0] == X0.N, "Input vector dimension inconsistent with LC state"
                N = self.X.N
                h = self.X.h
                aux = self.aux
                # print(v.shape)
                V = view_as_lc(v, N)  # Input vector converted
                if out is None:
                    G = LCState(N)  # Storage for gradient
                else:
                    G = view_as_lc(out, N)  # Write to existing memory
                V.phi[:] -= np.average(V.phi)  # Project phi delta

                # The contribution dP part in P gradient
                if part == 0 or part == "all":
                    # Bulk energy
                    G.p11[:] += 4 * self.wb * h**2 * \
                        ((self.wb1 + 6 * self.X.p11**2 + 2 * self.X.p12**2) * V.p11
                         + 4 * self.X.p11 * self.X.p12 * V.p12)
                    G.p12[:] += 4 * self.wb * h**2 * \
                        ((self.wb1 + 6 * self.X.p12**2 + 2 * self.X.p11**2) * V.p12
                         + 4 * self.X.p11 * self.X.p12 * V.p11)
                    # Elastic energy is linear
                    G.p11[:] += -2 * (np.diff(V.p11, n=2, axis=0, prepend=0, append=0)
                                      + np.diff(V.p11, n=2, axis=1, prepend=0, append=0))
                    G.p12[:] += -2 * (np.diff(V.p12, n=2, axis=0, prepend=0, append=0)
                                      + np.diff(V.p12, n=2, axis=1, prepend=0, append=0))
                if part == 2 or part == "all":
                    # Anchoring energy is linear. Constant term removed
                    G.p11[:] += 2 * self.wa * aux.Dphi2 * V.p11
                    G.p12[:] += 2 * self.wa * aux.Dphi2 * V.p12
                if part == 1 or part == "all":
                    # Void energy is linear
                    # (Note: 1-φ changed to (1-φ)^2)
                    G.p11[:] += 2 * self.wv * h**2 * \
                        (1. - aux.phi_avg)**2 * V.p11
                    G.p12[:] += 2 * self.wv * h**2 * \
                        (1. - aux.phi_avg)**2 * V.p12

                # The contribution of other parts
                dphi_avg = np_avg(np_avg(V.phi, axis=0, prepend=0, append=0),
                                  axis=1, prepend=0, append=0)  # δφ at nodes, (N-1)*(N-1)
                if part == 3 or part == "all":
                    # Perimeter
                    G.phi[:] += self.wp0 * h**2 * \
                        (2. - 12. * (self.X.phi - self.X.phi**2)) * V.phi
                    G.phi[:] -= self.wp1 * (np.diff(V.phi, n=2, axis=0, prepend=0, append=0)
                                            + np.diff(V.phi, n=2, axis=1, prepend=0, append=0))
                d_pinkie = 4 * (self.X.p11 * V.p11 +
                                self.X.p12 * V.p12)  # δ[tr(P^2)]
                if part == 2 or part == "all":
                    # Anchoring penalty
                    Dxdphi = np_avg(np.diff(V.phi, axis=0, prepend=0, append=0),
                                    axis=1, prepend=0, append=0)
                    Dydphi = np_avg(np.diff(V.phi, axis=1, prepend=0, append=0),
                                    axis=0, prepend=0, append=0)
                    G.p11[:] += self.wa * 2 * ((2 * self.X.p11 + self.sp) * aux.Dphi_x * Dxdphi
                                               + (2 * self.X.p11 - self.sp) * aux.Dphi_y * Dydphi)
                    G.p12[:] += self.wa * 2 * ((2 * self.X.p12 * aux.Dphi_x + self.sp * aux.Dphi_y) * Dxdphi
                                               + (2 * self.X.p12 * aux.Dphi_y + self.sp * aux.Dphi_x) * Dydphi)
                    pinkie = aux.trP2 + .5 * self.sp**2
                    d_flutter = 2 * self.sp * (V.p11 * aux.Dphi_x + self.X.p11 * Dxdphi
                                               + V.p12 * aux.Dphi_y + self.X.p12 * Dydphi)
                    d_twilight = 2 * self.sp * (V.p12 * aux.Dphi_x + self.X.p12 * Dxdphi
                                                - V.p11 * aux.Dphi_y - self.X.p11 * Dydphi)
                    # Apply \bar δ_x^* to this quantity
                    d_foo = d_pinkie * aux.Dphi_x + pinkie * Dxdphi + d_flutter
                    # Apply \bar δ_y^* to this quantity
                    d_bar = d_pinkie * aux.Dphi_y + pinkie * Dydphi + d_twilight
                    G.phi[:] -= self.wa * (np_avg(np.diff(d_foo, axis=0), axis=1)
                                           + np_avg(np.diff(d_bar, axis=1), axis=0))
                if part == 1 or part == "all":
                    # Void penalty
                    # (Note: 1-φ changed to (1-φ)^2)
                    rarity = (aux.phi_avg - 1) * dphi_avg  # δ[(1-\bar φ)^2]
                    G.p11[:] += 4 * self.wv * h**2 * rarity * self.X.p11
                    G.p12[:] += 2 * self.wv * h**2 * rarity * self.X.p12
                    G.phi[:] += self.wv * h**2 * np_avg(np_avg(aux.trP2 +(aux.phi_avg-1) * d_pinkie,
                                                               axis=0),
                                                        axis=1)
                # Project phi gradient
                G.phi[:] -= np.average(G.phi)
                return G.x

            def _matmat(self, M):
                """Matrix-matrix multiplication subroutine"""
                assert M.shape[0] == len(
                    self.X.x), "Input vector dimension inconsistent with LC state"
                # Fortran ordering makes columns contiguous
                G = np.zeros(M.shape, order='F')
                for i in range(M.shape[1]):
                    self._matvec(M[:, i], out=G[:, i])
                return G

        return LCHess(self)

    def newton(self, X0: LCState,
               damp=True, damp_value=0.5, damp_threshold=0.5,
               maxiter=None, tol=1e-8, maxsubiter=50, subtol=None, verbose=0):
        """
        Solve for critical point with Newton method. Inverse of Hessian is
        approxiamted via inexact GMRes.
        Because Hessian computation is very costly, only use this method when
        the initial value is very close.

        Options
        -------
        damp : bool
            Whether to continually apply damp to step length (always applied
            if gradient is large)
        damp_value : float
            Default damp value
        damp_threshold : float
            Threshold value of gradient norm, under which damp is removed
        maaxiter : None | int
        tol : float
        maxsubiter : int
            Maximum number of sub-iterations in GMRes algorithm
        subtol : float
            Relative tolerance of GMRes
        """
        if maxiter is None:
            maxiter = 200 * X0.N**2
        if subtol is None:
            subtol = tol

        X = view_as_lc(np.copy(X0.x), X0.N)
        aux = self.LCFuncAux(X, self.wb1)
        H = self.hess(X)
        for k in range(maxiter):
            aux.update(X, self.wb1)
            g = self.grad(X, aux=aux)
            gnorm = norm(g)
            if verbose >= 2 or verbose and k == 0:
                print("Iteration %d, |g| = %.6e" % (k, gnorm))
            if gnorm < tol:
                if verbose:
                    print("Newton's iter converges at No.",
                          k, "with |g| =", gnorm)
                break
            # Newton direction
            d, code = gmres(H, g, x0=g, maxiter=maxsubiter,
                            restart=maxsubiter // 5, rtol=subtol)
            if verbose >= 2:
                print("GMRes exit flag at iteration %d: %d" % (k, code))
            if damp or gnorm > damp_threshold:
                X.x[:] -= damp_value * d
            else:
                X.x[:] -= d
            H.update(X)
            if np.any(np.isnan(X.x)):
                print("NAN at iteration No.", k, file=sys.stderr)
        return X


def square_grid(N, centre=False):
    """@returns Coordinate mesh grid
    We use the `ij` indexing order, i.e. first index stand for x, and the second for y. """
    if centre:
        return np.meshgrid(np.linspace(.5 / N, 1 - .5 / N, N),
                           np.linspace(.5 / N, 1 - .5 / N, N),
                           indexing='ij')
    return np.meshgrid(np.arange(0, N + 1) / N,
                       np.arange(0, N + 1) / N,
                       indexing='ij')


def get_director(X: LCState):
    # Bound away from zero
    r = np.maximum(1e-14, np.sqrt(X.p11**2 + X.p12**2))
    c = X.p11 / r  # cos(2γ). s=B/C is positive
    nx = r * np.sqrt((1. + c) / 2)
    ny = r * np.sqrt((1. - c) / 2) * np.sign(X.p12)
    return nx, ny


def plot_P(X:LCState, ax=None, s_range=None, scale=3.0, density=0.1, colorbar=True,
           phi_form="bound", energy=None):
    """Plot P on given axis"""
    S = 2 * np.sqrt(X.p11**2 + X.p12**2)
    N = X.N
    h = X.h
    img = ax.imshow(S.T, cmap="RdBu", vmin=0,vmax=s_range, origin="lower",
                    extent=(0.5 * h, 1 - 0.5 * h, 0.5 * h, 1 - 0.5 * h))
    if colorbar:
        plt.colorbar(img, ax=ax)
    # Plot director as vector field
    # No arrow head because direction does not matter
    # After calling imshow() the y axis is flipped so we have to flip the y component
    # Plot region as contour lines
    xc, yc = square_grid(N, centre=True)
    PHI = np.zeros([N, N])  # Pad X.phi with 0
    PHI[1:N - 1, 1:N - 1] = X.phi
    if phi_form == "bound":
        # Plot the boundary line {φ=0.5}
        ax.contour(xc, yc, PHI, levels=[0.5], colors="white")
    elif phi_form == "contours":
        # Plot all contours of phi
        ax.contour(xc, yc, PHI, cmap="twilight",
                     linestyles='solid')  # Levels of phi
    # Plot director field
    # only applicable when scale is small
    if scale < 200.:
        nx, ny = get_director(X)
        # Sample at random sites and plot director
        numArrows = int(density * 10000)
        I = ran.randint(N - 2, size=(1, numArrows))
        J = ran.randint(N - 2, size=(1, numArrows))
        U = nx[I, J] / scale
        V = ny[I, J] / scale
        ax.quiver((I + 1) * h, (J + 1) * h, U, V,
                    headlength=0, headwidth=0, headaxislength=0, pivot="middle",
                    scale=scale, color=(.6, .8, .5, .6), width=0.01)
    if energy is not None:
        ax.set_title("$s$ ($E=\\tt %.3e$)"%energy)
    return img


def plot_state(X: LCState, fig=None, P_only=False, **kwargs):
    # Initialize fig window
    if fig is None:
        fig = plt.gcf()
    fig.set_figwidth(4.8 if P_only else 9.6)
    fig.set_figheight(3.6)

    # Plot P
    if P_only:
        sub1 = fig.add_subplot(1, 1, 1)
    else:
        sub1 = fig.add_subplot(1, 2, 1)
    img1 = plot_P(X, ax=sub1, **kwargs)

    # Plot phi
    if not P_only:
        sub2 = fig.add_subplot(1, 2, 2)
        # Transpose because we use IJ index order
        PHI = np.zeros([X.N, X.N])  # Pad X.phi with 0
        PHI[1:X.N - 1, 1:X.N - 1] = X.phi
        img2 = sub2.imshow(PHI.T, origin='lower', extent=(0, 1, 0, 1))
        plt.colorbar(img2, ax=sub2)
        sub2.set_title("$\phi$")
    return img1, img2


parser = ArgumentParser(prog="nlc2d_stagger",
                        description="Shape optimization in nematic liquid crystals")
parser.add_argument('-r', "--restart", action="store_true", default=False,
                    help="Compute everything from scratch")
parser.add_argument('-N', type=int, action="store", default=128,
                    help="Size of discrete grid")
parser.add_argument('-lam', type=float, action="store", default=5e-7,
                    help="Characteristic length")
parser.add_argument('-o', "--output-dir", action="store", default="test",
                    help="Output directory name")
parser.add_argument('-i', "--init", action="store", default="square",
                    help="Initial shape (circle,ellipse,mickey,square)")
parser.add_argument('-f', "--file", action="store", default=None,
                    help="Initial state from file")
# Optional parameters
parser.add_argument('--epsr', '--capillary-width', type=float, action="store", default=0.005,
                    help="Capillary width")
parser.add_argument('--vol', '--volume', type=float, action="store", default=0.04,
                    help="Volume of Ω after scaling")
parser.add_argument('--wa', '--anchor-penalty', type=float, action="store", default=1e8,
                    help="Penalty factor of anchoring energy")
parser.add_argument('--wv', '--void-penalty', type=float, action="store", default=4e14,
                    help="Penalty factor of void")
parser.add_argument('--wphi', '--perimeter-penalty', type=float, action="store", default=1e8,
                    help="Penalty factor of perimeter")

if __name__ == "__main__":
    args = parser.parse_args()
    ran.seed(20240117)
    N = args.N
    lam = args.lam
    NN = 3 * N**2 - 8 * N + 6  # Dimension of vector
    OUTDIR = args.output_dir
    print("Solving the problem on %dx%d grid with scale %.1e" % (N, N, lam))
    print("Writing results to", './' + OUTDIR + "/", flush=True)
    if args.restart:
        # Restart and remove all existing results
        for name in "solution.npy", "eigvals.npy", "eigvecs.npy":
            try:
                os.remove(OUTDIR + os.sep + name)
            except FileNotFoundError:
                pass
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    c = LCConfig(N=args.N, lam=args.lam, epsr=args.epsr, vol=args.vol,
                 wa=args.wa, wv=args.wv, wphi=args.wphi)
    save_lc_config(OUTDIR + os.sep + "params.json", c)

    # Initial shape (only used when no solution is saved)
    Xx, Yy = square_grid(N, centre=True)
    if args.init == "ellipse":
        phi0 = 1. * (((Xx - 0.5)**2 + 9 * (Yy - 0.5)**2) < 0.07)
    elif args.init == "square":
        # square
        r = .08
        phi0 = 1. * (np.abs(Xx - .5) < r) * (np.abs(Yy - .5) < r)
    elif args.init == "mickey":
        r0 = .1
        phi0 = ((Xx - .5)**2 + (Yy - (.5 - .3 * r0))**2) < r0**2
        phi0 |= (((Xx - (.5 + r0))**2 + (Yy - (.5 + .7 * r0))**2)
                 < .36 * r0**2)
        phi0 |= (((Xx - (.5 - r0))**2 + (Yy - (.5 + .7 * r0))**2)
                 < .36 * r0**2)
        phi0 = gblur(1.0 * phi0, N // 30)
    elif args.init == "eye":
        r = .1
        a = 3
        phi0 = ((Xx - 0.5) / r)**2 + (a * np.abs(Yy - 0.5) / r)**0.4 < 1
    else:
        # circle by default
        r = .08
        phi0 = 1.0 * (((Xx - 0.5)**2 + (Yy - 0.5)**2) < r**2)

    # Get smoothed initial phi field using reinitialization
    # phi0 is 0-1 indicator, so we change its sign
    phi0 = reinit(0.5 - phi0, 1. / N, args.vol)

    # Physical constants: B=6400, C=3500, L=4e-11
    # PLEASE KEEP THESE CONSTANTS FIXED AND CHANGE λ ONLY
    FF = LCFunc(char_len=lam, capil_rel=args.c0,
                Landau_coeff=(6400, 3500), elastic_coeff=4e-11,
                w_anch=args.wa, w_void=args.wv, w_per=args.wphi)
    print("""Full energy functional:
            E(x) = ∫ [ ½*|∇P|^2 + \\bar{λ}^2 F_b(P) ]
                + w_void * ∫ ½*(1-φ)*|P|^2
                + w_anchor * ∫ |(P+s/2)*∇φ|^2
                + w_phase_0 * ∫ (1-φ)^2*φ^2
                + w_phase_1 * ∫ ½*|∇φ|^2""")
    print("with     \\bar{λ} = %.3e, V = %.3e" %
          (np.sqrt(FF.wb), np.sum(phi0) / N**2))
    print("and      w_void=%.3e, w_anchor=%.3e, w_phase_1=%.3e, w_phase_0=%.3e" %
          (FF.wv, FF.wa, FF.wp1, FF.wp0))

    ############################################################################
    #############################  MINIMUM MODULE  #############################
    ############################################################################

    # tracemalloc.start()
    state_f = os.path.join(OUTDIR, "solution.npy")
    if args.file is not None:
        # Explicit input file
        print("Loading initial value from file...", flush=True)
        X = load_lc(args.file, N)
    elif os.path.exists(state_f):
        # Load directly
        print("Loading state file saved last time...", flush=True)
        X = load_lc(state_f, N)
    else:
        # Initialize X
        x0 = np.zeros(3 * N**2 - 8 * N + 6)
        X = view_as_lc(x0, N)
        X.phi[:] = phi0[1:N - 1, 1:N - 1]
        # X.p11[:] = ran.randn(N - 1, N - 1)
        # X.p12[:] = ran.randn(N - 1, N - 1)
    # Optimize energy functional with SciPy api
    # res = optimize.minimize(FF.energy_vec, x0=np.array(X), args=(N,), jac=FF.grad_vec,
    #                    method="L-BFGS-B", tol=1e-8, options={'maxiter': 50000, 'disp': False})
    # X = view_as_lc(res.x, N)
    X, flag = FF.grad_descent(
        X, maxiter=160000, eta=2e-3, tol=1e-8, bb=True, inspect=False)
    np.save(state_f, X.x)
    # if sys.platform in ["win32", "darwin"]:
    # # Graphic output for desktop systems only
    #     plt.plot(fvec)
    #     plt.show()

    fig = plt.figure(num=1, figsize=(9.6, 3.6))
    plot_state(X, fig, scale=3.0, density=0.1,
               phi_form="bound", energy=FF.energy(X))
    fig.savefig(os.path.join(OUTDIR, "critical_state.pdf"))
    if sys.platform in ["win32", "darwin"]:
        # Graphic output for desktop systems only
        plt.show()

    ############################################################################
    ###########################  EIGENVALUE MODULE  ############################
    ############################################################################
    HH = FF.hess(X)  # Hessian as LinearOperator instance
    # y1 = view_as_lc(np.zeros(NN), N)
    # y1.phi[:] = 1.  # Volume constraint
    # y2 = view_as_lc(np.zeros(NN), N)
    # y2.phi[:] = Xx[1:N - 1, 1:N - 1] - 0.5  # X centre constraint
    # y3 = view_as_lc(np.zeros(NN), N)
    # y3.phi[:] = Yy[1:N - 1, 1:N - 1] - 0.5  # Y centre constraint
    # constr_vecs = np.zeros([NN, 3], order='F')
    # constr_vecs[:, 0] = y1.x
    # constr_vecs[:, 1] = y2.x
    # constr_vecs[:, 2] = y3.x
    # if os.path.exists(os.path.join(OUTDIR, "eigvecs.npy")) \
    #         and os.path.exists(os.path.join(OUTDIR, "eigvals.npy")):
    #     print("Loading eigenvalues and eigenvectors...")
    #     vals = np.load(os.path.join(OUTDIR, "eigvals.npy"))
    #     vecs = np.load(os.path.join(OUTDIR, "eigvecs.npy"))
    # else:
    #     print("Solving for eigenvalues...")
    #     vals, vecs = lobpcg(HH, np.eye(NN, 4, order="F"),
    #                         largest=False, tol=1e-8, maxiter=80000)
    #     print(vals)
    #     np.save(os.path.join(OUTDIR, "eigvals.npy"), vals)
    #     np.save(os.path.join(OUTDIR, "eigvecs.npy"), vecs)
    # for i in range(len(vals)):
    #     fig.clf()
    #     fig = plot_state(view_as_lc(X.x + .03 * N * vecs[:, i], N), fig,
    #                      scale=3.0, density=0.1, phi_form="bound")
    #     fig.savefig(os.path.join(OUTDIR, "perturb_eigvec_%d.pdf" % (i + 1)))

    # for p in 0, 1, 2, 3, 'all':
    #     # Verify correctness of gradient and Hessian
    #     dx = ran.randn(NN) / np.sqrt(NN)
    #     dx_lc = view_as_lc(dx, N)
    #     dx_lc.phi -= np.average(dx_lc.phi)
    #     eps = 1e-5
    #     df = FF.energy_vec(X.x + eps * dx, N, part=p) - FF.energy_vec(X.x - eps * dx, N, part=p)
    #     g = FF.grad(X, part=p)
    #     dg = FF.grad_vec(X.x + eps * dx, N, part=p) - g
    #     Hdx = HH._matvec(dx, part=p)
    #     print(p)
    #     print(abs(df / (2 * eps) - np.dot(g, dx)))
    #     print(norm(dg / eps - Hdx))

    # Re-optimize along negative eigenvalue (if any)
    # run = 0
    # while vals[0] < 0 and run < 5:
    #     print("=============")
    #     print("Re-running minimization along smallest negative eigenvalue...")
    #     x_new = optimize.fmin_cg(FF.energy_vec, X.x + 1e-3 * N * vecs[:, 0], fprime=FF.grad_vec,
    #                              args=(N,), gtol=1e-8, maxiter=40000, disp=True)
    #     # X, flag = FF.grad_descent(X, maxiter=80000, eta=1e-3, tol=1e-6, bb=True)
    #     X.x[:] = x_new
    #     run += 1
    #     print("Energy:", FF.energy(X))
    #     fig.clf()
    #     fig = plot_state(X, fig=fig, scale=3.0, density=0.1, phi_form="bound", energy=FF.energy(X))
    #     fig.savefig(os.path.join(OUTDIR, "critical_rerun_%d.pdf" % run))
    #     np.save(os.path.join(OUTDIR, "critical_rerun_%d.npy" % run), X.x)
    #     HH = FF.hess(X)
    #     print("Solving for eigenvalues again...")
    #     vals, vecs = lobpcg(HH, vecs[:, 0:1], largest=False, tol=1e-8, maxiter=40000)
    #     print("Smallest eigenvalue after run No. %d: %.6f" % (run, vals[0]))

    ############################################################################
    ##########################  SADDLE POINT MODULE  ###########################
    ############################################################################

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot, limit=10)
