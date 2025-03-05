#!/usr/bin/env python3
import torch
import numpy as np


from deepinv.optim.data_fidelity import L2
from deepinv.optim.optim_iterators.hqs import gStepHQS
from deepinv.optim.optim_iterators.pgd import gStepPGD, fStepPGD
from deepinv.optim.optim_iterators import OptimIterator, fStep

from deepinv.optim.utils import conjugate_gradient


def get_custom_init(y, physics):
    density = physics.nufft.pipe(max_iter=20)

    return physics.nufft.A_adjoint(density * y)


def get_DPIR_params(sigma=1, xi=1, lamb=2, stepsize=1, lipschitz_cst=1.0, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    # sigma_denoiser = np.linspace(s1, s2, n_iter).astype(np.float32)
    # stepsize = (sigma_denoiser / max(1e-6, s2)) ** 2
    if isinstance(stepsize, str):
        stepsize = eval(stepsize)
    sigma_denoiser = (sigma * xi ** np.arange(n_iter)).astype(np.float32)
    stepsize = np.ones_like(sigma_denoiser) * stepsize
    # return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}
    return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}


class PreconditionedHQSIteration(OptimIterator):
    r"""
    Single iteration of half-quadratic splitting.

    Class for a single iteration of the Half-Quadratic Splitting (HQS) algorithm for minimising :math:` f(x) + \lambda g(x)`.
    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{\gamma f}(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\sigma \lambda g}(u_k).
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` and :math:`\sigma` are step-sizes. Note that this algorithm does not converge to
    a minimizer of :math:`f(x) + \lambda  g(x)`, but instead to a minimizer of
    :math:`\gamma\, ^1f+\sigma \lambda g`, where :math:`^1f` denotes
    the Moreau envelope of :math:`f`

    """

    def __init__(self, precond=None, **kwargs):
        super(PreconditionedHQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQSPrecond(**kwargs)
        self.requires_prox_g = True
        self.precond = Precond(precond)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        General form of a single iteration of splitting algorithms for minimizing :math:`F =  f + \lambda g`, alternating
        between a step on :math:`f` and a step on :math:`g`.
        The primal and dual variables as well as the estimated cost at the current iterate are stored in a dictionary
        $X$ of the form `{'est': (x,z), 'cost': F}`.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        if not self.g_first:
            z = self.f_step(
                x_prev, cur_data_fidelity, cur_params, y, physics, precond=self.precond
            )
            z = z.cfloat()  # disgusting but for some reason the above casts to double
            x = self.g_step(z, cur_prior, cur_params)
        #         else:
        #             Not implemented
        x = self.relaxation_step(x, x_prev, cur_params["beta"])
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, z), "cost": F}


class fStepHQSPrecond(fStep):
    r"""
    HQS fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepHQSPrecond, self).__init__(**kwargs)
        self.metric = L2()
        self._init_prox = None

    def forward(self, x, cur_data_fidelity, cur_params, y, physics, precond=None):
        r"""
        Single proximal step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        # in the standard case we would return this: (useful for debugging)
        # return cur_data_fidelity.prox(x, y, physics, gamma=cur_params["stepsize"])
        # instead we do that:
        return self.prox_l2_metric(
            x, y, physics, gamma=cur_params["stepsize"], precond=precond
        )

    def prox_l2_metric(
        self, z, y, physics, gamma, precond=None, max_iter=50, tol=1e-3, **kwargs
    ):
        r"""
        Computes proximal operator of :math:`f(x) = \frac{1}{2}\|Ax-y\|^2`, i.e.,

        .. math::

            \underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor z: signal tensor
        :param float gamma: hyperparameter of the proximal operator
        :return: (torch.Tensor) estimated signal tensor

        """
        lipschitz_cst = physics.nufft.get_lipschitz_cst()
        if precond is None:
            b = physics.A_adjoint(y, **kwargs) + 1 / gamma * z

            def H(x):
                return physics.A_adjoint(physics.A(x, **kwargs)) + 1 / gamma * x

        else:
            b = physics.A_adjoint(y, **kwargs) + 1 / gamma * precond.update_grad(
                {"stepsize": 1.0 / lipschitz_cst}, physics, z
            )

            def H(x):
                return physics.A_adjoint(
                    physics.A(x, **kwargs)
                ) + 1 / gamma * precond.update_grad(
                    {"stepsize": 1.0 / lipschitz_cst},
                    physics,
                    x,
                )

        x = conjugate_gradient(H, b, init=self._init_prox, max_iter=max_iter, tol=tol)
        self._init_prox = x.detach().clone()
        return x


class Precond:
    def __init__(self, name, theta1=0.2, theta2=2, delta=1 / 1.633):
        self.it = 0
        self.name = name
        self.theta1 = theta1
        self.theta2 = theta2
        self.delta = delta

    def get_alpha(self, s, m):
        alphas = np.linspace(0, 1, 1000)
        # line search of alpha

    def update_grad(self, cur_params, physics, grad, *args, **kwargs):
        if self.name == "static":
            grad = self._update_grad_static(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "cheby":
            grad = self._update_grad_cheby(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "dynamic":
            grad = self._update_grad_dynamic(cur_params, physics, grad, *args, **kwargs)
        return grad

    def _update_grad_static(self, cur_params, physics, grad_f, *args, **kwargs):
        """Update the gradient with the static preconditioner"""
        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))

        grad_f_preconditioned *= -alpha
        grad_f_preconditioned += 2 * grad_f

        return grad_f_preconditioned

    def _update_grad_cheby(self, cur_params, physics, grad_f, *args, **kwargs):
        """Update the gradient with the static cheby preconditioner"""
        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))
        grad_f_preconditioned *= -(10 / 3) * alpha
        grad_f_preconditioned += 4 * grad_f
        return grad_f_preconditioned

    def _update_grad_dynamic(self, cur_params, physics, grad_f, grad_f_prev, x, x_prev):
        """Update the gradient with the dynamic preconditioner"""
        s = x - x_prev
        m = grad_f - grad_f_prev

        # precompute dot products
        sf = s.squeeze(0).squeeze(0).reshape(-1)
        mf = m.squeeze(0).squeeze(0).reshape(-1)

        ss = torch.vdot(sf, sf).real
        sm = torch.vdot(sf, mf).real
        mm = torch.vdot(mf, mf).real

        if ss == 0 or mm == 0:
            return grad_f

        for a in np.linspace(0, 1, 1000):
            sv = a * ss + (1 - a) * sm
            vv = (a**2) * ss + ((1 - a) ** 2) * mm + (2 * a * (1 - a)) * sm
            if sv / ss >= self.theta1 and vv / sv <= self.theta2:
                break
        vf = a * sf + (1 - a) * mf
        tau = ss / sv - torch.sqrt((ss / sv) ** 2 - ss / vv)
        tmp = sv - tau * vv
        grad_f_preconditioned = tau * grad_f

        if tmp >= (
            self.delta * torch.sqrt(ss - 2 * tau * sv + tau**2 * vv) * torch.sqrt(vv)
        ):
            u = sf - tau * vf
            u = u.dot(grad_f.squeeze(0).squeeze(0).reshape(-1)) * u
            u = u.reshape(grad_f.shape)
            grad_f_preconditioned += u / tmp

        return grad_f_preconditioned


class PreconditionedPnPIteration(OptimIterator):
    """Implement the preconditioned PnP algorithm from Hong et al. 2024."""

    def __init__(self, preconditioner="dynamic", **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        self.requires_prox_g = True
        if self.g_first:
            raise ValueError(
                "The preconditioned PnP algorithm should start with a step on f."
            )

        self.preconditioner = Precond(preconditioner)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        try:
            x_prev, x_prev_prev, grad_f_prev = X["est"]
        except ValueError:
            x_prev, grad_f_prev = X["est"]
            x_prev_prev = x_prev.clone() # bootstrap
        k = 0 if "it" not in X else X["it"]

        # TODO add the preconditioner step
        grad_f = cur_data_fidelity.grad(x_prev, y, physics)

        grad_f_precond = self.preconditioner.update_grad(
            cur_params, physics, grad_f, grad_f_prev, x_prev, x_prev_prev
        )
        z = x_prev - cur_params["stepsize"] * grad_f_precond
        x = self.g_step(z, cur_prior, cur_params)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )

        return {"est": (x, x_prev, grad_f), "cost": F, "it": k + 1}
