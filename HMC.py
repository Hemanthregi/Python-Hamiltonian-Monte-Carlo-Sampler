# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:47:04 2026

@author: heman
"""

import numpy as np
import matplotlib.pyplot as plt


# Leapfrog integrator for Hamiltonian dynamics
def leapfrog(q, p, epsilon, L, U, gradU):
    # First half-step update for momentum
    p_half = p - 0.5 * epsilon * gradU(q)

    for step in range(L):

        # Full step update for position
        q_next = q + epsilon * p_half
        

        # Full momentum update except at final step
        if step != L - 1:
            p_next = p_half - epsilon * gradU(q_next)

            q = q_next
            p_half = p_next

    # Final half-step update for momentum
    p_next = p_half - 0.5 * epsilon * gradU(q_next)

    return q_next, p_next


# Single Hamiltonian Monte Carlo step
def HMC(q, epsilon, L, U, gradU):

    # Sample momentum from a Gaussian distribution
    p = np.random.normal()

    # Simulate Hamiltonian dynamics using leapfrog
    q_new, p_new = leapfrog(q, p, epsilon, L, U, gradU)

    # Compute Hamiltonian values
    H_current  = U(q)     + 0.5 * p * p
    H_proposed = U(q_new) + 0.5 * p_new * p_new

    # Log acceptance ratio for the Metropolis test
    log_r = H_current - H_proposed

    accepted = False

    # Accept proposal with probability min(1, exp(log_r))
    if np.log(np.random.rand()) < log_r:
        q = q_new
        accepted = True

    return q, accepted


def integrate(f, U, gradU, q0=0, N=100000, epsilon=0.8, L=20):
    """
    Monte Carlo expectation value using Hamiltonian Monte Carlo (HMC).

    The HMC algorithm generates samples from a distribution proportional
    to exp(-U(q)).  These samples are then used to estimate the expectation
    value of a function f(q).

    Parameters
    ----------
    f : function
        Function whose expectation value will be computed.

    U : function
        Potential defining the sampling distribution.
        The sampler generates q values from a distribution
        proportional to exp(-U(q)).

    gradU : function
        Derivative of U(q), required to simulate Hamiltonian dynamics.

    q0 : float
        Initial value of q for the Markov chain.

    N : int
        Number of Monte Carlo samples.

    epsilon : float
        Step size used in the leapfrog integrator.

    L : int
        Number of leapfrog steps per HMC trajectory.

    Returns
    -------
    result : float
        Estimated expectation value of f(q).

    acceptance_rate : float
        Fraction of proposed states that were accepted.
    """

    q = q0
    accepted_count = 0
    values = []
    samples = []

    for _ in range(N):

        # Perform one HMC step
        q, accepted = HMC(q, epsilon, L, U, gradU)

        if accepted:
            accepted_count += 1

        # Use the current state of the chain in the estimator
        values.append(f(q))
        samples.append(q)

    values = np.array(values)
    samples = np.array(samples)

    # Monte Carlo estimate of the expectation value
    result = np.mean(values)

    acceptance_rate = accepted_count / N

    return result, acceptance_rate, samples


# Example potential, gradient, and observable
def U(q):
    return 0.5 * q**2

def gradU(q):
    return q

def f(q):
    return q**4

result, acc, samples = integrate(f, U, gradU)

print("Expectation:", result)
print("Acceptance rate:", acc)


#############################################################
#Visualization
#############################################################

#1. Histogram vs Analytic
plt.figure()

plt.hist(samples,
         bins=100,
         density=True,
         alpha=0.6,
         range=(-4,4),
         label="HMC samples")

x = np.linspace(-4,4,400)
gaussian = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

plt.plot(x, gaussian, 'r', linewidth=2, label="Exact Gaussian")

plt.xlabel("q")
plt.ylabel("Probability density")
plt.title("HMC sampling of Gaussian distribution")
plt.legend()
plt.savefig("hmc_histogram.png", dpi=300)
plt.show()

#2. Markov chain Trace
plt.figure()

plt.plot(samples[:1000])
plt.xlabel("Iteration")
plt.ylabel("q")
plt.title("HMC Markov Chain Trace")
plt.savefig("hmc_trace.png", dpi=300)
plt.show()



