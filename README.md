# Python-Hamiltonian-Monte-Carlo-Sampler

Minimal Python implementation of Hamiltonian Monte Carlo using a leapfrog integrator to sample a Gaussian distribution and estimate expectation values.

---

## Overview

This repository provides a simple Python implementation of the **Hamiltonian Monte Carlo (HMC)** algorithm. The code demonstrates how Hamiltonian dynamics combined with a Metropolis acceptance step can efficiently sample from a target probability distribution. As an example, the script samples a **one-dimensional Gaussian distribution** and computes the expectation value of \(q^4\). The numerical distribution is compared against the analytic Gaussian probability density.

---

## Features

- Implements the **Hamiltonian Monte Carlo algorithm** using a leapfrog integrator.
- Samples from a probability distribution defined through a potential energy function \(U(q)\).
- Demonstrates sampling correctness by comparing **numerical and analytic distributions**.
- Generates visualization plots for the sampled distribution and **Markov chain trace**.
- Computes expectation values using **Monte Carlo averaging**.

---

## Algorithmic Characteristics

- Uses **Hamiltonian dynamics** to propose distant states in phase space.
- **Leapfrog integration** ensures approximate energy conservation and numerical stability.
- **Metropolis acceptance step** guarantees the correct stationary distribution.
- Avoids the slow diffusive behavior typical of **random-walk Monte Carlo methods**.

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib

---

## Installation

Install required packages:

```bash
pip install numpy matplotlib
```

---

## Usage

Clone or download this repository and run:

```bash
python HMC.py
```

The program will:

1. Perform Hamiltonian Monte Carlo sampling.
2. Estimate the expectation value of \(q^4\).
3. Generate two plots:
   - Histogram of sampled values compared to the analytic Gaussian distribution.
   - Trace plot of the Markov chain.

---

## Example Output

```
Expectation: 2.9634380659551423
Acceptance rate: 0.96305
```

The script also saves the following figures:

```
hmc_histogram.png
hmc_trace.png
```

---
## Customization

You can easily modify the script to sample different probability distributions or compute different expectation values.

- Edit the **potential function** `U(q)` to define a new target distribution.
- Modify `gradU(q)` accordingly to provide the derivative of the potential.

The function `f(q)` defines the **observable** whose expectation value is computed.  
You can change this function to estimate different quantities, for example:

```python
def f(q):
    return 1        # normalization check

def f(q):
    return q        # mean value

def f(q):
    return q**2     # variance

def f(q):
    return q**4     # higher moments
```

Adjust HMC parameters in the `integrate()` call if needed:

- `epsilon` - leapfrog step size  
- `L` - number of leapfrog steps per trajectory  
- `N` - number of Monte Carlo samples

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---
## Reference

The Hamiltonian Monte Carlo algorithm implemented here follows the standard formulation described in:
Radford M. Neal,  
*MCMC Using Hamiltonian Dynamics*,  
Handbook of Markov Chain Monte Carlo (2011).  
https://arxiv.org/abs/1206.1901

---

## Note on Momentum Negation

In the canonical presentation of Hamiltonian Monte Carlo, the momentum variable is often negated after the leapfrog trajectory to ensure exact time-reversal symmetry of the proposal.

In this minimal implementation, the momentum is **resampled independently at every HMC step and discarded after the trajectory**, so explicit momentum flipping is not required. This keeps the code simple while preserving the sampler's correctness.
