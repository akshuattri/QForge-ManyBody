# QForge-ManyBody

**QForge-ManyBody** is an open-source Python framework under active development for computational quantum many-body physics and open quantum systems.

The project provides a modular environment for implementing, testing, and extending numerical methods commonly used in condensed matter physics, quantum statistical mechanics, and open quantum systems. It is intended as a reusable research software framework for computational experiments, numerical benchmarking, and reproducible scientific workflows.

---

## Features

### Numerical Methods

* Exact Diagonalization
* Lindblad Master Equation Dynamics
* Time Evolution Algorithms
* Lanczos Solver (framework)
* Krylov Solver (framework)

### Quantum Models

* Heisenberg Spin Chain
* Ising Model
* SSH Model
* Hatano–Nelson Model

### Utilities

* Scientific benchmarking
* Validation scripts
* Example simulations
* Scientific visualization
* Reproducible computational workflows

---

## Repository Structure

```text
QForge-ManyBody/
│
├── qforge/                 # Core Python package
│   ├── analysis/
│   ├── benchmarks/
│   ├── core/
│   ├── geometry/
│   ├── hamiltonians/
│   ├── models/
│   ├── open_systems/
│   ├── physics/
│   ├── solvers/
│   ├── transport/
│   ├── validation/
│   └── visualization/
│
├── examples/
├── papers/
├── scripts/
├── tests/
├── tutorials/
├── docs/
├── results/
└── validation_results/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/akshuattri/QForge-ManyBody.git
```

Enter the project directory

```bash
cd QForge-ManyBody
```

Create a Python environment

```bash
conda create -n qforge python=3.10
conda activate qforge
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Start

Run the validation suite

```bash
PYTHONPATH=. python scripts/validate.py
```

Run benchmark calculations

```bash
PYTHONPATH=. python scripts/benchmark.py
```

Run all example workflows

```bash
PYTHONPATH=. python run_everything.py
```

Run an individual example

```bash
PYTHONPATH=. python examples/01_basic_usage.py
```

---

# Example Simulations

The repository currently contains example workflows including

* Exact diagonalization
* Lindblad dynamics
* Quantum spin chains
* SSH model
* Hatano–Nelson model
* Paper reproduction examples

---

# Validation

QForge-ManyBody includes validation and benchmark scripts intended to assist development and testing of implemented numerical methods.

Run

```bash
PYTHONPATH=. python scripts/validate.py
```

Benchmark examples can be executed using

```bash
PYTHONPATH=. python scripts/benchmark.py
```

---

# Documentation

Documentation is available in

```text
docs/
tutorials/
```

---

# Development Philosophy

The primary objective of QForge-ManyBody is to build reusable, modular software for computational quantum physics.

Rather than providing a single-purpose application, the framework is designed to support numerical experimentation, algorithm development, benchmarking, and reproducible computational research.

---

# Roadmap

Future development will focus on

* Improved exact diagonalization algorithms
* Additional open quantum system solvers
* Quantum trajectory methods
* Expanded topological models
* Tensor-network interfaces
* Additional benchmark reproductions
* Improved documentation
* Continuous validation of implemented methods

---

# Contributing

Contributions, suggestions, bug reports, and discussions are welcome.

Please open an Issue or Pull Request.

---

# Citation

If you use QForge-ManyBody in academic work, please cite the repository.

```text
Akshu Attri

QForge-ManyBody

https://github.com/akshuattri/QForge-ManyBody
```
---

