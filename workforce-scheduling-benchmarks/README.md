# Workforce Scheduling Benchmarks

**Comparative analysis of workforce scheduling algorithms**

A research-oriented implementation and benchmarking framework for comparing different algorithmic approaches to the workforce scheduling problem. This repository demonstrates various optimization techniques applied to shift assignment problems common in service industries.

## 📋 Overview

Workforce scheduling is a fundamental operations research problem involving the assignment of workers to shifts while satisfying various constraints (availability, skills, labor regulations) and optimizing objectives (cost, coverage, fairness).

This repository implements and compares several classical approaches:
- **Greedy heuristics** - Fast, constructive algorithms
- **Genetic algorithms** - Metaheuristic evolutionary approach
- **Benchmarking framework** - Systematic comparison with metrics

## 🎯 Problem Formulation

Given:
- Set of workers with skills, availability, and preferences
- Set of shifts with demand requirements and skill needs
- Planning horizon (typically 1-4 weeks)

Constraints:
- Coverage: Each shift must meet minimum staffing requirements
- Skills: Workers must possess required skills for assigned shifts
- Availability: Workers can only be assigned to available time slots
- Workload: Limits on hours per day/week per worker
- Rest periods: Minimum time between shifts

Objectives:
- Minimize total labor cost
- Maximize shift coverage
- Maximize worker preference satisfaction
- Balance workload across workers

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Kwekudomp/workforce-scheduling-benchmarks.git
cd workforce-scheduling-benchmarks

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from data_generator import generate_scheduling_problem
from greedy_scheduler import GreedyScheduler
from genetic_scheduler import GeneticScheduler
from benchmark import compare_algorithms

# Generate synthetic problem
problem = generate_scheduling_problem(
    num_workers=20,
    num_shifts=10,
    num_days=7
)

# Run greedy algorithm
greedy = GreedyScheduler()
greedy_solution = greedy.solve(problem)
print(f"Greedy cost: {greedy_solution.total_cost}")

# Run genetic algorithm
genetic = GeneticScheduler(population_size=100, generations=50)
genetic_solution = genetic.solve(problem)
print(f"Genetic cost: {genetic_solution.total_cost}")

# Compare multiple algorithms
results = compare_algorithms(problem)
print(results)
```

### Run Benchmarks

```bash
# Run full benchmark suite
python benchmark.py

# Run with custom parameters
python benchmark.py --workers 30 --shifts 15 --days 14
```

## 📊 Algorithms Implemented

### 1. Greedy Heuristic

**Approach:** Iteratively assign workers to shifts based on a priority metric (lowest cost, highest preference, etc.)

**Complexity:** O(W × S) where W = workers, S = shifts

**Advantages:**
- Fast execution
- Guaranteed to find feasible solution (if one exists)
- Intuitive and explainable

**Disadvantages:**
- No optimality guarantee
- Can get stuck in local optima
- Doesn't explore solution space broadly

**Implementation:** `greedy_scheduler.py`

### 2. Genetic Algorithm

**Approach:** Evolutionary algorithm that evolves a population of candidate solutions through selection, crossover, and mutation.

**Complexity:** O(P × G × W × S) where P = population size, G = generations

**Advantages:**
- Explores broader solution space
- Can escape local optima
- Flexible - easy to add new constraints

**Disadvantages:**
- Slower than greedy
- Stochastic - results vary between runs
- Requires parameter tuning

**Implementation:** `genetic_scheduler.py`

## 📈 Benchmark Results

Performance comparison on problems of varying size:

| Problem Size | Greedy Time | Greedy Cost | GA Time | GA Cost | Improvement |
|--------------|-------------|-------------|---------|---------|-------------|
| Small (10W, 5S, 7D) | 0.02s | $1,250 | 2.1s | $1,180 | 5.6% |
| Medium (20W, 10S, 7D) | 0.08s | $2,840 | 8.3s | $2,620 | 7.7% |
| Large (50W, 20S, 14D) | 0.45s | $8,920 | 42.1s | $8,150 | 8.6% |

*Results are averaged over 10 random problem instances*

**Key Findings:**
- Greedy provides reasonable solutions very quickly
- Genetic algorithm improves solution quality by 5-10% at cost of longer runtime
- For large problems (>100 workers), greedy may be preferred for practical use
- GA performance highly dependent on parameter tuning

## 🔧 Repository Structure

```
workforce-scheduling-benchmarks/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data_generator.py         # Synthetic problem instance generator
├── greedy_scheduler.py       # Greedy heuristic implementation
├── genetic_scheduler.py      # Genetic algorithm implementation
├── benchmark.py              # Benchmarking and comparison script
│
├── examples/
│   └── example_usage.ipynb   # Jupyter notebook with examples
│
├── data/
│   └── sample_problems/      # Pre-generated test problems
│
└── results/
    └── benchmark_results.csv # Saved benchmark data
```

## 🧪 Extending the Framework

### Add New Algorithm

Create a new file `your_algorithm.py`:

```python
from typing import Dict, List
from dataclasses import dataclass

class YourScheduler:
    def solve(self, problem):
        """
        Solve the scheduling problem.
        
        Args:
            problem: SchedulingProblem instance
            
        Returns:
            SchedulingSolution with assignments and metrics
        """
        # Your algorithm implementation
        pass
```

Then add to benchmarking in `benchmark.py`.

### Add New Constraints

Modify the `is_feasible()` method in schedulers to check additional constraints.

### Add New Objectives

Extend the `calculate_cost()` method to include additional objective terms.

## 📚 References

This implementation is inspired by classical workforce scheduling literature:

1. Ernst, A. T., et al. (2004). "Staff scheduling and rostering: A review of applications, methods and models." *European Journal of Operational Research*.

2. Brucker, P., et al. (2011). "Personnel scheduling: Models and complexity." *European Journal of Operational Research*.

3. Burke, E. K., et al. (2004). "The state of the art of nurse rostering." *Journal of Scheduling*.

## 🎓 Research Context

This repository was developed as part of PhD application materials demonstrating:
- Understanding of operations research fundamentals
- Ability to implement optimization algorithms
- Scientific approach to algorithm comparison
- Clean, documented code suitable for research

For questions about methodology or extensions, contact: nanadompreh@hotmail.com

## 📄 License

MIT License - Feel free to use for research or educational purposes.

## 🙏 Acknowledgments

Built by Ernest Owusu as part of PhD application portfolio for Fall 2026 programs in Industrial Engineering and Operations Research.

---

*Note: This is a simplified academic implementation for benchmarking purposes. For production workforce scheduling systems with advanced features, see my commercial work (repositories private).*
