"""
Genetic Algorithm for Workforce Scheduling

An evolutionary metaheuristic that evolves a population of candidate schedules
through selection, crossover, and mutation operations.

Author: Ernest Owusu
"""

import random
import time
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from data_generator import Worker, Shift, SchedulingProblem
from greedy_scheduler import Assignment, SchedulingSolution


class Schedule:
    """
    Represents a candidate schedule (chromosome).
    
    Encoding: For each shift, list of assigned worker IDs
    """
    
    def __init__(self, problem: SchedulingProblem):
        self.problem = problem
        self.assignments = {shift.id: [] for shift in problem.shifts}
        self.fitness = None
        
    def copy(self):
        """Create a deep copy of this schedule."""
        new_schedule = Schedule(self.problem)
        new_schedule.assignments = {
            shift_id: workers.copy() 
            for shift_id, workers in self.assignments.items()
        }
        new_schedule.fitness = self.fitness
        return new_schedule
    
    def to_assignment_list(self) -> List[Assignment]:
        """Convert schedule to list of Assignment objects."""
        assignments = []
        for shift in self.problem.shifts:
            for worker_id in self.assignments[shift.id]:
                worker = self.problem.workers[worker_id]
                cost = worker.hourly_cost * shift.duration_hours * shift.base_cost_multiplier
                assignments.append(Assignment(
                    worker_id=worker_id,
                    shift_id=shift.id,
                    cost=cost
                ))
        return assignments


class GeneticScheduler:
    """
    Genetic Algorithm for workforce scheduling.
    
    Key operations:
    - Selection: Tournament selection
    - Crossover: Two-point crossover
    - Mutation: Random shift reassignment
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 5,
        elitism: int = 2
    ):
        """
        Initialize genetic algorithm.
        
        Args:
            population_size: Number of candidate solutions
            generations: Number of evolution iterations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size for tournament selection
            elitism: Number of best solutions to preserve
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """
        Solve scheduling problem using genetic algorithm.
        
        Args:
            problem: SchedulingProblem instance
            
        Returns:
            SchedulingSolution with best found schedule
        """
        start_time = time.time()
        
        # Initialize population
        population = [self._create_random_schedule(problem) for _ in range(self.population_size)]
        
        # Evaluate initial population
        for schedule in population:
            schedule.fitness = self._evaluate_fitness(schedule)
        
        best_schedule = max(population, key=lambda s: s.fitness)
        best_fitness_history = [best_schedule.fitness]
        
        # Evolution loop
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: Keep best solutions
            population.sort(key=lambda s: s.fitness, reverse=True)
            for i in range(self.elitism):
                new_population.append(population[i].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                # Repair (ensure feasibility)
                child1 = self._repair(child1)
                child2 = self._repair(child2)
                
                # Evaluate fitness
                child1.fitness = self._evaluate_fitness(child1)
                child2.fitness = self._evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
            
            # Track best solution
            generation_best = max(population, key=lambda s: s.fitness)
            if generation_best.fitness > best_schedule.fitness:
                best_schedule = generation_best.copy()
            
            best_fitness_history.append(best_schedule.fitness)
        
        # Convert best schedule to solution format
        assignments = best_schedule.to_assignment_list()
        total_cost = sum(a.cost for a in assignments)
        
        # Calculate coverage
        shift_coverage = {s.id: 0 for s in problem.shifts}
        for assignment in assignments:
            shift_coverage[assignment.shift_id] += 1
        
        covered_shifts = sum(
            1 for shift in problem.shifts 
            if shift_coverage[shift.id] >= shift.required_workers
        )
        coverage_rate = covered_shifts / len(problem.shifts)
        
        solve_time = time.time() - start_time
        
        return SchedulingSolution(
            assignments=assignments,
            total_cost=total_cost,
            coverage_rate=coverage_rate,
            solve_time=solve_time,
            feasible=(coverage_rate >= 0.95)
        )
    
    def _create_random_schedule(self, problem: SchedulingProblem) -> Schedule:
        """Create a random feasible schedule."""
        schedule = Schedule(problem)
        worker_hours = {w.id: 0 for w in problem.workers}
        
        for shift in problem.shifts:
            # Find eligible workers
            eligible = []
            for worker in problem.workers:
                if (shift.day, shift.shift_type) in worker.availability:
                    if shift.required_skills.issubset(worker.skills):
                        if worker_hours[worker.id] + shift.duration_hours <= worker.max_hours_per_week:
                            eligible.append(worker.id)
            
            # Randomly assign workers
            num_to_assign = min(shift.required_workers, len(eligible))
            if eligible:
                assigned = random.sample(eligible, num_to_assign)
                schedule.assignments[shift.id] = assigned
                
                # Update hours
                for worker_id in assigned:
                    worker_hours[worker_id] += shift.duration_hours
        
        return schedule
    
    def _evaluate_fitness(self, schedule: Schedule) -> float:
        """
        Evaluate fitness of a schedule (higher is better).
        
        Fitness considers:
        - Coverage: Percentage of shifts adequately staffed
        - Cost: Lower cost is better (negated)
        - Balance: Even workload distribution
        """
        problem = schedule.problem
        assignments = schedule.to_assignment_list()
        
        # Coverage component
        shift_coverage = {s.id: 0 for s in problem.shifts}
        for assignment in assignments:
            shift_coverage[assignment.shift_id] += 1
        
        coverage_score = sum(
            min(shift_coverage[s.id] / s.required_workers, 1.0)
            for s in problem.shifts
        ) / len(problem.shifts)
        
        # Cost component (normalized and negated)
        total_cost = sum(a.cost for a in assignments)
        max_possible_cost = sum(
            s.required_workers * max(w.hourly_cost for w in problem.workers) * 
            s.duration_hours * s.base_cost_multiplier
            for s in problem.shifts
        )
        cost_score = 1.0 - (total_cost / max_possible_cost if max_possible_cost > 0 else 0)
        
        # Balance component (variance in worker assignments)
        worker_counts = {w.id: 0 for w in problem.workers}
        for assignment in assignments:
            worker_counts[assignment.worker_id] += 1
        
        if len(assignments) > 0:
            counts = list(worker_counts.values())
            balance_score = 1.0 - (np.std(counts) / (np.mean(counts) + 1e-6))
        else:
            balance_score = 0.0
        
        # Weighted combination
        fitness = (
            0.6 * coverage_score +  # Coverage is most important
            0.3 * cost_score +       # Then cost
            0.1 * balance_score      # Then balance
        )
        
        return fitness
    
    def _tournament_selection(self, population: List[Schedule]) -> Schedule:
        """Select a schedule using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda s: s.fitness)
    
    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
        """
        Perform two-point crossover.
        
        Randomly select two crossover points and swap shift assignments
        between parents.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        shifts = list(parent1.problem.shifts)
        if len(shifts) < 3:
            return child1, child2
        
        # Select two crossover points
        point1, point2 = sorted(random.sample(range(len(shifts)), 2))
        
        # Swap assignments in the middle section
        for i in range(point1, point2):
            shift_id = shifts[i].id
            child1.assignments[shift_id] = parent2.assignments[shift_id].copy()
            child2.assignments[shift_id] = parent1.assignments[shift_id].copy()
        
        return child1, child2
    
    def _mutate(self, schedule: Schedule) -> Schedule:
        """
        Mutate schedule by randomly changing some assignments.
        
        For a random shift, reassign workers.
        """
        mutated = schedule.copy()
        problem = schedule.problem
        
        # Select random shift to mutate
        if not problem.shifts:
            return mutated
        
        shift = random.choice(problem.shifts)
        
        # Find eligible workers for this shift
        worker_hours = self._calculate_worker_hours(mutated)
        eligible = []
        
        for worker in problem.workers:
            if (shift.day, shift.shift_type) in worker.availability:
                if shift.required_skills.issubset(worker.skills):
                    # Allow some flexibility in hours for mutation
                    if worker_hours[worker.id] <= worker.max_hours_per_week:
                        eligible.append(worker.id)
        
        # Reassign with random eligible workers
        if eligible:
            num_to_assign = min(shift.required_workers, len(eligible))
            mutated.assignments[shift.id] = random.sample(eligible, num_to_assign)
        
        return mutated
    
    def _repair(self, schedule: Schedule) -> Schedule:
        """
        Repair schedule to ensure feasibility.
        
        Removes assignments that violate hard constraints.
        """
        repaired = schedule.copy()
        problem = schedule.problem
        worker_hours = {w.id: 0 for w in problem.workers}
        
        for shift in problem.shifts:
            valid_assignments = []
            
            for worker_id in repaired.assignments[shift.id]:
                worker = problem.workers[worker_id]
                
                # Check all constraints
                if (shift.day, shift.shift_type) not in worker.availability:
                    continue
                if not shift.required_skills.issubset(worker.skills):
                    continue
                if worker_hours[worker_id] + shift.duration_hours > worker.max_hours_per_week:
                    continue
                
                valid_assignments.append(worker_id)
                worker_hours[worker_id] += shift.duration_hours
            
            repaired.assignments[shift.id] = valid_assignments
        
        return repaired
    
    def _calculate_worker_hours(self, schedule: Schedule) -> dict:
        """Calculate current hours for each worker."""
        worker_hours = {w.id: 0 for w in schedule.problem.workers}
        
        for shift in schedule.problem.shifts:
            for worker_id in schedule.assignments[shift.id]:
                worker_hours[worker_id] += shift.duration_hours
        
        return worker_hours


if __name__ == "__main__":
    from data_generator import generate_scheduling_problem
    from greedy_scheduler import print_solution_summary
    
    print("Testing Genetic Algorithm Scheduler\n")
    
    # Test on small problem
    problem = generate_scheduling_problem(
        num_workers=15,
        num_shifts=8,
        num_days=7,
        seed=42
    )
    
    print("Running Genetic Algorithm...")
    ga = GeneticScheduler(
        population_size=50,
        generations=30,
        crossover_rate=0.8,
        mutation_rate=0.15
    )
    
    solution = ga.solve(problem)
    print_solution_summary(solution, problem)
