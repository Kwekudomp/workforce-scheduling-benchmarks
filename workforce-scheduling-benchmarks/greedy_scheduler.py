"""
Greedy Heuristic for Workforce Scheduling

A fast, constructive algorithm that builds schedules by iteratively assigning
workers to shifts based on a priority metric.

Author: Ernest Owusu
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import time
from data_generator import Worker, Shift, SchedulingProblem


@dataclass
class Assignment:
    """Represents an assignment of a worker to a shift."""
    worker_id: int
    shift_id: int
    cost: float


@dataclass
class SchedulingSolution:
    """Complete solution with assignments and metrics."""
    assignments: List[Assignment]
    total_cost: float
    coverage_rate: float  # Percentage of shifts adequately covered
    solve_time: float
    feasible: bool = True
    

class GreedyScheduler:
    """
    Greedy heuristic scheduler.
    
    Strategy: For each shift, assign the lowest-cost available worker
    who meets the skill requirements.
    """
    
    def __init__(self, priority_metric: str = 'cost'):
        """
        Initialize scheduler.
        
        Args:
            priority_metric: How to prioritize workers ('cost', 'skills', 'balanced')
        """
        self.priority_metric = priority_metric
        
    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """
        Solve the scheduling problem using greedy heuristic.
        
        Args:
            problem: SchedulingProblem instance
            
        Returns:
            SchedulingSolution with assignments
        """
        start_time = time.time()
        
        # Initialize tracking structures
        assignments = []
        worker_hours = {w.id: 0 for w in problem.workers}
        worker_shifts = {w.id: [] for w in problem.workers}
        
        # Sort shifts by day and time for sequential assignment
        sorted_shifts = sorted(problem.shifts, key=lambda s: (s.day, s.shift_type))
        
        # For each shift, assign workers greedily
        for shift in sorted_shifts:
            shift_assignments = 0
            
            # Find eligible workers for this shift
            eligible_workers = self._get_eligible_workers(
                shift, problem.workers, worker_hours, worker_shifts
            )
            
            # Sort workers by priority metric
            eligible_workers = self._prioritize_workers(
                eligible_workers, shift, worker_hours
            )
            
            # Assign workers until shift is covered
            for worker in eligible_workers:
                if shift_assignments >= shift.required_workers:
                    break
                    
                # Calculate cost for this assignment
                cost = self._calculate_assignment_cost(worker, shift)
                
                # Make assignment
                assignments.append(Assignment(
                    worker_id=worker.id,
                    shift_id=shift.id,
                    cost=cost
                ))
                
                # Update tracking
                worker_hours[worker.id] += shift.duration_hours
                worker_shifts[worker.id].append(shift.id)
                shift_assignments += 1
        
        # Calculate solution metrics
        total_cost = sum(a.cost for a in assignments)
        coverage_rate = self._calculate_coverage(assignments, problem.shifts)
        solve_time = time.time() - start_time
        
        return SchedulingSolution(
            assignments=assignments,
            total_cost=total_cost,
            coverage_rate=coverage_rate,
            solve_time=solve_time,
            feasible=(coverage_rate >= 0.95)  # 95% coverage threshold
        )
    
    def _get_eligible_workers(
        self,
        shift: Shift,
        workers: List[Worker],
        worker_hours: Dict[int, float],
        worker_shifts: Dict[int, List[int]]
    ) -> List[Worker]:
        """
        Find workers eligible for this shift.
        
        Checks:
        - Worker is available for this shift
        - Worker has required skills
        - Worker has not exceeded max hours
        """
        eligible = []
        
        for worker in workers:
            # Check availability
            if (shift.day, shift.shift_type) not in worker.availability:
                continue
            
            # Check skills
            if not shift.required_skills.issubset(worker.skills):
                continue
            
            # Check hours limit
            if worker_hours[worker.id] + shift.duration_hours > worker.max_hours_per_week:
                continue
            
            eligible.append(worker)
        
        return eligible
    
    def _prioritize_workers(
        self,
        workers: List[Worker],
        shift: Shift,
        worker_hours: Dict[int, float]
    ) -> List[Worker]:
        """
        Sort workers by priority for this shift.
        
        Different strategies:
        - 'cost': Lowest cost first
        - 'skills': Most specialized (exact skill match) first  
        - 'balanced': Balance cost and utilization
        """
        if self.priority_metric == 'cost':
            # Sort by hourly cost (lowest first)
            return sorted(workers, key=lambda w: w.hourly_cost)
        
        elif self.priority_metric == 'skills':
            # Prefer workers with exactly the required skills (avoid over-qualification)
            def skill_score(w):
                extra_skills = len(w.skills - shift.required_skills)
                return (extra_skills, w.hourly_cost)
            return sorted(workers, key=skill_score)
        
        elif self.priority_metric == 'balanced':
            # Balance cost and current utilization
            def balanced_score(w):
                cost_factor = w.hourly_cost
                utilization = worker_hours[w.id] / w.max_hours_per_week
                # Prefer workers with lower utilization (spread work evenly)
                return cost_factor * (1 + utilization)
            return sorted(workers, key=balanced_score)
        
        else:
            return workers
    
    def _calculate_assignment_cost(self, worker: Worker, shift: Shift) -> float:
        """Calculate cost of assigning worker to shift."""
        base_cost = worker.hourly_cost * shift.duration_hours
        adjusted_cost = base_cost * shift.base_cost_multiplier
        return adjusted_cost
    
    def _calculate_coverage(
        self,
        assignments: List[Assignment],
        shifts: List[Shift]
    ) -> float:
        """
        Calculate percentage of shifts adequately covered.
        
        A shift is covered if assigned workers >= required workers.
        """
        shift_coverage = {s.id: 0 for s in shifts}
        
        for assignment in assignments:
            shift_coverage[assignment.shift_id] += 1
        
        covered_shifts = 0
        for shift in shifts:
            if shift_coverage[shift.id] >= shift.required_workers:
                covered_shifts += 1
        
        return covered_shifts / len(shifts) if shifts else 0.0


def print_solution_summary(solution: SchedulingSolution, problem: SchedulingProblem):
    """Print a summary of the solution."""
    print("=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print(f"Feasible: {solution.feasible}")
    print(f"Total Cost: ${solution.total_cost:.2f}")
    print(f"Coverage Rate: {solution.coverage_rate * 100:.1f}%")
    print(f"Solve Time: {solution.solve_time:.4f} seconds")
    print(f"Total Assignments: {len(solution.assignments)}")
    
    # Calculate average assignments per worker
    worker_assignments = {}
    for assignment in solution.assignments:
        worker_id = assignment.worker_id
        worker_assignments[worker_id] = worker_assignments.get(worker_id, 0) + 1
    
    if worker_assignments:
        avg_assignments = sum(worker_assignments.values()) / len(problem.workers)
        max_assignments = max(worker_assignments.values())
        min_assignments = min(worker_assignments.values())
        
        print(f"\nWorkload Distribution:")
        print(f"  Average shifts per worker: {avg_assignments:.1f}")
        print(f"  Max shifts (single worker): {max_assignments}")
        print(f"  Min shifts (single worker): {min_assignments}")
    
    print("=" * 60)


if __name__ == "__main__":
    from data_generator import generate_scheduling_problem
    
    # Test on small problem
    print("Testing Greedy Scheduler on Small Problem\n")
    problem = generate_scheduling_problem(
        num_workers=10,
        num_shifts=5,
        num_days=7,
        seed=42
    )
    
    scheduler = GreedyScheduler(priority_metric='cost')
    solution = scheduler.solve(problem)
    print_solution_summary(solution, problem)
    
    # Test different priority metrics
    print("\n\nComparing Priority Metrics\n")
    metrics = ['cost', 'skills', 'balanced']
    
    for metric in metrics:
        print(f"\n{metric.upper()} Priority:")
        scheduler = GreedyScheduler(priority_metric=metric)
        solution = scheduler.solve(problem)
        print(f"  Cost: ${solution.total_cost:.2f}")
        print(f"  Coverage: {solution.coverage_rate * 100:.1f}%")
        print(f"  Time: {solution.solve_time:.4f}s")
