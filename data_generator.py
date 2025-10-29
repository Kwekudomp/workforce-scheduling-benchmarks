"""
Data Generator for Workforce Scheduling Problems

Generates synthetic scheduling problem instances with configurable parameters.
Useful for algorithm testing and benchmarking.

Author: Ernest Owusu
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np


@dataclass
class Worker:
    """Represents a worker with skills, availability, and preferences."""
    id: int
    name: str
    skills: Set[str]
    hourly_cost: float
    availability: List[Tuple[int, int]]  # List of (day, shift) tuples
    max_hours_per_week: int = 40
    

@dataclass
class Shift:
    """Represents a shift with timing and requirements."""
    id: int
    day: int  # 0-6 for Mon-Sun
    shift_type: int  # 0=morning, 1=afternoon, 2=evening
    start_hour: int
    duration_hours: int
    required_workers: int
    required_skills: Set[str]
    base_cost_multiplier: float = 1.0  # e.g., 1.5 for night shifts
    

@dataclass
class SchedulingProblem:
    """Complete problem instance."""
    workers: List[Worker]
    shifts: List[Shift]
    num_days: int
    skill_types: Set[str]
    

def generate_scheduling_problem(
    num_workers: int = 20,
    num_shifts: int = 10,
    num_days: int = 7,
    skill_types: List[str] = None,
    seed: int = None
) -> SchedulingProblem:
    """
    Generate a synthetic workforce scheduling problem.
    
    Args:
        num_workers: Number of workers in the workforce
        num_shifts: Number of shift types per day
        num_days: Number of days in planning horizon
        skill_types: List of required skills (auto-generated if None)
        seed: Random seed for reproducibility
        
    Returns:
        SchedulingProblem instance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Define skill types if not provided
    if skill_types is None:
        skill_types = ['general', 'customer_service', 'technical', 'management']
    
    skill_set = set(skill_types)
    
    # Generate workers
    workers = []
    for i in range(num_workers):
        # Random skill assignment (each worker has 1-3 skills)
        num_skills = random.randint(1, min(3, len(skill_types)))
        worker_skills = set(random.sample(skill_types, num_skills))
        
        # Hourly cost varies by skill level (more skills = higher cost)
        base_cost = 15.0
        skill_premium = len(worker_skills) * 2.0
        hourly_cost = base_cost + skill_premium + random.uniform(-2, 3)
        
        # Generate availability (workers available for 60-100% of shifts)
        availability = []
        for day in range(num_days):
            # Each day has 3 shift types (morning, afternoon, evening)
            available_shifts = random.sample(
                range(3), 
                k=random.randint(1, 3)
            )
            for shift_type in available_shifts:
                availability.append((day, shift_type))
        
        # Max hours per week (between 20-40 hours)
        max_hours = random.choice([20, 25, 30, 35, 40])
        
        workers.append(Worker(
            id=i,
            name=f"Worker_{i}",
            skills=worker_skills,
            hourly_cost=hourly_cost,
            availability=availability,
            max_hours_per_week=max_hours
        ))
    
    # Generate shifts
    shifts = []
    shift_id = 0
    shift_times = {
        0: (8, 8),    # Morning: 8am-4pm
        1: (16, 8),   # Afternoon: 4pm-12am
        2: (0, 8)     # Night: 12am-8am
    }
    
    for day in range(num_days):
        for shift_type in range(3):  # Morning, afternoon, evening
            start_hour, duration = shift_times[shift_type]
            
            # Weekend and night shifts need fewer workers
            if day >= 5:  # Weekend
                required_workers = max(1, num_workers // 8)
            elif shift_type == 2:  # Night shift
                required_workers = max(1, num_workers // 10)
            else:
                required_workers = max(2, num_workers // 6)
            
            # Randomly select 1-2 required skills
            num_required_skills = random.randint(1, 2)
            required_skills = set(random.sample(skill_types, num_required_skills))
            
            # Night and weekend shifts have cost multiplier
            multiplier = 1.0
            if shift_type == 2:  # Night
                multiplier = 1.5
            elif day >= 5:  # Weekend
                multiplier = 1.3
            
            shifts.append(Shift(
                id=shift_id,
                day=day,
                shift_type=shift_type,
                start_hour=start_hour,
                duration_hours=duration,
                required_workers=required_workers,
                required_skills=required_skills,
                base_cost_multiplier=multiplier
            ))
            shift_id += 1
    
    return SchedulingProblem(
        workers=workers,
        shifts=shifts,
        num_days=num_days,
        skill_types=skill_set
    )


def print_problem_summary(problem: SchedulingProblem):
    """Print a summary of the problem instance."""
    print("=" * 60)
    print("SCHEDULING PROBLEM SUMMARY")
    print("=" * 60)
    print(f"Workers: {len(problem.workers)}")
    print(f"Shifts: {len(problem.shifts)}")
    print(f"Days: {problem.num_days}")
    print(f"Skills: {problem.skill_types}")
    print()
    
    # Worker statistics
    avg_cost = np.mean([w.hourly_cost for w in problem.workers])
    avg_skills = np.mean([len(w.skills) for w in problem.workers])
    avg_availability = np.mean([len(w.availability) for w in problem.workers])
    
    print("WORKER STATISTICS:")
    print(f"  Average hourly cost: ${avg_cost:.2f}")
    print(f"  Average skills per worker: {avg_skills:.1f}")
    print(f"  Average availability: {avg_availability:.1f} shifts")
    print()
    
    # Shift statistics
    total_demand = sum(s.required_workers for s in problem.shifts)
    avg_demand = total_demand / len(problem.shifts)
    
    print("SHIFT STATISTICS:")
    print(f"  Total worker-shifts needed: {total_demand}")
    print(f"  Average workers per shift: {avg_demand:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Generating small problem instance...")
    problem = generate_scheduling_problem(
        num_workers=10,
        num_shifts=5,
        num_days=7,
        seed=42
    )
    print_problem_summary(problem)
    
    print("\nGenerating medium problem instance...")
    problem = generate_scheduling_problem(
        num_workers=20,
        num_shifts=10,
        num_days=7,
        seed=42
    )
    print_problem_summary(problem)
    
    print("\nGenerating large problem instance...")
    problem = generate_scheduling_problem(
        num_workers=50,
        num_shifts=20,
        num_days=14,
        seed=42
    )
    print_problem_summary(problem)
