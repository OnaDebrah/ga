from typing import Tuple

import numpy as np
from mpi4py import MPI
from numba import cuda

from cuda_kernels.fitness import evaluate_fitness_gpu
from utils.logger import logger
from .cpu_ga_optimiser import PortfolioGA_CPU
from .gpu_ga_optimiser import PortfolioGA_GPU


class ParallelPortfolioOptimizer:
    """MPI-based parallel portfolio optimizer."""

    def __init__(self,
                 returns: np.ndarray,
                 risks: np.ndarray,
                 corr_matrix: np.ndarray,
                 risk_capacity: float):
        """Initialize parallel optimizer."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize data on all processes
        self.returns = returns
        self.risks = risks
        self.corr = corr_matrix
        self.C = risk_capacity

        # Each process handles different population subset
        self.pop_per_process = 500 // self.size

        # Initialize optimizer based on GPU availability
        try:
            cuda.current_context()
            self.optimizer = PortfolioGA_GPU(
                returns, risks, corr_matrix, risk_capacity,
                pop_size=self.pop_per_process
            )
            self.using_gpu = True
        except:
            self.optimizer = PortfolioGA_CPU(
                returns, risks, corr_matrix, risk_capacity,
                pop_size=self.pop_per_process
            )
            self.using_gpu = False

    def run(self, generations: int = 100) -> Tuple[np.ndarray, float]:
        """Run parallel optimization."""
        best_solution = None
        best_fitness = 0.0

        for gen in range(generations):
            # Run one generation
            if self.using_gpu:
             self.optimizer.evaluate_fitness_cpu()

            else:
                self.optimizer.evaluate_fitness_cpu()

            self.optimizer._selection_and_reproduction()

            # Collect best solutions from all processes
            local_best = (self.optimizer.best_fitness, self.optimizer.best_solution)
            all_results = self.comm.gather(local_best, root=0)

            # Master process selects best overall solution
            if self.rank == 0:
                for fitness, solution in all_results:
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = solution

                # Broadcast the best solution to all processes
                best_data = (best_fitness, best_solution)
            else:
                best_data = None

            # Share the best solution with all processes
            best_data = self.comm.bcast(best_data, root=0)
            best_fitness, global_best = best_data

            # Migration strategy: Replace the worst solutions with global best
            if gen % 10 == 0 and global_best is not None:
                worst_idx = np.argmin(self.optimizer.fitness)
                self.optimizer.population[worst_idx] = global_best.copy()

            # Progress reporting
            if self.rank == 0 and gen % 10 == 0:
                logger.info(f"Generation {gen}: Best fitness = {best_fitness:.4f}")

        return best_solution, best_fitness
