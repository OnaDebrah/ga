import math

import numpy as np
from numba import cuda

from cuda_kernels.fitness import evaluate_fitness_gpu
from utils.logger import logger
from .cpu_ga_optimiser import PortfolioGA_CPU


# class PortfolioGA_GPU(PortfolioGA):
#     """GPU-accelerated portfolio optimization using genetic algorithm."""
#
#     def __init__(self, *args, **kwargs):
#         """Initialize GPU-accelerated genetic algorithm optimizer."""
#         super().__init__(*args, **kwargs)
#         self.gpu_available = self._check_gpu()
#
#         if self.gpu_available:
#             # Initialize GPU arrays
#             self._initialize_gpu_arrays()
#
#     def _check_gpu(self) -> bool:
#         """Check if GPU is available."""
#         try:
#             cuda.current_context()
#             return True
#         except Exception as e:
#             logger.warning(f"CUDA GPU not available. Falling back to CPU. {e}")
#             return False
#
#     def _initialize_gpu_arrays(self) -> None:
#         """Initialize arrays on GPU."""
#         try:
#             self.d_returns = cuda.to_device(self.returns)
#             self.d_risks = cuda.to_device(self.risks)
#             self.d_corr = cuda.to_device(self.corr)
#             self.d_population = cuda.to_device(self.population)
#             self.d_fitness = cuda.device_array(self.pop_size, dtype=np.float32)
#         except Exception as e:
#             logger.error(f"Error initializing GPU arrays: {e}")
#             self.gpu_available = False
#
#     def evaluate_fitness(self) -> None:
#         """Evaluate fitness using GPU if available, otherwise use CPU."""
#         if not self.gpu_available:
#             self.evaluate_fitness_cpu()
#             return
#
#         try:
#             # Copy latest population to GPU
#             self.d_population = cuda.to_device(self.population)
#
#             # Configure grid dimensions
#             threads_per_block = min(256, self.pop_size)
#             blocks_per_grid = (self.pop_size + threads_per_block - 1) // threads_per_block
#
#             # Launch kernel
#             evaluate_fitness_gpu[blocks_per_grid, threads_per_block](
#                 self.d_returns, self.d_risks, self.d_corr,
#                 self.d_population, self.d_fitness, self.C
#             )
#
#             # Copy results back
#             self.fitness = self.d_fitness.copy_to_host()
#
#             # Update best solution
#             best_idx = np.argmax(self.fitness)
#             if self.fitness[best_idx] > self.best_fitness:
#                 self.best_fitness = self.fitness[best_idx]
#                 self.best_solution = self.population[best_idx].copy()
#
#         except Exception as e:
#             logger.error(f"GPU evaluation failed: {e}")
#             logger.info("Falling back to CPU evaluation")
#             self.gpu_available = False
#             self.evaluate_fitness_cpu()

class PortfolioGA_GPU(PortfolioGA_CPU):
    """GPU-accelerated portfolio optimization using genetic algorithm."""

    def __init__(self, *args, **kwargs):
        """Initialize GPU-accelerated genetic algorithm optimizer."""
        super().__init__(*args, **kwargs)
        self.gpu_available = self._check_gpu()

        if self.gpu_available:
            # Initialize GPU arrays
            self._initialize_gpu_arrays()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            cuda.current_context()
            return True
        except Exception as e:
            logger.warning(f"CUDA GPU not available. Falling back to CPU.{e}")
            return False

    def _initialize_gpu_arrays(self) -> None:
        """Initialize arrays on GPU."""
        try:
            self.d_returns = cuda.to_device(self.returns)
            self.d_risks = cuda.to_device(self.risks)
            self.d_corr = cuda.to_device(self.corr)
            self.d_population = cuda.to_device(self.population)
            self.d_fitness = cuda.device_array(self.pop_size, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error initializing GPU arrays: {e}")
            self.gpu_available = False

    def evaluate_fitness(self) -> None:
        """Evaluate fitness using GPU if available, otherwise use CPU."""
        if not self.gpu_available:
            self.evaluate_fitness_cpu()
            return

        try:
            # Copy latest population to GPU
            self.d_population = cuda.to_device(self.population)

            # Configure grid dimensions
            threads_per_block = min(256, self.pop_size)
            blocks_per_grid = (self.pop_size + threads_per_block - 1) // threads_per_block

            # Launch kernel
            evaluate_fitness_gpu[blocks_per_grid, threads_per_block](
                self.d_returns, self.d_risks, self.d_corr,
                self.d_population, self.d_fitness, self.C
            )

            # Copy results back
            self.fitness = self.d_fitness.copy_to_host()

            # Update best solution
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

        except Exception as e:
            logger.error(f"GPU evaluation failed: {e}")
            logger.info("Falling back to CPU evaluation")
            self.gpu_available = False
            self.evaluate_fitness_cpu()
