import math

from numba import cuda

# @cuda.jit
# def evaluate_fitness_gpu(returns, risks, corr, population, fitness, C):
#     tid = cuda.grid(1)
#     n = len(returns)
#
#     if tid < population.shape[0]:
#         total_return = 0.0
#         total_risk = 0.0
#
#         # First pass: calculate return and diagonal risk terms
#         for i in range(n):
#             if population[tid, i] == 1:
#                 total_return += returns[i]
#                 total_risk += (risks[i] ** 2)  # Var = σ²
#
#         # Second pass: add correlation terms (upper triangular only)
#         for i in range(n):
#             if population[tid, i] == 1:
#                 for j in range(i+1, n):
#                     if population[tid, j] == 1:
#                         total_risk += 2 * risks[i] * risks[j] * corr[i,j]
#
#         # Safer risk calculation
#         portfolio_risk = math.sqrt(max(0.0, total_risk))
#
#         # Flexible fitness assignment
#         if portfolio_risk <= C * 1.05:  # 5% tolerance
#             fitness[tid] = total_return
#         else:
#             penalty = max(0, (portfolio_risk - C)/C)  # Proportional penalty
#             fitness[tid] = total_return / (1 + penalty)

# GPU kernel for fitness evaluation
@cuda.jit
def evaluate_fitness_gpu(returns, risks, corr, population, fitness, C):
    """CUDA kernel for parallel fitness evaluation."""
    tid = cuda.grid(1)
    n = len(returns)

    if tid < population.shape[0]:
        total_return = 0.0
        total_risk = 0.0

        # Compute return and weighted risk
        for i in range(n):
            if population[tid, i] == 1:
                total_return += returns[i]

                # Calculate correlated risk
                for j in range(n):
                    if population[tid, j] == 1:
                        total_risk += risks[i] * risks[j] * corr[i, j]

        total_risk = math.sqrt(max(0.0, total_risk))  # Avoid negative values due to numerical issues

        # Assign fitness (return if feasible, 0 if not)
        if total_risk <= C:
            fitness[tid] = total_return
        else:
            fitness[tid] = 0.0