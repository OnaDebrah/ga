import time

import numpy as np

from utils.logger import logger
from optimiser.dp_solver import KnapsackSolver
from optimiser.cpu_ga_optimiser import PortfolioGA_CPU
from optimiser.gpu_ga_optimiser import PortfolioGA_GPU
from utils.data_generator import generate_test_data

# def benchmark_portfolio_optimization(n_assets=100, generations=100, print_results=True):
#     """
#     Benchmark different portfolio optimization approaches.
#
#     Args:
#         n_assets: Number of assets
#         generations: Number of generations for GA
#         print_results: Whether to print results
#
#     Returns:
#         Dictionary with benchmark results
#     """
#     results = {}
#
#     # Generate test data
#     returns, risks, corr, C = generate_test_data(n_assets=n_assets)
#
#     if print_results:
#         logger.info(f"Running portfolio optimization for {n_assets} assets")
#         logger.info(f"Risk capacity: {C:.2f}, Max possible return: {np.sum(returns):.2f}")
#
#     # Benchmark DP solution (only for smaller problems)
#     if n_assets <= 30:
#         start = time.time()
#         dp_result, dp_solution = KnapsackSolver.solve(returns, risks, C)
#         dp_time = time.time() - start
#
#         dp_risk = 0
#         if np.sum(dp_solution) > 0:
#             selected_risks = risks * dp_solution
#             dp_risk = np.sqrt(selected_risks @ corr @ selected_risks)
#
#         results['dp'] = {
#             'value': float(dp_result),
#             'risk': float(dp_risk),
#             'time': dp_time,
#             'solution': dp_solution
#         }
#
#         if print_results:
#             logger.info(f"DP Solution: Value={dp_result:.4f}, Risk={dp_risk:.4f}, Time={dp_time:.2f}s")
#
#     # Serial GA
#     start = time.time()
#     ga_serial = PortfolioGA(returns, risks, corr, C, pop_size=200)
#     for _ in range(generations):
#         ga_serial.run_generation()
#     serial_time = time.time() - start
#
#     serial_value = ga_serial.best_fitness
#     serial_solution = ga_serial.best_solution
#     serial_risk = 0
#     if np.sum(serial_solution) > 0:
#         selected_risks = risks * serial_solution
#         serial_risk = np.sqrt(selected_risks @ corr @ selected_risks)
#
#     results['serial_ga'] = {
#         'value': float(serial_value),
#         'risk': float(serial_risk),
#         'time': serial_time,
#         'solution': serial_solution
#     }
#
#     if print_results:
#         logger.info(f"Serial GA: Value={serial_value:.4f}, Risk={serial_risk:.4f}, Time={serial_time:.2f}s")
#
#     # GPU GA if available
#     try:
#         start = time.time()
#         ga_gpu = PortfolioGA_GPU(returns, risks, corr, C, pop_size=200)
#
#         if ga_gpu.gpu_available:
#             for _ in range(generations):
#                 ga_gpu.evaluate_fitness()
#                 ga_gpu._selection_and_reproduction()
#
#             gpu_time = time.time() - start
#
#             gpu_value = ga_gpu.best_fitness
#             gpu_solution = ga_gpu.best_solution
#             gpu_risk = 0
#             if np.sum(gpu_solution) > 0:
#                 selected_risks = risks * gpu_solution
#                 gpu_risk = np.sqrt(selected_risks @ corr @ selected_risks)
#
#             results['gpu_ga'] = {
#                 'value': float(gpu_value),
#                 'risk': float(gpu_risk),
#                 'time': gpu_time,
#                 'solution': gpu_solution
#             }
#
#             if print_results:
#                 logger.info(f"GPU GA: Value={gpu_value:.4f}, Risk={gpu_risk:.4f}, Time={gpu_time:.2f}s")
#                 logger.info(f"Speedup: {serial_time/gpu_time:.1f}x")
#         else:
#             if print_results:
#                 logger.warning("GPU not available for benchmarking")
#     except Exception as e:
#         if print_results:
#             logger.error(f"GPU evaluation failed: {e}")
#
#     return results

def benchmark_portfolio_optimization(n_assets=100, generations=100, print_results=True):
    """
    Benchmark different portfolio optimization approaches.

    Args:
        n_assets: Number of assets
        generations: Number of generations for GA
        print_results: Whether to print results

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # Generate test data
    returns, risks, corr, C = generate_test_data(n_assets=n_assets)

    if print_results:
        logger.info(f"Running portfolio optimization for {n_assets} assets")
        logger.info(f"Risk capacity: {C:.2f}, Max possible return: {np.sum(returns):.2f}")

    # Benchmark DP solution (only for smaller problems)
    if n_assets <= 30:
        start = time.time()
        dp_result, dp_solution = KnapsackSolver.solve(returns, risks, C)
        dp_time = time.time() - start

        dp_risk = 0
        if np.sum(dp_solution) > 0:
            selected_risks = risks * dp_solution
            dp_risk = np.sqrt(selected_risks @ corr @ selected_risks)

        results['dp'] = {
            'value': float(dp_result),
            'risk': float(dp_risk),
            'time': dp_time,
            'solution': dp_solution
        }

        if print_results:
            logger.info(f"DP Solution: Value={dp_result:.4f}, Risk={dp_risk:.4f}, Time={dp_time:.2f}s")

    # Serial GA
    start = time.time()
    ga_serial = PortfolioGA_CPU(returns, risks, corr, C, pop_size=200)
    for _ in range(generations):
        ga_serial.run_generation()
    serial_time = time.time() - start

    serial_value = ga_serial.best_fitness
    serial_solution = ga_serial.best_solution
    serial_risk = 0
    if np.sum(serial_solution) > 0:
        selected_risks = risks * serial_solution
        serial_risk = np.sqrt(selected_risks @ corr @ selected_risks)

    results['serial_ga'] = {
        'value': float(serial_value),
        'risk': float(serial_risk),
        'time': serial_time,
        'solution': serial_solution
    }

    if print_results:
        logger.info(f"Serial GA: Value={serial_value:.4f}, Risk={serial_risk:.4f}, Time={serial_time:.2f}s")

    # GPU GA if available
    try:
        start = time.time()
        ga_gpu = PortfolioGA_GPU(returns, risks, corr, C, pop_size=200)

        if ga_gpu.gpu_available:
            for _ in range(generations):
                ga_gpu.evaluate_fitness()
                ga_gpu._selection_and_reproduction()

            gpu_time = time.time() - start

            gpu_value = ga_gpu.best_fitness
            gpu_solution = ga_gpu.best_solution
            gpu_risk = 0
            if np.sum(gpu_solution) > 0:
                selected_risks = risks * gpu_solution
                gpu_risk = np.sqrt(selected_risks @ corr @ selected_risks)

            results['gpu_ga'] = {
                'value': float(gpu_value),
                'risk': float(gpu_risk),
                'time': gpu_time,
                'solution': gpu_solution
            }

            if print_results:
                logger.info(f"GPU GA: Value={gpu_value:.4f}, Risk={gpu_risk:.4f}, Time={gpu_time:.2f}s")
                logger.info(f"Speedup: {serial_time/gpu_time:.1f}x")
        else:
            if print_results:
                logger.warning("GPU not available for benchmarking")
    except Exception as e:
        if print_results:
            logger.error(f"GPU evaluation failed: {e}")

    return results