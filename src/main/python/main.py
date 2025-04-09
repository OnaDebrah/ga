import logging

import numpy as np
import time

from mpi4py import MPI

from utils.logger import logger
from optimiser.parallel import ParallelPortfolioOptimizer
from utils.benchmark import benchmark_portfolio_optimization
from utils.market_data import download_price_data, compute_portfolio_statistics
from utils.tickers import tickers


def main():
    """Main entry point."""
    # Parse command line arguments (if any)
    import argparse
    parser = argparse.ArgumentParser(description='Portfolio Optimization Benchmark')
    parser.add_argument('--assets', type=int, default=100, help='Number of assets')
    parser.add_argument('--generations', type=int, default=100, help='Generations for GA')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # For MPI, use command-line arguments
    try:
        args = parser.parse_args()
    except:
        # Default values if running in non-CLI environment
        class Args:
            assets = 100
            generations = 100
            verbose = True
        args = Args()

    # Set logging level based on verbosity
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)

    # Run benchmark
    results = benchmark_portfolio_optimization(
        n_assets=args.assets,
        generations=args.generations,
        print_results=args.verbose
    )

    # If running with MPI, use parallel optimizer
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        logger.info("Starting parallel optimization...")

    # Generate same test data on all processes
    returns, risks, corr, C = generate_test_data(n_assets=args.assets)

    # Run parallel optimization
    optimizer = ParallelPortfolioOptimizer(returns, risks, corr, C)
    start = time.time()
    best_solution, best_fitness = optimizer.run(generations=args.generations)
    parallel_time = time.time() - start

    # Master process reports results
    if rank == 0:
        # Calculate risk
        parallel_risk = 0
        if best_solution is not None and np.sum(best_solution) > 0:
            selected_risks = risks * best_solution
            parallel_risk = np.sqrt(selected_risks @ corr @ selected_risks)

        logger.info(f"Parallel Optimization: Value={best_fitness:.4f}, Risk={parallel_risk:.4f}, Time={parallel_time:.2f}s")

        # Compare with serial
        if 'serial_ga' in results:
            serial_time = results['serial_ga']['time']
            logger.info(f"Parallel Speedup: {serial_time/parallel_time:.1f}x")

if __name__ == "__main__":
    main()