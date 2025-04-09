from typing import Tuple, List

import numpy as np

from .base import PortfolioBase


# class PortfolioGA(PortfolioBase):
#     """Portfolio optimization using genetic algorithm."""
#
#     def __init__(self,
#                  returns: np.ndarray,
#                  risks: np.ndarray,
#                  corr_matrix: np.ndarray,
#                  risk_capacity: float,
#                  pop_size: int = 100,
#                  mutation_rate: float = 0.05,
#                  crossover_rate: float = 0.5,
#                  elitism_rate: float = 0.1):
#         """
#         Initialize genetic algorithm optimizer.
#
#         Args:
#             returns: Expected returns for each asset
#             risks: Risk measures for each asset
#             corr_matrix: Correlation matrix between assets
#             risk_capacity: Maximum acceptable portfolio risk
#             pop_size: Population size
#             mutation_rate: Mutation probability
#             crossover_rate: Crossover probability
#             elitism_rate: Proportion of population preserved as elite
#         """
#         super().__init__(returns, risks, corr_matrix, risk_capacity)
#
#         # GA parameters
#         self.pop_size = max(10, pop_size)
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.elitism_count = max(1, int(elitism_rate * self.pop_size))
#
#         # Initialize population with heuristic bias
#         self.population = self._initialize_population()
#         self.fitness = np.zeros(self.pop_size, dtype=np.float32)
#         self.best_solution = None
#         self.best_fitness = 0.0
#
#     def _initialize_population(self) -> np.ndarray:
#         """Initialize population with heuristic bias toward high Sharpe ratio assets."""
#         population = np.zeros((self.pop_size, self.n), dtype=np.int8)
#
#         # First member is based purely on Sharpe ratios
#         sharpe_ranks = np.argsort(-self.sharpe_weights)
#         for i in range(self.n):
#             if self.calculate_portfolio_risk(population[0]) < self.C:
#                 population[0, sharpe_ranks[i]] = 1
#             else:
#                 population[0, sharpe_ranks[i]] = 0
#                 break
#
#         # Rest of population is random with bias toward high Sharpe assets
#         for i in range(1, self.pop_size):
#             # Biased random initialization
#             prob = self.sharpe_weights / np.sum(self.sharpe_weights)
#             candidates = np.random.choice(self.n, size=min(20, self.n),
#                                           p=prob, replace=False)
#
#             # Add assets until risk capacity is reached
#             risk = 0.0
#             for idx in candidates:
#                 if risk < self.C:
#                     population[i, idx] = 1
#                     risk = self.calculate_portfolio_risk(population[i])
#                     if risk > self.C:
#                         population[i, idx] = 0
#
#         return population
#
#     def evaluate_fitness_cpu(self) -> None:
#         """Evaluate fitness of entire population on CPU."""
#         for i in range(self.pop_size):
#             selected = self.population[i]
#             total_risk = self.calculate_portfolio_risk(selected)
#
#             # Calculate fitness (return if feasible, penalized if not)
#             if total_risk <= self.C:
#                 self.fitness[i] = self.calculate_portfolio_return(selected)
#             else:
#                 # Penalty proportional to constraint violation
#                 self.fitness[i] = 0
#
#         # Update best solution
#         best_idx = np.argmax(self.fitness)
#         if self.fitness[best_idx] > self.best_fitness:
#             self.best_fitness = self.fitness[best_idx]
#             self.best_solution = self.population[best_idx].copy()
#
#     def run_generation(self) -> float:
#         """Run one generation of the genetic algorithm."""
#         self.evaluate_fitness_cpu()
#         self._selection_and_reproduction()
#         return self.best_fitness
#
#     def _selection_and_reproduction(self) -> None:
#         """Perform selection, crossover and mutation to create next generation."""
#         # Sort by fitness
#         sorted_indices = np.argsort(-self.fitness)
#         elite = self.population[sorted_indices[:self.elitism_count]].copy()
#
#         # Tournament selection and reproduction
#         new_population = np.zeros_like(self.population)
#
#         # Copy elite members
#         new_population[:self.elitism_count] = elite
#
#         # Create rest of population through crossover and mutation
#         for i in range(self.elitism_count, self.pop_size, 2):
#             # Tournament selection
#             parent1_idx = self._tournament_selection(3)
#             parent2_idx = self._tournament_selection(3)
#
#             # Crossover
#             if np.random.random() < self.crossover_rate and i+1 < self.pop_size:
#                 child1, child2 = self._crossover(
#                     self.population[parent1_idx],
#                     self.population[parent2_idx]
#                 )
#                 new_population[i] = child1
#                 new_population[i+1] = child2
#             else:
#                 new_population[i] = self.population[parent1_idx].copy()
#                 if i+1 < self.pop_size:
#                     new_population[i+1] = self.population[parent2_idx].copy()
#
#             # Mutation
#             self._mutate(new_population[i])
#             if i+1 < self.pop_size:
#                 self._mutate(new_population[i+1])
#
#         self.population = new_population
#
#     def _tournament_selection(self, tournament_size: int) -> int:
#         """Select individual using tournament selection."""
#         candidates = np.random.choice(
#             self.pop_size, size=min(tournament_size, self.pop_size), replace=False
#         )
#         winner = candidates[np.argmax(self.fitness[candidates])]
#         return winner
#
#     def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Perform crossover between two parents."""
#         # Two-point crossover
#         points = sorted(np.random.choice(self.n - 1, size=2, replace=False) + 1)
#
#         child1 = parent1.copy()
#         child2 = parent2.copy()
#
#         # Swap segments
#         child1[points[0]:points[1]] = parent2[points[0]:points[1]]
#         child2[points[0]:points[1]] = parent1[points[0]:points[1]]
#
#         return child1, child2
#
#     def _mutate(self, individual: np.ndarray) -> None:
#         """Mutate an individual."""
#         for i in range(self.n):
#             if np.random.random() < self.mutation_rate:
#                 individual[i] = 1 - individual[i]  # Flip bit

class PortfolioGA_CPU(PortfolioBase):
    """Portfolio optimization using genetic algorithm."""

    def __init__(self,
                 returns: np.ndarray,
                 risks: np.ndarray,
                 corr_matrix: np.ndarray,
                 risk_capacity: float,
                 tickers: List[str] = None,
                 pop_size: int = 100,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.7,
                 elitism_rate: float = 0.1):
        """
        Initialize genetic algorithm optimizer.

        Args:
            returns: Expected returns for each asset
            risks: Risk measures for each asset
            corr_matrix: Correlation matrix between assets
            risk_capacity: Maximum acceptable portfolio risk
            tickers: List of ticker symbols
            pop_size: Population size
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            elitism_rate: Proportion of population preserved as elite
        """
        super().__init__(returns, risks, corr_matrix, risk_capacity, tickers)

        # GA parameters
        self.pop_size = max(10, pop_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = max(1, int(elitism_rate * self.pop_size))

        # Initialize population with heuristic bias
        self.population = self._initialize_population()
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
        self.best_solution = None
        self.best_fitness = 0.0

    def _initialize_population(self) -> np.ndarray:
        """Initialize population with heuristic bias toward high Sharpe ratio assets."""
        population = np.zeros((self.pop_size, self.n), dtype=np.int8)

        # First member is based purely on Sharpe ratios
        sharpe_ranks = np.argsort(-self.sharpe_weights)
        for i in range(self.n):
            if self.calculate_portfolio_risk(population[0]) < self.C:
                population[0, sharpe_ranks[i]] = 1
            else:
                population[0, sharpe_ranks[i]] = 0
                break

        # Rest of population is random with bias toward high Sharpe assets
        for i in range(1, self.pop_size):
            # Biased random initialization
            prob = self.sharpe_weights / np.sum(self.sharpe_weights)
            candidates = np.random.choice(self.n, size=min(20, self.n),
                                          p=prob, replace=False)

            # Add assets until risk capacity is reached
            risk = 0.0
            for idx in candidates:
                if risk < self.C:
                    population[i, idx] = 1
                    risk = self.calculate_portfolio_risk(population[i])
                    if risk > self.C:
                        population[i, idx] = 0

        return population

    def evaluate_fitness_cpu(self) -> None:
        """Evaluate fitness of entire population on CPU."""
        for i in range(self.pop_size):
            selected = self.population[i]
            total_risk = self.calculate_portfolio_risk(selected)

            # Calculate fitness (return if feasible, penalized if not)
            if total_risk <= self.C:
                self.fitness[i] = self.calculate_portfolio_return(selected)
            else:
                # Penalty proportional to constraint violation
                self.fitness[i] = 0

        # Update best solution
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def run_generation(self) -> float:
        """Run one generation of the genetic algorithm."""
        self.evaluate_fitness_cpu()
        self._selection_and_reproduction()
        return self.best_fitness

    def _selection_and_reproduction(self) -> None:
        """Perform selection, crossover and mutation to create next generation."""
        # Sort by fitness
        sorted_indices = np.argsort(-self.fitness)
        elite = self.population[sorted_indices[:self.elitism_count]].copy()

        # Tournament selection and reproduction
        new_population = np.zeros_like(self.population)

        # Copy elite members
        new_population[:self.elitism_count] = elite

        # Create rest of population through crossover and mutation
        for i in range(self.elitism_count, self.pop_size, 2):
            # Tournament selection
            parent1_idx = self._tournament_selection(3)
            parent2_idx = self._tournament_selection(3)

            # Crossover
            if np.random.random() < self.crossover_rate and i+1 < self.pop_size:
                child1, child2 = self._crossover(
                    self.population[parent1_idx],
                    self.population[parent2_idx]
                )
                new_population[i] = child1
                new_population[i+1] = child2
            else:
                new_population[i] = self.population[parent1_idx].copy()
                if i+1 < self.pop_size:
                    new_population[i+1] = self.population[parent2_idx].copy()

            # Mutation
            self._mutate(new_population[i])
            if i+1 < self.pop_size:
                self._mutate(new_population[i+1])

        self.population = new_population

    def _tournament_selection(self, tournament_size: int) -> int:
        """Select individual using tournament selection."""
        candidates = np.random.choice(
            self.pop_size, size=min(tournament_size, self.pop_size), replace=False
        )
        winner = candidates[np.argmax(self.fitness[candidates])]
        return winner

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        # Two-point crossover
        points = sorted(np.random.choice(self.n - 1, size=2, replace=False) + 1)

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Swap segments
        child1[points[0]:points[1]] = parent2[points[0]:points[1]]
        child2[points[0]:points[1]] = parent1[points[0]:points[1]]

        return child1, child2

    def _mutate(self, individual: np.ndarray) -> None:
        """Mutate an individual."""
        for i in range(self.n):
            if np.random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # Flip bit
