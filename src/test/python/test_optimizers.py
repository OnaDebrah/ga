
from src.main.python.optimiser.dp_solver import KnapsackSolver
from src.main.python.utils.data_generator import generate_test_data

def test_dp_solver():
    returns, risks, corr, C = generate_test_data(n_assets=10)
    value, solution = KnapsackSolver.solve(returns, risks, C)
    assert value > 0