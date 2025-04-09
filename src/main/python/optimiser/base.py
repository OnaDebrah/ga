from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt


# class PortfolioBase:
#     """Base class for portfolio optimization."""
#
#     def __init__(self,
#                  returns: np.ndarray,
#                  risks: np.ndarray,
#                  corr_matrix: np.ndarray,
#                  risk_capacity: float):
#         """
#         Initialize portfolio optimization base class.
#
#         Args:
#             returns: Expected returns for each asset
#             risks: Risk measures for each asset
#             corr_matrix: Correlation matrix between assets
#             risk_capacity: Maximum acceptable portfolio risk
#         """
#         # Validate inputs
#         self._validate_inputs(returns, risks, corr_matrix, risk_capacity)
#
#         # Store portfolio data
#         self.returns = returns.astype(np.float32)  # Use float32 for GPU compatibility
#         self.risks = risks.astype(np.float32)
#         self.corr = corr_matrix.astype(np.float32)
#         self.C = float(risk_capacity)
#         self.n = len(returns)
#
#         # Calculate Sharpe ratio weights for heuristic initialization
#         self.sharpe_weights = self.returns / (self.risks + 1e-6)
#         self.sharpe_weights /= np.max(self.sharpe_weights) + 1e-10
#
#     def _validate_inputs(self, returns, risks, corr_matrix, risk_capacity):
#         """Validate input data."""
#         if len(returns) != len(risks):
#             raise ValueError("Returns and risks arrays must be the same length")
#
#         n = len(returns)
#         if corr_matrix.shape != (n, n):
#             raise ValueError(f"Correlation matrix must be {n}x{n}")
#
#         if not np.allclose(corr_matrix, corr_matrix.T, rtol=1e-5):
#             raise ValueError("Correlation matrix must be symmetric")
#
#         if risk_capacity <= 0:
#             raise ValueError("Risk capacity must be positive")
#
#     def calculate_portfolio_risk(self, selection: np.ndarray) -> float:
#         """
#         Calculate the risk of a portfolio.
#
#         Args:
#             selection: Binary array indicating selected assets
#
#         Returns:
#             Portfolio risk
#         """
#         selected_risks = self.risks * selection
#         return float(np.sqrt(selected_risks @ self.corr @ selected_risks))
#
#     def calculate_portfolio_return(self, selection: np.ndarray) -> float:
#         """
#         Calculate the return of a portfolio.
#
#         Args:
#             selection: Binary array indicating selected assets
#
#         Returns:
#             Portfolio return
#         """
#         return float(np.sum(self.returns * selection))

class PortfolioBase:
    """Base class for portfolio optimization."""

    def __init__(self,
                 returns: np.ndarray,
                 risks: np.ndarray,
                 corr_matrix: np.ndarray,
                 risk_capacity: float,
                 tickers: List[str] = None):
        """
        Initialize portfolio optimization base class.

        Args:
            returns: Expected returns for each asset
            risks: Risk measures for each asset
            corr_matrix: Correlation matrix between assets
            risk_capacity: Maximum acceptable portfolio risk
            tickers: List of ticker symbols
        """
        # Validate inputs
        self._validate_inputs(returns, risks, corr_matrix, risk_capacity)

        # Store portfolio data
        self.returns = returns.astype(np.float32)  # Use float32 for GPU compatibility
        self.risks = risks.astype(np.float32)
        self.corr = corr_matrix.astype(np.float32)
        self.C = float(risk_capacity)
        self.n = len(returns)
        self.tickers = tickers if tickers is not None else [f"Asset_{i}" for i in range(self.n)]

        # Calculate Sharpe ratio weights for heuristic initialization
        self.sharpe_weights = self.returns / (self.risks + 1e-6)
        self.sharpe_weights /= np.max(self.sharpe_weights) + 1e-10

    def _validate_inputs(self, returns, risks, corr_matrix, risk_capacity):
        """Validate input data."""
        if len(returns) != len(risks):
            raise ValueError("Returns and risks arrays must be the same length")

        n = len(returns)
        if corr_matrix.shape != (n, n):
            raise ValueError(f"Correlation matrix must be {n}x{n}")

        if not np.allclose(corr_matrix, corr_matrix.T, rtol=1e-5):
            raise ValueError("Correlation matrix must be symmetric")

        if risk_capacity <= 0:
            raise ValueError("Risk capacity must be positive")

    def calculate_portfolio_risk(self, selection: np.ndarray) -> float:
        """
        Calculate the risk of a portfolio.

        Args:
            selection: Binary array indicating selected assets

        Returns:
            Portfolio risk
        """
        selected_risks = self.risks * selection
        return float(np.sqrt(selected_risks @ self.corr @ selected_risks))

    def calculate_portfolio_return(self, selection: np.ndarray) -> float:
        """
        Calculate the return of a portfolio.

        Args:
            selection: Binary array indicating selected assets

        Returns:
            Portfolio return
        """
        return float(np.sum(self.returns * selection))

    def portfolio_summary(self, selection: np.ndarray) -> Dict:
        """
        Generate a summary of the portfolio.

        Args:
            selection: Binary array indicating selected assets

        Returns:
            Dictionary with portfolio statistics
        """
        selected_indices = np.where(selection == 1)[0]
        selected_tickers = [self.tickers[i] for i in selected_indices]
        selected_returns = self.returns[selected_indices]
        selected_risks = self.risks[selected_indices]

        portfolio_return = self.calculate_portfolio_return(selection)
        portfolio_risk = self.calculate_portfolio_risk(selection)

        # Calculate portfolio Sharpe ratio (assuming risk-free rate of 0.01)
        sharpe_ratio = (portfolio_return - 0.01) / portfolio_risk if portfolio_risk > 0 else 0

        return {
            'tickers': selected_tickers,
            'count': len(selected_tickers),
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'weights': {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers},
            'asset_returns': dict(zip(selected_tickers, selected_returns)),
            'asset_risks': dict(zip(selected_tickers, selected_risks))
        }

    def visualize_portfolio(self, selection: np.ndarray, title: str = "Optimized Portfolio"):
        """
        Visualize the selected portfolio.

        Args:
            selection: Binary array indicating selected assets
            title: Plot title
        """
        # Prepare data for visualization
        summary = self.portfolio_summary(selection)

        # Create scatter plot of all assets
        plt.figure(figsize=(12, 8))

        # Plot all assets
        plt.scatter(self.risks, self.returns, alpha=0.4, color='gray', label='All Assets')

        # Calculate efficient frontier
        frontier_x = np.linspace(0, max(self.risks) * 1.2, 100)
        frontier_y = summary['sharpe_ratio'] * frontier_x
        plt.plot(frontier_x, frontier_y, 'r--', alpha=0.5, label='Efficient Frontier')

        # Plot selected assets
        selected_indices = np.where(selection == 1)[0]
        plt.scatter(self.risks[selected_indices], self.returns[selected_indices],
                    color='blue', s=100, alpha=0.7, label='Selected Assets')

        # Annotate selected assets
        for i in selected_indices:
            plt.annotate(self.tickers[i],
                         (self.risks[i], self.returns[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)

        # Plot portfolio point
        plt.scatter([summary['risk']], [summary['return']], color='green', s=200,
                    marker='*', label='Portfolio')
        plt.annotate(f"Portfolio (SR: {summary['sharpe_ratio']:.2f})",
                     (summary['risk'], summary['return']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=12, fontweight='bold')

        # Set labels and title
        plt.xlabel('Risk (Annual Standard Deviation)')
        plt.ylabel('Return (Annual Expected Return)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Show stats in text box
        stats_text = (
            f"Portfolio Statistics:\n"
            f"Number of Assets: {summary['count']}\n"
            f"Expected Return: {summary['return']:.2%}\n"
            f"Risk: {summary['risk']:.2%}\n"
            f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}"
        )
        plt.figtext(0.15, 0.15, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()
