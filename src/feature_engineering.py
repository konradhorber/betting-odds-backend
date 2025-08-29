import numpy as np
import pandas as pd
from typing import Tuple, List
from math import exp
from scipy.stats import norm


class FeatureEngineer:
    """Extract and engineer numeric features for Poisson"""

    def __init__(self, sigma_gd: float = 1.6, min_odds: float = 1.01):
        """
        sigma_gd: stdev used in AH→goal-diff normal approximation (tune via CV)
        min_odds: lower clip for odds to avoid divide-by-zero / nonsense odds
        """
        self.sigma_gd = sigma_gd
        self.min_odds = min_odds

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _p_over_from_mu(
        mu: float
    ) -> float:
        """P(N >= 3) when N ~ Poisson(mu)."""
        return 1 - exp(-mu) * (1 + mu + mu**2 / 2)

    def _invert_mu(
        self,
        p: float,
        lo: float = 0.05,
        hi: float = 6.0,
        iters: int = 32
    ) -> float:
        """Invert P(N>=3 | Poisson(mu)) = p via bisection."""
        a, b = lo, hi
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        for _ in range(iters):
            m = (a + b) / 2
            if self._p_over_from_mu(m) < p:
                a = m
            else:
                b = m
        return (a + b) / 2

    # ---- public API -------------------------------------------------------

    def engineer_X(
        self,
        X_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Args:
          X_data (pd.DataFrame): X with the following columns:
            - Div (str): League Division
            - Date (str): Match Date (dd/mm/yy format)
            - HomeTeam (str): Home Team name
            - AwayTeam (str): Away Team name
            - AvgH (float): Market average home win odds
            - AvgD (float): Market average draw win odds
            - AvgA (float): Market average away win odds
            - AvgOver2_5 (float): Market average over 2.5 goals odds
            - AvgUnder2_5 (float): Market average under 2.5 goals odds
            - AHh (float): Market size of handicap (home team)
            - AvgAHH (float): Market average Asian handicap home team odds
            - AvgAHA (float): Market average Asian handicap away team odds
        Returns:
          X (pd.DataFrame): Engineered X with following columns:
            Market-derived probabilities and margins:
            - pH: Fair probability of home win (from 1X2 odds)
            - pD: Fair probability of draw (from 1X2 odds)
            - pA: Fair probability of away win (from 1X2 odds)
            - logit_HA: Log-odds ratio of home vs away (ln(pH/pA))
            - logit_draw: Log-odds of draw vs non-draw (ln(pD/(pH+pA)))
            - overround_1x2: Bookmaker margin on 1X2 market
            - pOver25: Fair probability of over 2.5 goals
            - pUnder25: Fair probability of under 2.5 goals
            - overround_OU: Bookmaker margin on over/under market
            - ah_margin: Bookmaker margin on Asian handicap market

            Latent parameters and derived lambdas:
            - mu_total: Implied total goals
                (inverted from Poisson tail probability at 2.5)
            - mu_gd: Implied goal difference (from Asian handicap)
            - lambda_home0: Implied home team Poisson rate
            - lambda_away0: Implied away team Poisson rate
            - lambda_diff: Difference between home and away lambda rates

            Simple interactions and raw features:
            - mu_total_x_logit_HA: Interaction between total goals and home/
                away log-odds
            - p_diff: Difference between home and away win probabilities
            - p_non_draw: Combined probability of non-draw outcomes (pH + pA)
            - AHh: Raw Asian handicap line (positive favors home team)

            All features are float32 format for efficient model training.
        """
        X = X_data.copy()

        # Validate required columns exist
        required: List[str] = [
            'AvgH', 'AvgD', 'AvgA',
            'AvgOver2_5', 'AvgUnder2_5',
            'AHh', 'AvgAHH', 'AvgAHA'
        ]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Safety clips for odds (avoid zero / sub-1 values)
        for c in [
            'AvgH', 'AvgD', 'AvgA', 'AvgOver2_5', 'AvgUnder2_5', 'AvgAHH', 'AvgAHA'
        ]:
            X[c] = pd.to_numeric(
                X[c], errors='coerce'
                ).clip(
                    lower=self.min_odds
                    )

        # --- 1X2 fair probs + margins ---
        qH, qD, qA = 1 / X['AvgH'], 1 / X['AvgD'], 1 / X['AvgA']
        s = qH + qD + qA
        X['pH'], X['pD'], X['pA'] = qH / s, qD / s, qA / s
        X['logit_HA'] = np.log(X['pH'] / X['pA'])
        X['logit_draw'] = np.log(X['pD'] / (X['pH'] + X['pA']))
        X['overround_1x2'] = s - 1

        # --- OU(2.5) fair probs + margin ---
        qo, qu = 1 / X['AvgOver2_5'], 1 / X['AvgUnder2_5']
        so = qo + qu
        X['pOver25'], X['pUnder25'] = qo / so, qu / so
        X['overround_OU'] = so - 1

        # --- implied total goals mu_total (invert Poisson tail at 2.5) ---
        X['mu_total'] = X['pOver25'].apply(self._invert_mu)

        # --- AH fair prob + mu_gd ---
        qHh, qAh = 1 / X['AvgAHH'], 1 / X['AvgAHA']
        sh = qHh + qAh
        pAH_home = (qHh / sh).clip(1e-6, 1 - 1e-6)
        X['ah_margin'] = sh - 1
        X['mu_gd'] = X['AHh'] + self.sigma_gd * norm.ppf(pAH_home)

        # --- implied lambdas ---
        X['lambda_home0'] = np.clip((X['mu_total'] + X['mu_gd']) / 2, 0, None)
        X['lambda_away0'] = np.clip((X['mu_total'] - X['mu_gd']) / 2, 0, None)

        # --- simple interactions ---
        X['mu_total_x_logit_HA'] = X['mu_total'] * X['logit_HA']
        X['p_diff'] = X['pH'] - X['pA']
        X['p_non_draw'] = X['pH'] + X['pA']
        X['lambda_diff'] = X['lambda_home0'] - X['lambda_away0']

        # Return numeric features only (drop strings/datetime)
        feature_cols = [
            # market-derived probabilities/margins
            'pH', 'pD', 'pA', 'logit_HA', 'logit_draw', 'overround_1x2',
            'pOver25', 'pUnder25', 'overround_OU', 'ah_margin',
            # latent totals / GD / lambdas
            'mu_total', 'mu_gd', 'lambda_home0', 'lambda_away0', 'lambda_diff',
            # simple interactions + useful raw line
            'mu_total_x_logit_HA', 'p_diff', 'p_non_draw', 'AHh',
        ]
        X_out = X[feature_cols].astype('float32')

        return X_out

    def engineer_full(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
          data (pd.DataFrame): X and Y with the following columns:
            - Div (str): League Division
            - Date (str): Match Date (dd/mm/yy format)
            - HomeTeam (str): Home Team name
            - AwayTeam (str): Away Team name
            - FTHG (int): Full Time Home Team Goals
            - FTAG (int): Full Time Away Team Goals
            - AvgH (float): Market average home win odds
            - AvgD (float): Market average draw win odds
            - AvgA (float): Market average away win odds
            - AvgOver2_5 (float): Market average over 2.5 goals odds
            - AvgUnder2_5 (float): Market average under 2.5 goals odds
            - AHh (float): Market size of handicap (home team)
            - AvgAHH (float): Market average Asian handicap home team odds
            - AvgAHA (float): Market average Asian handicap away team odds
        Returns:
          X_engineered (pd.DataFrame): numeric features for sklearn (see
            engineer_X() docs)
          Y_data (pd.DataFrame): raw integer goals columns:
            - FTHG (int): Full Time Home Team Goals
            - FTAG (int): Full Time Away Team Goals
        """
        Y_data = data[['FTHG', 'FTAG']].astype(int).copy()
        X_data = data.drop(columns=['FTHG', 'FTAG'])
        X_engineered = self.engineer_X(X_data)
        return X_engineered, Y_data
