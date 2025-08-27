import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from ..main import MatchData, Market, Outcome
except ImportError:
    from main import MatchData, Market, Outcome


class FeatureEngineer:
    """Extract and engineer features from betting market data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_market_features(self, match_data: MatchData) -> Dict[str, float]:
        """Extract features from betting market data"""
        features = {}
        
        # H2H Market Features
        h2h_market = self._get_market(match_data.markets, "h2h")
        if h2h_market:
            h2h_features = self._extract_h2h_features(h2h_market, match_data)
            features.update(h2h_features)
        
        # Totals Market Features
        totals_market = self._get_market(match_data.markets, "totals")
        if totals_market:
            totals_features = self._extract_totals_features(totals_market)
            features.update(totals_features)
        
        # Spreads Market Features
        spreads_market = self._get_market(match_data.markets, "spreads")
        if spreads_market:
            spreads_features = self._extract_spreads_features(spreads_market, match_data)
            features.update(spreads_features)
        
        return features
    
    def _get_market(self, markets: List[Market], key: str) -> Optional[Market]:
        """Get market by key"""
        return next((m for m in markets if m.key == key), None)
    
    def _extract_h2h_features(self, market: Market, match_data: MatchData) -> Dict[str, float]:
        """Extract features from head-to-head market"""
        features = {}
        
        # Get odds for each outcome
        home_odds = None
        away_odds = None
        draw_odds = None
        
        for outcome in market.outcomes:
            outcome_name = outcome.name.lower()
            if outcome_name == "home" or outcome.name == match_data.home_team:
                home_odds = outcome.price
            elif outcome_name == "away" or outcome.name == match_data.away_team:
                away_odds = outcome.price
            elif outcome_name == "draw":
                draw_odds = outcome.price
        
        if home_odds and away_odds and draw_odds:
            # Implied probabilities
            total_prob = (1/home_odds + 1/away_odds + 1/draw_odds)
            features['home_prob'] = (1/home_odds) / total_prob
            features['away_prob'] = (1/away_odds) / total_prob  
            features['draw_prob'] = (1/draw_odds) / total_prob
            
            # Market efficiency (overround)
            features['overround'] = total_prob - 1.0
            
            # Home advantage
            features['home_advantage'] = features['home_prob'] - features['away_prob']
            
            # Favorite indicator
            features['is_home_favorite'] = 1.0 if home_odds < away_odds else 0.0
            features['favorite_odds'] = min(home_odds, away_odds)
            features['underdog_odds'] = max(home_odds, away_odds)
            
            # Odds ratios
            features['home_away_odds_ratio'] = home_odds / away_odds
            features['draw_vs_winner_ratio'] = draw_odds / min(home_odds, away_odds)
        
        return features
    
    def _extract_totals_features(self, market: Market) -> Dict[str, float]:
        """Extract features from totals (over/under) market"""
        features = {}
        
        over_odds = None
        under_odds = None
        total_line = None
        
        for outcome in market.outcomes:
            if outcome.name.lower() == "over":
                over_odds = outcome.price
                total_line = outcome.point if outcome.point else 2.5
            elif outcome.name.lower() == "under":
                under_odds = outcome.price
        
        if over_odds and under_odds and total_line:
            # Implied probabilities
            total_prob = (1/over_odds + 1/under_odds)
            features['over_prob'] = (1/over_odds) / total_prob
            features['under_prob'] = (1/under_odds) / total_prob
            features['total_line'] = total_line
            
            # Expected goals based on totals market
            features['expected_goals'] = total_line * features['over_prob'] + \
                                       (total_line * 0.8) * features['under_prob']
            
            # High/low scoring game indicator
            features['high_scoring_expected'] = 1.0 if features['over_prob'] > 0.5 else 0.0
        
        return features
    
    def _extract_spreads_features(self, market: Market, match_data: MatchData) -> Dict[str, float]:
        """Extract features from spreads (handicap) market"""
        features = {}
        
        handicap_line = None
        home_spread_odds = None
        away_spread_odds = None
        
        for outcome in market.outcomes:
            if outcome.point is not None:
                if handicap_line is None:
                    handicap_line = abs(outcome.point)
                
                outcome_name = outcome.name.lower()
                if outcome_name == "home" or outcome.name == match_data.home_team:
                    home_spread_odds = outcome.price
                elif outcome_name == "away" or outcome.name == match_data.away_team:
                    away_spread_odds = outcome.price
        
        if handicap_line is not None and home_spread_odds and away_spread_odds:
            features['handicap_line'] = handicap_line
            features['home_spread_odds'] = home_spread_odds
            features['away_spread_odds'] = away_spread_odds
            
            # Implied probabilities
            total_prob = (1/home_spread_odds + 1/away_spread_odds)
            features['home_spread_prob'] = (1/home_spread_odds) / total_prob
            features['away_spread_prob'] = (1/away_spread_odds) / total_prob
            
            # Strength differential
            features['strength_differential'] = handicap_line * (features['home_spread_prob'] - 0.5)
        
        return features
    
    def create_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        vector = np.zeros(len(self.feature_names))
        for i, feature_name in enumerate(self.feature_names):
            vector[i] = features.get(feature_name, 0.0)
        
        return vector
    
    def load_historical_data(self, data_path: str = "src/data/") -> pd.DataFrame:
        """Load and combine historical Bundesliga data"""
        import os
        import glob
        
        csv_files = glob.glob(os.path.join(data_path, "D1*.csv"))
        
        # Required columns for consistent training data
        required_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        optional_columns = ['AvgAHH', 'AvgAHA', 'AHh']  # Asian Handicap columns
        
        dfs = []
        for file in csv_files:
            try:
                # Try UTF-8 first, handle parsing errors
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                try:
                    # Fall back to latin-1
                    df = pd.read_csv(file, encoding='latin-1', on_bad_lines='skip')
                except Exception as e:
                    print(f"Failed to load {os.path.basename(file)}: {e}")
                    continue
            except Exception as e:
                print(f"Failed to parse {os.path.basename(file)}: {e}")
                continue
            
            # Check if required columns exist
            if all(col in df.columns for col in required_columns):
                # Select only the columns we need
                columns_to_keep = required_columns.copy()
                
                # Add optional columns if they exist
                for col in optional_columns:
                    if col in df.columns:
                        columns_to_keep.append(col)
                
                df_filtered = df[columns_to_keep].copy()
                dfs.append(df_filtered)
                print(f"Loaded {len(df_filtered)} rows from {os.path.basename(file)}")
            else:
                missing_cols = [col for col in required_columns if col not in df.columns]
                print(f"Skipping {os.path.basename(file)} - missing columns: {missing_cols}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            return self._clean_historical_data(combined_df)
        
        return pd.DataFrame()
    
    def _clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare historical data"""
        # Remove rows with missing essential data
        essential_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        
        # Convert numeric columns to proper types
        numeric_cols = ['FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle optional Asian Handicap columns
        ah_cols = ['AvgAHH', 'AvgAHA', 'AHh']
        for col in ah_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing essential data
        df = df.dropna(subset=essential_cols)
        
        # Filter out invalid odds (less than 1.0)
        odds_cols = ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        for col in odds_cols:
            df = df[df[col] >= 1.0]
        
        print(f"Cleaned data: {len(df)} valid matches")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical DataFrame"""
        # Define consistent feature set based on available columns
        feature_names = [
            'home_prob', 'away_prob', 'draw_prob', 'overround', 'home_advantage',
            'is_home_favorite', 'home_away_odds_ratio', 'over_prob', 'under_prob',
            'total_line', 'expected_goals', 'high_scoring_expected'
        ]
        
        # Add Asian Handicap features if available
        if 'AvgAHH' in df.columns and 'AvgAHA' in df.columns and 'AHh' in df.columns:
            feature_names.extend(['handicap_line', 'home_spread_prob', 'strength_differential'])
        
        self.feature_names = feature_names
        
        X = []
        y = []
        
        for _, row in df.iterrows():
            features = self._extract_historical_features_consistent(row)
            if features is not None:
                # Create feature vector in consistent order
                feature_vector = [features.get(name, 0.0) for name in self.feature_names]
                X.append(feature_vector)
                y.append([int(row['FTHG']), int(row['FTAG'])])
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        return np.array(X), np.array(y)
    
    def _extract_historical_features(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract features from historical data row"""
        features = {}
        
        try:
            # Use Bet365 odds as primary source
            if all(col in row and pd.notna(row[col]) for col in ['B365H', 'B365D', 'B365A']):
                home_odds = float(row['B365H'])
                draw_odds = float(row['B365D'])
                away_odds = float(row['B365A'])
                
                # Implied probabilities
                total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
                features['home_prob'] = (1/home_odds) / total_prob
                features['away_prob'] = (1/away_odds) / total_prob
                features['draw_prob'] = (1/draw_odds) / total_prob
                features['overround'] = total_prob - 1.0
                features['home_advantage'] = features['home_prob'] - features['away_prob']
                features['is_home_favorite'] = 1.0 if home_odds < away_odds else 0.0
                features['home_away_odds_ratio'] = home_odds / away_odds
            
            # Totals market features
            if all(col in row and pd.notna(row[col]) for col in ['B365>2.5', 'B365<2.5']):
                over_odds = float(row['B365>2.5'])
                under_odds = float(row['B365<2.5'])
                
                total_prob = (1/over_odds + 1/under_odds)
                features['over_prob'] = (1/over_odds) / total_prob
                features['under_prob'] = (1/under_odds) / total_prob
                features['total_line'] = 2.5
                features['expected_goals'] = 2.5 * features['over_prob'] + 2.0 * features['under_prob']
            
            # Asian handicap features
            if all(col in row and pd.notna(row[col]) for col in ['B365AHH', 'B365AHA', 'AHh']):
                handicap_line = abs(float(row['AHh']))
                home_spread_odds = float(row['B365AHH'])
                away_spread_odds = float(row['B365AHA'])
                
                features['handicap_line'] = handicap_line
                total_prob = (1/home_spread_odds + 1/away_spread_odds)
                features['home_spread_prob'] = (1/home_spread_odds) / total_prob
                
            return features if features else None
            
        except (ValueError, KeyError):
            return None
    
    def _extract_historical_features_consistent(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract features from historical data row using consistent columns"""
        features = {}
        
        try:
            # H2H features using market averages
            home_odds = float(row['AvgH'])
            draw_odds = float(row['AvgD'])
            away_odds = float(row['AvgA'])
            
            # Implied probabilities
            total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
            features['home_prob'] = (1/home_odds) / total_prob
            features['away_prob'] = (1/away_odds) / total_prob
            features['draw_prob'] = (1/draw_odds) / total_prob
            features['overround'] = total_prob - 1.0
            features['home_advantage'] = features['home_prob'] - features['away_prob']
            features['is_home_favorite'] = 1.0 if home_odds < away_odds else 0.0
            features['home_away_odds_ratio'] = home_odds / away_odds
            
            # Totals features using market averages
            over_odds = float(row['Avg>2.5'])
            under_odds = float(row['Avg<2.5'])
            
            total_prob_totals = (1/over_odds + 1/under_odds)
            features['over_prob'] = (1/over_odds) / total_prob_totals
            features['under_prob'] = (1/under_odds) / total_prob_totals
            features['total_line'] = 2.5
            features['expected_goals'] = 2.5 * features['over_prob'] + 2.0 * features['under_prob']
            features['high_scoring_expected'] = 1.0 if features['over_prob'] > 0.5 else 0.0
            
            # Asian handicap features (if available)
            if all(col in row.index and pd.notna(row[col]) for col in ['AvgAHH', 'AvgAHA', 'AHh']):
                handicap_line = abs(float(row['AHh']))
                home_spread_odds = float(row['AvgAHH'])
                away_spread_odds = float(row['AvgAHA'])
                
                features['handicap_line'] = handicap_line
                total_prob_ah = (1/home_spread_odds + 1/away_spread_odds)
                features['home_spread_prob'] = (1/home_spread_odds) / total_prob_ah
                features['strength_differential'] = handicap_line * (features['home_spread_prob'] - 0.5)
            
            return features
            
        except (ValueError, KeyError, ZeroDivisionError) as e:
            print(f"Error extracting features: {e}")
            return None