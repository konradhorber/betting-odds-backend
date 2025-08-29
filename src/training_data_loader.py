import pandas as pd
import logging
from pathlib import Path


class TrainingDataLoader():
    def __init__(self, directory: Path):
        self.directory = directory

    def load_data(self, b1_only: bool = False) -> pd.DataFrame:
        """Load training data from CSV files in the specified directory.

        Args:
            b1_only (bool): If True, only load D1_*.csv files (top division).
                          If False, load all *.csv files in the directory.

        Returns:
            pd.DataFrame: Combined training data with the following columns:
                - Div (str): League Division
                - Date (str): Match Date (dd/mm/yy format)
                - HomeTeam (str): Home Team name
                - AwayTeam (str): Away Team name
                - FTHG (int): Full Time Home Team Goals
                - FTAG (int): Full Time Away Team Goals
                - AvgH (float): Market average home win odds
                - AvgD (float): Market average draw win odds
                - AvgA (float): Market average away win odds
                - Avg>2.5 (float): Market average over 2.5 goals odds
                - Avg<2.5 (float): Market average under 2.5 goals odds

            - Files from before 2019-20 season use 'BbAv*' column names which are
              automatically remapped to 'Avg*' for consistency
            - Incomplete rows (with any NaN values) are automatically filtered out
            - Warning logs are generated when incomplete rows are dropped

        Raises:
            Exception: If loading fails due to file access or parsing errors
        """
        try:
            files_to_load = "*.csv"
            if b1_only:
                files_to_load = "D1_*.csv"

            cols_to_keep_from19_20 = [
                "Div",        # League Division
                "Date",       # Match Date (dd/mm/yy)
                "HomeTeam",   # Home Team
                "AwayTeam",   # Away Team
                "FTHG",       # Full Time Home Team Goals
                "FTAG",       # Full Time Away Team Goals
                "AvgH",       # Market average home win odds
                "AvgD",       # Market average draw win odds
                "AvgA",       # Market average away win odds
                "Avg>2.5",    # Market average over 2.5 goals
                "Avg<2.5",    # Market average under 2.5 goals
            ]
            cols_to_keep_before19_20 = [
                "Div",       # League Division
                "Date",      # Match Date (dd/mm/yy)
                "HomeTeam",  # Home Team
                "AwayTeam",  # Away Team
                "FTHG",      # Full Time Home Team Goals
                "FTAG",      # Full Time Away Team Goals
                "BbAvH",     # Betbrain average home win odds
                "BbAvD",     # Betbrain average draw win odds
                "BbAvA",     # Betbrain average away win odds
                "BbAv>2.5",  # Betbrain average over 2.5 goals
                "BbAv<2.5",  # Betbrain average under 2.5 goals
            ]
            remapper_after19_20 = {
                "Avg>2.5": "AvgOver2_5",
                "Avg<2.5": "AvgUnder2_5"
            }
            remapper_before19_20 = {
                "BbAvH": "AvgH",
                "BbAvD": "AvgD",
                "BbAvA": "AvgA",
                "BbAv>2.5": "AvgOver2_5",
                "BbAv<2.5": "AvgUnder2_5",
            }

            dataframes = []

            for file in self.directory.glob(files_to_load):
                interim_df = pd.read_csv(file)
                try:
                    interim_df = interim_df[cols_to_keep_from19_20]
                    interim_df.rename(
                        columns=remapper_after19_20, inplace=True
                    )
                except KeyError:
                    try:
                        interim_df = interim_df[cols_to_keep_before19_20].copy()
                        interim_df.rename(
                            columns=remapper_before19_20, inplace=True
                        )
                    except KeyError as e:
                        print(e)
                        continue

                # Filter out incomplete rows
                rows_before = interim_df.shape[0]
                complete_rows_mask = interim_df.notna().all(axis=1)
                interim_df = interim_df[complete_rows_mask]
                rows_after = interim_df.shape[0]
                rows_dropped = rows_before - rows_after

                if rows_dropped > 0:
                    logging.warning(
                        f"{file} Dropped {rows_dropped} incomplete rows "
                        f"(kept {rows_after}/{rows_before} complete rows)"
                    )

                if not interim_df.empty:
                    dataframes.append(interim_df)

            # Concatenate all dataframes at once
            df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

            return df

        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            raise
