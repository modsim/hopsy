# ©2020-​2021 ETH Zurich, Axel Theorell

import os
from pathlib import Path

import pandas as pd


class CSV:
    @staticmethod
    def polytope_to_csv(polytope, dirname):
        Path(dirname).mkdir(parents=True, exist_ok=True)
        name = dirname.rstrip("/").split("/")[-1]
        for attribute in dir(polytope):
            tentative_df = getattr(polytope, attribute)
            if isinstance(tentative_df, pd.DataFrame) or isinstance(
                tentative_df, pd.Series
            ):

                if attribute == "transformation":
                    zero_solution_df = pd.Series(0, index=tentative_df.columns)
                    zero_solution_df.to_csv(
                        os.path.join(dirname, "start_" + name + "_rounded.csv"),
                        header=False,
                        index=False,
                    )
                    tentative_df.to_csv(
                        os.path.join(dirname, "N_" + name + "_rounded.csv"),
                        header=False,
                        index=False,
                    )
                elif attribute == "shift":
                    tentative_df.to_csv(
                        os.path.join(dirname, "p_shift_" + name + "_rounded.csv"),
                        header=False,
                        index=False,
                    )
                    name_series = pd.Series(tentative_df.index)
                    name_series.to_csv(
                        os.path.join(
                            dirname, "reaction_names_" + name + "_rounded.csv"
                        ),
                        header=False,
                        index=False,
                    )
                else:
                    tentative_df.to_csv(
                        os.path.join(dirname, attribute + "_" + name + "_rounded.csv"),
                        header=False,
                        index=False,
                    )
