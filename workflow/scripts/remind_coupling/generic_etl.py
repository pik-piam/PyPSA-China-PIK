""" generic etl development, to be rebalanced with the remind_coupling package"""

from typing import Any
import logging
import pandas as pd
import re
import os.path

from os import PathLike

from rpycpl.utils import read_remind_csv
import rpycpl.utils as coupl_utils
from rpycpl.etl import ETL_REGISTRY, Transformation

logger = logging.getLogger(__name__)


class RemindLoader:
    def __init__(self, remind_dir: PathLike):
        self.remind_dir = remind_dir

    def load_frames_csv(self, frames: dict[str, str])-> dict[str, pd.DataFrame]:
        """ Remind Frames to read
        Args:
            frames (dict): (param: remind_symbol_name) to read
        Returns:
            dict[str, pd.DataFrame]: dictionary (param: dataframe)
        """
        paths = {
            k: os.path.join(self.remind_dir, v + ".csv")
            for k, v in frames.items()
            if v
        }
        return {k: read_remind_csv(v) for k, v in paths.items()}

    def load_frames_gdx(self, frames: dict[str, str], gdx_file: PathLike) -> dict[str, pd.DataFrame]:
        # TODO add the variable renaming in the frame
        p = os.path.join(self.remind_dir, "gdx")
        read_data = {}
        for k, v in frames.items():
            read_data[k] = coupl_utils.read_gdx(p, v)
        raise NotImplementedError("GDX loading not implemented yet")

    def merge_split_frames(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """In case several REMIND parameters are needed, group them by their base name
        Args:
            frames (dict): Dictionary with all dataframes
        Example:
            frames = {eta: 'pm_dataeta', eta_part2: 'pm_eta_conv'}
            merge_split_frames(frames)
            >> {eta: pd.concat([pm_dataeta, pm_eta_conv], axis=0).drop_duplicates()
        """

        def group_parts(keys):
            """Chat gpt regex magic"""
            grouped = {}
            for k in keys:
                base = re.sub(r"_part\d+$", "", k)
                grouped.setdefault(base, []).append(k)
            return grouped

        grouped = group_parts(frames)
        unmerged = {k: v for k, v in grouped.items() if len(v) > 1}
        merged = {
            k: pd.concat([frames[v] for v in multi], axis=0)
            for k, multi in unmerged.items()
        }
        merged = {k: v.drop_duplicates().reset_index(drop=True) for k, v in merged.items()}

        to_drop = [item for sublist in unmerged.values() for item in sublist]
        frames = {k: v for k, v in frames.items() if k not in to_drop}
        frames.update(merged)
        return frames


class ETLRunner:
    """Container class to execute ETL steps."""

    @staticmethod
    def load_frames_csv(remind_dir: PathLike, frames: dict[str, str], filters: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
        """Load and optionally filter frames from CSV files."""
        loader = RemindLoader(remind_dir)
        loaded_frames = loader.load_frames_csv(frames)
        loaded_frames = loader.merge_split_frames(loaded_frames)
        if filters:
            loaded_frames.update({k: loaded_frames[k].query(v) for k, v in filters.items()})
        return loaded_frames

    @staticmethod
    def load_frames_gdx(remind_dir: PathLike, frames: dict[str, str], gdx_file: PathLike) -> dict[str, pd.DataFrame]:
        """Load frames from GDX files."""
        loader = RemindLoader(remind_dir)
        return loader.load_frames_gdx(frames, gdx_file)

    @staticmethod
    def run(step: Transformation, loaded_frames: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Run the ETL step using the provided frames and extra arguments.
        Args:
            step (Transformation): The ETL step to run.
            loaded_frames (dict): Dictionary of loaded frames.
            **kwargs: Additional arguments for the ETL method.
        Returns:
            pd.DataFrame: The result of the ETL step.
        """
        method = step.name if not step.method else step.method
        func = ETL_REGISTRY.get(method)
        if not func:
            raise ValueError(f"ETL method '{method}' not found in registry.")
        if kwargs:
            return func(loaded_frames, **kwargs)
        else:
            return func(loaded_frames)


def _mock_snakemake() -> object:
    """wrapper around mock snakemake"""
    import sys

    # ugly hack to make rel imports work as expected
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow_dir = os.path.dirname(scripts_dir)
    
    sys.path.append(scripts_dir)
    from _helpers import mock_snakemake

    # Detect running out    side of snakemake and mock snakemake for testing

    snakemake = mock_snakemake(
        "transform_remind_data",
        snakefile_path=workflow_dir,
    )
    return snakemake


if __name__ == "__main__":

    if "snakemake" not in globals():
        snakemake = _mock_snakemake()

    params = snakemake.params
    remind_dir = os.path.expanduser(snakemake.input.remind_output_dir)
    region = params.region
    config = params.etl_cfg
    if not config:
        raise ValueError("Aborting: No REMIND data ETL config provided")

    # load anscilliary data
    pypsa_cost_files = [
        os.path.join(snakemake.input.pypsa_costs, f)
        for f in os.listdir(snakemake.input.pypsa_costs)
        if f.endswith(".csv")
    ]
    aux_data = {"pypsa_costs": coupl_utils.read_pypsa_costs(pypsa_cost_files)}
    # Can generalise with a "reader" field and data class if needed later
    for k, path in config["data"].items():
        aux_data[k] = pd.read_csv(path)

    logger.info(f"Loaded auxiliary data files {aux_data.keys()}")

    # transform remind data
    steps = config.get("etl_steps", [])
    outputs = {}
    for step_dict in steps:
        step = Transformation(**step_dict)
        frames = ETLRunner.load_frames_csv(remind_dir, step.frames, step.filters)
        if step.method == "convert_load":
            result = ETLRunner.run(step, frames, region=region)
        elif step.name == "technoeconomic_data":
            result = ETLRunner.run(step, frames,
                mappings=aux_data["tech_mapping"],
                pypsa_costs=aux_data["pypsa_costs"],
            )
            result = {k: v for k, v in result.groupby("year")}
        else:
            result = ETLRunner.run(step, frames)
        outputs[step.name] = result

    # save outputs
    outputs["loads"].to_csv(
        snakemake.output.loads,
        index=False,
    )

    for year, df in outputs["technoeconomic_data"].items():
        df.to_csv(
            os.path.join(snakemake.output.technoeconomic_data, f"costs_{year}.csv"),
            index=False,
        )