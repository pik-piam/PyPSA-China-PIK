"""generic etl development, to be rebalanced with the remind_coupling package

The ETL operations are governed by the config file. Allowed fields are defined by the
rpycpl.etl.Transformation class and are
    name: str
    method: Optional[str]
    frames: Dict[str, Any]
    params: Dict[str, Any]
    filters: Dict[str, Any
    kwargs: Dict[str, Any]
    dependencies: Dict[str, Any]

The sequence of operations matters: Dependencies represents previous step outputs.


"""

from typing import Any
import logging
import pandas as pd
import re
import os.path
from os import PathLike

import sys

import setup  # sets up paths
from _helpers import configure_logging

# remind pypsa coupling package
import rpycpl.utils as coupl_utils
from rpycpl.utils import read_remind_csv
from rpycpl.etl import ETL_REGISTRY, Transformation


logger = logging.getLogger(__name__)


class RemindLoader:
    """Load Remind symbol tables from csvs or gdx"""

    def __init__(self, remind_dir: PathLike, backend="csv"):
        self.remind_dir = remind_dir

        supported_backends = ["csv", "gdx"]
        if backend not in supported_backends:
            raise ValueError(f"Backend {backend} not supported. Available: {supported_backends}")
        self.backend = backend

    def _group_split_frames(self, keys, pattern: str = r"_part\d+$") -> dict[str, list[str]]:
        """Chat gpt regex magic to group split frames
        Args:
            keys (list): list of keys
            pattern (str, optional): regex pattern to group split frames by. Defaults to r"_part\\d+$"."
        Returns:
            dict[str, list[str]]: dictionary with base name as key and list of keys as value
        """
        grouped = {}
        for k in keys:
            base = re.sub(pattern, "", k)
            grouped.setdefault(base, []).append(k)
        return grouped

    def load_frames_csv(self, frames: dict[str, str]) -> dict[str, pd.DataFrame]:
        """Remind Frames to read
        Args:
            frames (dict): (param: remind_symbol_name) to read
        Returns:
            dict[str, pd.DataFrame]: dictionary (param: dataframe)
        """
        paths = {k: os.path.join(self.remind_dir, v + ".csv") for k, v in frames.items() if v}
        return {k: read_remind_csv(v) for k, v in paths.items()}

    def load_frames_gdx(
        self, frames: dict[str, str], gdx_file: PathLike
    ) -> dict[str, pd.DataFrame]:

        raise NotImplementedError("GDX loading not implemented yet")

    def merge_split_frames(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """In case several REMIND parameters are needed, group them by their base name
        Args:
            frames (dict): Dictionary with all dataframes
        Example:
            frames = {eta: 'pm_dataeta', eta_part2: 'pm_eta_conv'}
            merge_split_frames(frames)
            >> {eta: pd.concat([pm_dataeta, pm_eta_conv], axis=0).drop_duplicates()}
        """

        grouped = self._group_split_frames(frames)
        unmerged = {k: v for k, v in grouped.items() if len(v) > 1}
        merged = {k: pd.concat([frames[v] for v in multi], axis=0) for k, multi in unmerged.items()}
        merged = {k: v.drop_duplicates().reset_index(drop=True) for k, v in merged.items()}

        to_drop = [item for sublist in unmerged.values() for item in sublist]
        frames = {k: v for k, v in frames.items() if k not in to_drop}
        frames.update(merged)
        return frames

    def auto_load(
        self, frames: dict[str, str], filters: dict[str, str] = None
    ) -> dict[str, pd.DataFrame]:
        """Automatically load, merge, and filter frames in one step.

        Args:
            frames: Dictionary mapping parameter names to REMIND symbol names.
            filters: Optional dictionary of filter expressions to apply to frames.

        Returns:
            Dictionary of processed DataFrames ready for transformation.
        """
        # Load raw frames
        if self.backend == "gdx":
            raw_frames = self.load_frames_gdx(frames, os.path.join(self.remind_dir, "gdx"))
        elif self.backend == "csv":
            raw_frames = self.load_frames_csv(frames)

        # Merge split frames
        processed_frames = self.merge_split_frames(raw_frames)

        # Apply filters if any
        if filters:
            for k, filter_expr in filters.items():
                if k in processed_frames:
                    processed_frames[k] = processed_frames[k].query(filter_expr)
                else:
                    logger.warning(f"Filter specified for non-existent frame: {k}")

        return processed_frames


class ETLRunner:
    """Container class to execute ETL steps."""

    @staticmethod
    def run(
        step: Transformation,
        frames: dict[str, pd.DataFrame],
        previous_outputs: dict[str, Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run the ETL step using the provided frames and extra arguments.
        Args:
            step (Transformation): The ETL step to run.
            frames (dict): Dictionary of loaded frames.
            previous_outputs (dict, optional): Dictionary of outputs from previous
                steps that can be used as inputs.
            **kwargs: Additional arguments for the ETL method.
        Returns:
            pd.DataFrame: The result of the ETL step.
        """
        method = step.name if not step.method else step.method
        func = ETL_REGISTRY.get(method)
        if not func:
            raise ValueError(f"ETL method '{method}' not found in registry.")

        # Handle dependencies on previous outputs if specified in the step
        if hasattr(step, "dependencies") and step.dependencies and previous_outputs:
            for output_key in step.dependencies:
                if output_key in previous_outputs:
                    # Add the dependency to frames with the specified key
                    frames[output_key] = previous_outputs[output_key]
                else:
                    msg = f"Dependency '{output_key}' not found in previous outputs"
                    msg += f" for step '{step.name}'"
                    raise ValueError(msg)

        kwargs.update(step.kwargs)
        if kwargs:
            return func(frames, **kwargs)
        else:
            return func(frames)


if __name__ == "__main__":

    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "transform_remind_data",
            co2_pathway="SSP2-PkBudg1000_CHA_adjcost",
            topology="current+FCG",
            configfiles="resources/tmp/remind_coupled_cg.yaml",
            heating_demand="positive",
        )

    configure_logging(snakemake)
    logger.info("Transforming REMIND data")

    params = snakemake.params
    remind_dir = os.path.expanduser(snakemake.input.remind_output_dir)
    region = params.region
    config = params.etl_cfg
    if not config:
        raise ValueError("Aborting: No REMIND data ETL config provided")

    # load anscilliary data
    logger.info(f"Loading PyPSA costs from {snakemake.input.pypsa_costs}")
    pypsa_cost_files = [
        os.path.join(snakemake.input.pypsa_costs, f)
        for f in os.listdir(snakemake.input.pypsa_costs)
        if f.endswith(".csv")
    ]

    aux_data = {"pypsa_costs": coupl_utils.read_pypsa_costs(pypsa_cost_files)}

    # Can generalise with a "reader" field and data class if needed later
    for k, path in config["data"].items():
        aux_data[k] = pd.read_csv(path)
    # tech_map = coupl_utils.build_tech_map(aux_data["tech_mapping"], map_param="investment")

    logger.info(f"Loaded auxiliary data files {aux_data.keys()}")

    # transform remind data
    steps = config.get("etl_steps", [])
    outputs = {}
    loader = RemindLoader(remind_dir)
    for step_dict in steps:
        step = Transformation(**step_dict)
        logger.info(f"Running ETL step: {step.name} with method {step.method}")
        frames = loader.auto_load(step.frames, step.filters)
        if step.method == "convert_load":
            result = ETLRunner.run(step, frames, region=region, previous_outputs=outputs)
        elif step.name == "technoeconomic_data":
            result = ETLRunner.run(
                step,
                frames,
                mappings=aux_data["tech_mapping"],
                pypsa_costs=aux_data["pypsa_costs"],
                years=snakemake.config["scenario"]["planning_horizons"],
            )
            result = {k: v for k, v in result.groupby("year")}
        elif step.name == "tech_groups":
            result = ETLRunner.run(step, frames=aux_data)
        else:
            result = ETLRunner.run(step, frames, previous_outputs=outputs)
        outputs[step.name] = result

    logger.info(f"ETL steps completed for {list(outputs.keys())}")
    logger.info("saving data")
    # TODO make more generic
    # save outputs

    outp_files = dict(snakemake.output.items())

    outputs["loads"].to_csv(snakemake.output.loads)
    outputs["caps"].to_csv(snakemake.output.remind_caps)
    outputs["tech_groups"].to_csv(snakemake.output.remind_tech_groups)
    for year, df in outputs["technoeconomic_data"].items():
        if f"costs_{year}" in outp_files:
            df.to_csv(
                outp_files[f"costs_{year}"],
                index=False,
            )

    logger.info("REMIND-> pypsa ETL completed")
