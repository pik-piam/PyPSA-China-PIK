"""Setup scripts/remind as standalone for dev

- add paths
- mock snakemake
"""

import os
import sys


def setup_paths():
    """
    Add the scripts directory to the Python path to enable direct imports
    from workflow/scripts subdirectories without relative imports. (for debugging)
    Call this at the beginning of any script that needs to import from sibling modules.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


setup_paths()
# needs to be imported after setup_paths
from _helpers import mock_snakemake  # noqa: E402


def _mock_snakemake(rule_name, **kwargs) -> object:
    """Wrapper around mock snakemake for the remind/ subfoldder

    Args:
        rule_name (str): the name of the rule
        **kwargs: additional arguments to pass to the helpers.mock_snakemake function
    Returns:
        object: the mocked snakemake object
    """

    # ugly hack to make rel imports work as expected
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow_dir = os.path.dirname(scripts_dir)

    snakemake = mock_snakemake(
        rule_name,
        snakefile_path=workflow_dir,
        **kwargs,
    )
    return snakemake
