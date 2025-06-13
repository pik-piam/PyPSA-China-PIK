==== UNDER CONSTRUCTION ==== 

# Installation and setup
Please contact us if needed. Note that pypsa-China-PIK is currently under active development and we recommend waiting until the alpha or first stable release.


## Quick start

- Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager (miniconda is lightest option). You can check whether you already have conda with `which anaconda` or `which conda `. On windows use "where conda"
- Setup the environment (on unix `source activate pypsa-china`)
- Activate it
- [Run locally](#local_exec)

# Running the workflow 
PyPSA-China execution is controlled by the [snakemake](https://snakemake.readthedocs.io/en/stable/) workflow manager. The `snakefile` should be understood as a control file. The control file can be edited if you need new features, however all implemented functionalities are accessible via the config and CLI args. The workflow is intended to be managed via the config files rather than the CLI argumnents. As explained below, the config files allow the control of nearly all aspects of the PyPSA-China execution.  

## dry runs
The `-n` flag from snakemake allows to start a "dry run", which is a mock run that will show what rules would be executed. We recommend always running with this flag before launching an actual run.

```
cd <workflow_root_folder>
snakemake -n <optional_additional_snakemake options>
```

## local execution
<a name="local_exec"></a>
You can execute the workflow locally using
```
cd <workflow_root_folder>
snakemake <optional_additional_snakemake options>
```
PyPSA-China is resource intensive and you will need to decrease the time resolution (see configs)

## profiles & remote/hpc execution

The `--profile` arg allows you to specify a yaml config file that will control the workflow execution. This profile can include snakemake flags, such as re-run conditions and verbosity.

The `--profile` is especially useful for specifying and controlling remote execution, for example on an HPC. You will find a slurm HPC example in the config under `pik_hpc_profile/`. This allows you to set the resources per rule. Note that the profile must be called `config.yaml` 

To execute the workflow on a remote resource, e.g. a slurm cluster you can do the below from a login node:
```
cd <workflow_root_folder>
snakemake --profile=<configs/my_profile_parent_dir> <optional_additional_snakemake_opts>
```

## custom config options

Snakemake overwrites configs in order that they are added. Instead of editing the default config, it is recommended to have your own config file. This config file then has the subset of options you want to overwrite and can be executed using

```
cd <workflow_root_folder>
snakemake --configfile=<configs/my_variations.yml> <optional_additional_snakemake_opts>
```

## running the examples

EXAMPLES ARE CURRENTLY UNAVAILABLE

## snakemake tricks
The first rule is the default target rule. Pseudo rules allow to call a whole workflow (e.g. Plot_all).

- `--allowed_rule=RULENAME` is useful if you only want one rule (e.g. one of the plotting rules)
- `--forcerun=prepare_networks` is useful if you want to regenerate the network
- `-t` allows to specify a target file. Snakemake will then automatically workout the right rules (check with `-n` first). We do recommend configuration over this approach
- it is possible to plot the compute graph. See the `dag` rule in the snakefile

# Configuration options

Under construction


# Development and debugging

It is possible to run each script as standalone using the `mock_snakemake` helper utility. The python file will run the __main__ script, reading the Snakefile.

### specific settings
You can edit the wildcards in mocksnakemake. You can also mock passing a configfile ontop of defaults by adding it to the snakefile (add configfile:"my_config" after the default configs)  


# Questions?
Please contact us if needed. Note that pypsa-China-PIK is currently under active development and we recommend waiting until the alpha or first stable release.