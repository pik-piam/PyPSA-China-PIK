
# Running the workflow 

## Snakefile & config
PyPSA-China execution is controlled by the [snakemake](https://snakemake.readthedocs.io/en/stable/) workflow manager. The `snakefile` should be understood as a control file. All implemented functionalities are accessible via the config and CLI args, the control should only be edited if you need new features (or to fix bugs). 

The workflow is intended to be managed via the config files rather than the CLI argumnents. As explained below, the config files allow the control of nearly all aspects of the PyPSA-China execution.  

## Default run (local)
If the pypsa-china environment is  [installed & activated](../../installation/quick_start/), you can lauch run a with the default settings 
```bash title="launch default run"
cd <my_install_location>
snakemake
```
This is a large job! It is unwise to run it locally or on login nodes.

## custom config options
The default config may not fit your solver or preferences. Rather than overwriting it, which may cause merge isues with future versions, it is recommended to make a small `my_config.yaml`.

Snakemake overwrites configs in order that they are added. Your config file then only needs the subset of options you want to overwrite and can be executed using/

```bash title="launch custom run"
cd <workflow_root_folder>
snakemake --configfile=<configs/my_changes.yml> <optional_additional_snakemake_opts>
```

In the config you can specify the solver you would like to use. By default gurobi is chosen.

## dry runs
The `-n` flag from snakemake allows to start a "dry run", which is a mock run that will show what rules would be executed. We **recommend always running with this flag first** before launching an actual run.

```
cd <workflow_root_folder>
snakemake -n <additional_options>
```

## Snakemake command line options

You can find the full list of options [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html). The most important ones are summarised in the [tricks page](../snakemake_tricks)

## Running a module in standalone

You can run any of the modules as standalone python thanks to the `mock_snakemake` function.

## Remote/hpc execution with profiles

The `--profile` arg allows you to specify execution snakemake options via a yaml config file. This is a better alternative to the CLI in many cases. 

The `--profile` is especially useful for specifying and controlling remote execution, for example on an HPC. You will find a slurm HPC example in the config under `pik_hpc_profile/`. This allows you to set the compute resources per rule. Note that the profile must be called `config.yaml` 

The profile can also include any other snakemake flag, such as re-run conditions and verbosity.

To execute the workflow on a remote resource, e.g. a slurm cluster you can do the below from a login node:

```bash title="execute remotely`
cd <workflow_root_folder>
snakemake --profile=<configs/my_profile_parent_dir> <optional_additional_snakemake_opts>
```

## running the examples

EXAMPLES ARE CURRENTLY UNAVAILABLE



# Development and debugging
It is possible to run each script as standalone using the `mock_snakemake` helper utility. The python file will run the __main__ script, reading the Snakefile.

### specific settings
You can edit the wildcards in mocksnakemake. You can also mock passing a configfile ontop of defaults by adding it to the snakefile (add configfile:"my_config" after the default configs)  

# Questions?
Please contact us if needed. Note that pypsa-China-PIK is currently under active development and we recommend waiting until the alpha or first stable release.