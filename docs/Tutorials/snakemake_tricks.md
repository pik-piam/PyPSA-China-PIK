
# Snakemake tricks

## Good to know
Snakemake workflow execution is based on input files. If inputs to a rule are missing, outdated or changed, snakemake will work out the compute graph needed to build all required data.

The first rule in the snajefile is the default target rule. Pseudo rules such as `plot_all` can be put at the top to call a whole workflow. Otherwise the target rule can be specified with `-t`.

Snakemake executes lazily based on change criteria, such as changes in input data. Most of the decision are based on timestamps. You can specify change criterias via a profile or the [cli](https://snakemake.readthedocs.io/en/stable/executing/cli.html)

## Useful command line arguments 
- `--touch` : update all the timestamps of previously runned files. Good if you don't want to re-run a partially completed workflow
- `-n`: dry run - see what would be computed
- `-t`: specify the target rule
- `f`: specify the target output file

## DAG printout
It is possible to visualise the workflow with 
`snakemake results/dag/rules_graph.png -f`
