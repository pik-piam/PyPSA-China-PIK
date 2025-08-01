
# Quick Start

!!!note "System requirements"
    With the low-resolution settings, PyPSA-China-PIK will run on a local machine or laptop. Solving a full year at hourly resolution will require a high performance cluster or server with around 50GB of RAM - depending on your settings.

## Installation

1. Install the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager. 

    It may be possible to run the workflow with a better manager such as `uv` but it will not work out of the box

    === "Unix/MacOs/WSL"

        !!!note "Conda tips"

            Conda is a rather large install with a GUI and features not required by PyPSA-China. You may prefer to use the lighter miniconda, which is our recommendation. 
            
            You can check whether you already have conda with `which anaconda` or `which conda`. Newer condas have a faster dependcy solver - as the package is rather large we strongly recommend you update to `v> 2024.10`.

    === "Windows"

        !!!note "Conda tips"

            Conda is a rather large install with a GUI and features not required by PyPSA-China. You may prefer to use the lighter miniconda, which is our recommendation
            
            You can check whether you already have conda with `where anaconda` or `where conda`. Newer condas have a faster dependcy solver - as the package is rather large we strongly recommend you update to `v> 2024.9`.


2. Setup the environment 
    This can take a long time 

    === "Unix/MacOS/WSL"
        ```bash title="install dependencies"
        cd <workflow_location>
        conda env create --file=workflow/envs/environment.yaml
        ```
3. Activate environment
    === "Unix/MacOS/WSL"
        ```bash title="activate environment"
        source activate pypsa-china
        ```
    === "Windows"
        ```bash title="activate environment"
        conda activate pypsa-china
        ```
4. Fetch data
    herefore have to run data fetches
5. Install a Solver: e.g. [gurobi](https://www.gurobi.com/) [HiGHS](https://highs.dev/) or cplex. The current default configuration uses gurobi.
6. [Run locally](#local_exec)


## Testing the installation

=== "Unix/MacOS/WSL"
    ```bash title="Test install"
    source activate pypsa-china
    cd <workflow_location>
    pytest tests/integration
    ```
=== "Windows"
    ```bat title="Test install"
    conda activate pypsa-china
    cd <workflow_location>
    pytest tests/integration
    ```