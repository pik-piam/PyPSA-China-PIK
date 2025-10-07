# Model description

The model minimizes system costs over the full 8760 hours of a year. Optionally, the resolution can be decreased (sample every n hours) to speed up calculations.

## Objective function

The objective function of the model is to minimize the total yearly system costs:

$$
\text{Min} \left[ \text{Yearly System Costs} \right ] =\text{Min} \left[ \sum_{r,t} \text{Annualised CAPEX} \right ] + \left[ \sum_{r,t} \text{OPEX} \right ]
$$

subject to
- meeting energy demand for each region *r* and time *t*
- transmission constraints between nodes (volume or cost expansion)
- wind, solar and hydro availability for all *r, t*
- respecting geographical potentials for renewables
- emission budget (or pricing added to OPEX)
- flexibility (optional load shedding, storage, etc)


## Renewable resources
Renewable resources are computed using the `atlite` package which comptues potentials and availability series from ERA5 weather data (0.25x0.25 degree<sup>2</sup> cells). 
Renewable supply is described at subprovincial resolution by default, with user-set binning of cells. Each bin is aggregated into a generator with it's own potential and availability curve.

## Emission budgets & prices

There are two options to penalise emissions. 

1. A global constraint can be added (as reduction vs a reference year or budget). This is a hard bound.
2. An emission price can be added.

Currently, only CO2 is controlled but it is straightforward to add new GHGs or harmful emissions.

## Electricity Demand
Historical hourly electricity demand for each province is scaled according to future demand projections/scenarios. The default historical data is NDRC data for the [year 2019](https://doi.org/10.5281/zenodo.8322210), which includes peak/valley daily demands and typical hourly profiles. It is possible to split projections by sectors.

### REMIND coupling
In coupled runs, sectoral or total demands are provided by the REMIND IAM. Hourly profiles for each sector are scaled by the REMIND total

### EV
In REMIND coupled mode, Electric Vehicle demand can be separated from the main AC demand. Details to be added.

## Heat demand

Heating demand is split into domestic hot water, centralised space heating and decentralised space heating. In a future release centralised heating will include demand from light industry. Space heating is considered to be seasonal and switched off outside a user-set heating window. Within this window, the profile is determined by the heating degree day (HDD) model. All HDD parameters can be set by the user in the configuration.

- centralised vs de-centralised fraction: is an exogenous input. By default, this data comes from the 2020 Statistics year book.
- SPH provincial demand: is an exogenous input. By default, this data comes from the 2020 Statistics year book.
- SPH national projections: from the [PyPSA-China paper](https://doi.org/10.1049/ein2.12011) "positive or constant". To be reworked
- DHW: demand is exogenous. By default it is based on iea demand projections (An Energy Sector Roadmap to Carbon Neutrality in China, IEA, 2021). Demand is considered constant after 2060
- Totals: the China totals are scaled according to (exogenous) future demand projections.


## Combined heat and power
- Coal CHP plants are considered to be extraction mode, with a flexible but constrained heat to power ratio. The ratio at time $t$ and for region $r$ is a model decision variable.
- OCGT CHP is considered to be 

## Brownfield data
- brownfield capacities can be toggled on or off in the configuration (default: on)
- brownfield capacities are derived from the Global Energy Monitor Integrated Plant tracker data set. The included operation/pipeline statuses can be selected in the `global_energy_monitor.yaml` config. 
- only plants that have been built before the plan_year and not retired by the plan year are added to the model

Technologies available for Brownfield:

- OCGT
- CCGT
- CHP CCGT
- CHP coal
- coal power
- hydro (always on)
- nuclear
- PHS
- solar & wind (considered to have filled the best available capacity factor)
- inter provincial transmission
- & more

## Transmission
- due to resolution limitations, interprovincial transmission is simplified to dispatchable lossy HVDC

## Operational reserves
It is possible to toggle operational reserves in the configuration. The reserves are determined by a contigency + a fraction of the load and VRE dispatch at time $t$ 

## Workflow
The workflow consists of gathering and preparing relevant data, formulating the problem as a PyPSA network object, minimising the system costs using a solver and post-processing the data. An example workflow can be found below.

![PyPSA-China Workflow](./assets/img/pypsa-china-workflow.png)

Each operation is represented by a snakemake `rule`. The `.yaml` configuration options determine which snakemake `rule`s will be executed. This means that the workflow graph will depend on your activated sectors (eg `heat_coupling=False`) and whether coupling to an IAM is selected.

### snakemake

The `snakefile` and included `*.smk` blocks contain the rule declarations. The rules are linked by their inputs and outputs. These links determine the execution order.  All execution options are controlled by the setting files, therefore the `snakefile` does not need to be edited unless you want to *change data inputs* or *extend the model*. 

Data input sources are currently either hardcoded `snakefile` or in the config `paths` section. 

## Key strenghts & limitations

### Strengths
- multi sector, fully open model
- highly configurable code, easy to extend
- easy to replace data

### Limitations
- The full electrical network data is not open and HVAC is there not described. This means operational planning is coarse.
- Connections of renewable generators: HV connections to load centers are not currently included.
- Default projections for demand do not account for future regional variations.
