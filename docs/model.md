# Model description

The model minimizes system costs over the full 8760 hours of a year. Optionally, the resolution can be decreased (sample every n hours) to speed up calculations.

## Objective function

The objective function of the model is to minimize the total yearly system costs:

$$
\text{Min} \left[ \text{Yearly System Costs} \right ] =\text{Min} \left[ \sum_{r,t} \text{Annualised CAPEX} \right ] + \left[ \sum_{r,t} \text{OPEX} \right ]
$$

subject to
- meeting energy demand for each region *r* and time *t*
- transmission constraints between nodes
- wind, solar and hydro availability for all *r, t*
- respecting geographical potentials for renewables
- emission budget (or pricing added to OPEX)
- flexibility (optional load shedding, storage, etc)


## Renewable resources
Renewable resources are computed using the `atlite` package which comptues potentials and availability series from ERA5 weather data (0.25x0.25 degree^2 cells). 

Renewable supply is described at subprovincial resolution by default, with binning of cells

## Emission budgets & prices

A global constraint

## Electricity Demand

### REMIND coupling
- scaling
- eVs can be separated

### EV

## Heat demand

## Combined heat and power
- Coal CHP plants are considered to be extraction mode, with a flexible but constrained heat to power ratio. The ratio at time $t$ and for region $r$ is a model decision variable.
- OCGT CHP is considered to be 

## Brownfield data
- brownfield capacities can be toggled on or off in the configuration (default: on)
- brownfield capacities are derived from the Global Energy Monitor Integrated Plant tracker data set. The included operation/pipeline statuses can be selected in the `global_energy_monitor.yaml` config. 
- only plants that have been built before the plan_year and not retired by the plan year are added to the model

Technologies:
- OCGT
- CHP CCGT
- CHP coal
- coal power
- hydro (always on)
- nuclear (always on)
- PHS
- solar & wind (considered to have filled the best capacity factor)
- transmission
- & more

## Transmission
- due to resolution limitations, interprovincial transmission is simplified to dispatchable lossy HVDC

## Operational reserves
It is possible to toggle operational reserves in the configuration. The reserves are determined by a contigency + a fraction of the load and VRE dispatch at time $t$ 