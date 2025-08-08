# Configuration Reference

This is documentation for the PyPSA-China configuration (`config/default_config.yaml` & `config/technology_config.yaml`). The configuration file controls various aspects of the PyPSA-China energy system modeling workflow.

## Table of Contents 

- [Run Configuration](#run-configuration)
- [File Paths](#file-paths)
- [Grid Topology](#grid-topology)
- [Scenarios](#scenario-configuration)
- [CO2 Scenarios](#co2-scenarios)
- [Time Snapshots](#snapshots)
- [Logging](#logging)
- [Data Fetch Toggles](#data-fetch-toggles)
- [Atlite Weather Settings](#atlite-weather)
- [Renewable Energy Technologies](#renewable-energy-technologies-atlite)
- [Heat Demand](#heat-demand)
- [Reporting](#reporting)
- [Bus and Carriers](#bus-and-carrier-configuration)
- [Technology Categories](#technology-categories)
- [Sectors & Component](#sector-and-component-switches)
- [Hydro Dams](#hydro-dams)
- [Hydrogen Storage](#hydrogen-storage)
- [Solving](#solving-configuration)
- [Transmission Lines](#transmission-lines)
- [Security Constraints](#security-constraints)
- [Existing Capacities](#existing-capacities)
- [Regions](#region-configuration)
- [Input/Output Settings](#inputoutput-settings)
- [Transmission Efficiency](#transmission-efficiency)
- [Combined Heat and Power (CHP) Parameters](#combined-heat-and-power-chp-parameters)
- [Solar Technology Parameters](#solar-technology-parameters)
- [Heat Pump Configuration](#heat-pump-configuration)
- [Thermal Energy Storage](#thermal-energy-storage)
- [Electricity Sector Configuration](#electricity-sector-configuration)
- [Hydroelectric Power](#hydroelectric-power)
- [Fossil Fuel Ramping Constraints](#fossil-fuel-ramping-constraints)
- [Nuclear Reactor Parameters](#nuclear-reactor-parameters)
- [Economic Parameters](#economic-parameters)


## Usage Notes

1. **File Organization**: Configuration files should be placed in the `config/` directory.

2. **Customization**: Do not edit `default_config.yaml`. Overwrite the variables you need in `my_config.yaml`. See [running section](../running)

3. **Technology Configuration**: Additional technology parameters are defined in separate files in  `config/technology_config.yaml` 

4. **Solver Selection**: Remember to select a solver that is installed.


## Run Configuration

```yaml
run:
  name: unamed_run
  is_remind_coupled: true
foresight: "overnight"
```

- **`run.name`**: Identifier for the model run. Used for organizing results and logs.
- **`run.is_remind_coupled`**: Boolean indicating whether the model run is coupled with REMIND. Set to `false` for standalone PyPSA-China runs. True overwrites numerous settings.
- **`foresight`**: ["overnight"|"myopic"] Set to `"overnight"` for perfect foresight within one time horizon. Each horizon is solved independently. Set to `myopic` to obtain a sequential pathway of overnight horizons, with the result of the previous horizon acting as the brownfield for the next horizon.

## File Paths

```yaml
paths:
  results_dir: "results/"
  costs_dir: "resources/data/costs/default"
  yearly_regional_load: 
    ac: "resources/data/load/Provincial_Load_2020_2060_MWh.csv"
    ac_to_mwh: 1
```
- **`results_dir`**: Directory where model results are stored
- **`costs_dir`**: Directory containing technology & cost data
- **`yearly_regional_load.ac`**: Path to provincial electricity load data file
- **`yearly_regional_load.ac_to_mwh`**: Conversion factor to MWh (set to 1 if load file is already in MWh)

## Grid Topology

```yaml
edge_paths:
  current: "resources/data/grids/edges_current.csv"
  current+FCG: "resources/data/grids/edges_current_FCG.csv"
  current+Neighbor: "resources/data/grids/edges_current_neighbor.csv"
```
Paths to named transmission line topology files for different scenarios. Predefined
  - `"current"`: Existing transmission network
  - `"current+FCG"`: Fully connected grid (represents current expansion plans)
  - `"current+Neighbor"`: current + all neighbours connected

## Scenario Configuration

A scenario is a set of time horizons with a carbon reduction pathway. The variations become snakemake wildcards.

```yaml
scenario: 
  co2_pathway: ["exp175default"]
  topology: "current+FCG"
  planning_horizons: [year_list]
  heating_demand: ["positive"]
```

- **`co2_pathway`**: List of named CO2 emission scenarios to model
- **`topology`**: Grid topology configuration. Must correspond to an edge_path
- **`planning_horizons`**: Years to model
- **`heating_demand`**: Heating demand scenarios (will be overhauled)

## CO2 Scenarios
Emission reduction pathways. `scenario.co2_pathway` entries must be defined here.

```yaml
co2_scenarios: 
  exp175default: # pathway name
    control: "reduction"
    pathway:
      '2020': 0.0
      '2025': 0.22623418
      # ... additional years
```

Defines CO2 emission reduction scenarios:
- **`control`**: Control mechanism (`"price"`, `"reduction"`, `"budget"`, or `None`)
- **`pathway`**: Yearly values for the price, reduction or budget. Ignored if `control=None`.

## Snapshots
Snapshots are the modelled timestamps

```yaml
snapshots:
  start: "01-01 00:00"
  end: "12-31 23:00"
  bounds: 'both'
  freq: '5h'
  frequency: 5.
  end_year_plus1: false
```
Controls the temporal resolution of the model:
- **`start`**: Start date and time (MM-DD HH:MM format)
- **`end`**: End date and time
- **`bounds`**: Include start/end points (`'both'`, `'left'`, `'right'`)
- **`freq`**: Frequency string for pandas (e.g., `'5h'` for 5-hour intervals)
- **`frequency`**: weight of a snapshot (eg 5 means that the generation for that timestamp will be multiplifed by 5 to get actual MWh values & costs)
- **`end_year_plus1`**: bool: not single year?

!!!Warning "Time sampling"
    Lower resolutions are currently dumbly sampled as 1 in n. Nonetheless it is possible to get mixes and costs close to those of the 1hr resolution.

## Logging

```yaml
logging_level: INFO
logging:
  level: INFO
  format: '%(levelname)s:%(name)s:%(message)s'
```

- **`logging_level`**: Global logging level
- **`logging.level`**: Detailed logging level
- **`logging.format`**: Log message format string

## Data fetch toggles

```yaml
enable:
  build_cutout: false
  retrieve_cutout: false
  retrieve_raster: false
```

Controls whether the data fetch workflow steps are enabled:
- **`build_cutout`**: Build new weather data cutouts
- **`retrieve_cutout`**: Download existing cutouts (not in correct file structure, must be manually copied)
- **`retrieve_raster`**: Download existing raster data

## Atlite Weather

```yaml
atlite:
  cutout_name: "China-2020c"
  freq: "h"
  nprocesses: 1
  show_progress: true
  monthly_requests: true
  cutouts:
    China-2020c:
      module: era5
      dx: 0.25
      dy: 0.25
      weather_year: 2020
```

Configuration for weather data processing:
- **`cutout_name`**: Name of cutout (searched for in `resources/cutouts/`)
- **`freq`**: Temporal frequency for weather data
- **`nprocesses`**: Number of parallel processes
- **`show_progress`**: Display progress bars
- **`monthly_requests`**: Split requests by month
- **`cutouts`**: config(s) for one or more cutout names
  - **`module`**: Weather data source (e.g., `era5`)
  - **`dx`/`dy`**: Spatial resolution in degrees
  - **`weather_year`**: weather year to fetch/use

## Renewable Energy Technologies Atlite

### Wind 
```yaml
renewable:
  onwind | offwind: 
    cutout: cutout-name
    resource:
      method: wind
      turbine: model
    resource_classes:
      min_cf_delta: 0.05
      n: 3
    capacity_per_sqkm: 3
    potential: simple
    natura: false
    clip_p_max_pu: 1.e-2
    min_p_nom_max: 1.e-2
```

### Solar PV

```yaml
  solar:
    cutout: cutout-name
    resource:
      method: pv
      panel: model
      orientation:
        slope: 35.
        azimuth: 180.
    resource_classes:
      min_cf_delta: 0.02
      n: 2
    capacity_per_sqkm: 5.1
    potential: simple
    correction_factor: 0.85
    natura: false
    clip_p_max_pu: 1.e-2
    min_p_nom_max: 1.e-2
```

**Common renewable parameters:**
- **`cutout`**: Weather data cutout name to use
- **`resource.method`**: Resource calculation method
- **`resource.turbine | panel`**: Technology specification
- **`resource_classes`**: Capacity factor binning to reduce provincial aggregation effects
  - **`n`**: Number of resource bins
  - **`min_cf_delta`**: Minimum capacity factor difference between bins
- **`capacity_per_sqkm`**: Max installable Power density (MW/km²)
- **`potential`**: Potential calculation method (`simple` or `conservative`)
- **`correction_factor`**: Technology-specific correction factor
- **`natura`**: Consider nature protection areas
- **`max_depth`**: Maximum water depth for offshore wind (m)
- **`clip_p_max_pu`**: Minimum capacity factor threshold
- **`min_p_nom_max`**: Minimum installable capacity threshold

```yaml
renewable_potential_cutoff: 200  # MW
```
Skip locations with potential below this threshold to reduce problem size.

## Heat Demand

```yaml
heat_demand:
  start_day: "01-04"
  end_day: "30-09"
  heating_start_temp: 15.0
  heating_lin_slope: 1
  heating_offet: 0
solar_thermal_angle: 45
```

- **`start_day`/`end_day`**: Heating season dates (DD-MM format)
- **`heating_start_temp`**: Temperature threshold for heating demand (°C)
- **`heating_lin_slope`**: Linear relationship slope
- **`heating_offset`**: Linear model offset
- **`solar_thermal_angle`**: Solar thermal collector angle (degrees)



## Reporting

```yaml
reporting:
  adjust_link_capacities_by_efficiency: true
```

- **`adjust_link_capacities_by_efficiency`**: PyPSA links capacities are in input. Typical reporting for AC is in output capacitiy. If true Adjust link capacities by efficiency for consistent AC-side reporting.

## Bus and Carrier Configuration

```yaml
bus_suffix: [""," central heat"," decentral heat"," gas"," coal"]
bus_carrier: {
    "": "AC",
    " central heat": "heat",
    " decentral heat": "heat",
    " gas": "gas",
    " coal": "coal",
}
```

Defines bus types and their corresponding energy carriers:
- **`bus_suffix`**: List of bus name suffixes
- **`bus_carrier`**: Mapping of suffixes to carrier types

## Technology Categories

```yaml
Techs:
  vre_techs: ["onwind","offwind","solar","solar thermal","hydroelectricity", "nuclear","biomass","beccs","heat pump","resistive heater","Sabatier","H2 CHP", "fuel cell"]
  conv_techs: ["OCGT", "CCGT", "CHP gas", "gas boiler","coal boiler","coal power plant","CHP coal"]
  store_techs: ["H2","battery","water tanks","PHS"]
  coal_cc: true
  hydrogen_lines: true
```

Technology categorization:
- **`vre_techs`**: Variable renewable energy technologies
- **`conv_techs`**: Conventional generation technologies
- **`store_techs`**: Storage technologies
- **`coal_cc`**: Enable coal with carbon capture retrofit (myopic only). Coal carbon capture new-builds are controled via "vre_techs" for overnight.
- **`hydrogen_lines`**: Enable hydrogen transmission lines

## Sector and component Switches 

```yaml
heat_coupling: false
add_biomass: True
add_hydro: True
add_H2: True
add_methanation: True
line_losses: True
no_lines: False
```

Control which components to include in the model:
- **`heat_coupling`**: Enable heat sector coupling
- **`add_biomass`**: Include biomass technologies
- **`add_hydro`**: Include hydroelectric power
- **`add_H2`**: Include hydrogen technologies (and pipelines)
- **`add_methanation`**: Include methanation processes
- **`line_losses`**: Model transmission line losses
- **`no_lines`**: Disable transmission lines (autartik)

## Hydro Dams

```yaml
hydro_dams:
  dams_path: "resources/data/hydro/dams_large.csv"
  inflow_path: "resources/data/hydro/daily_hydro_inflow_per_dam_1979_2016_m3.pickle"
  inflow_date_start: "1979-01-01"
  inflow_date_end: "2017-01-01"
  reservoir_initial_capacity_path: "resources/data/hydro/reservoir_initial_capacity.pickle"
  reservoir_effective_capacity_path: "resources/data/hydro/reservoir_effective_capacity.pickle"
  river_links_stations: ""
  p_nom_path: "resources/data/p_nom/hydro_p_nom.h5"
  p_max_pu_path: "resources/data/p_nom/hydro_p_max_pu.h5"
  p_max_pu_key: "hydro_p_max_pu"
  damn_flows_path: "resources/data/hydro/dam_flow_links.csv"
```

Hydroelectric dam configuration:
- **`dams_path`**: CSV file with dam locations and characteristics
- **`inflow_path`**: Historical inflow data
- **`inflow_date_start`/`inflow_date_end`**: Date range for inflow data
- **`reservoir_*_capacity_path`**: Reservoir capacity data files
- **`p_nom_path`**: Installed capacity data
- **`p_max_pu_path`**: Maximum capacity factor time series
- **`damn_flows_path`**: Dam flow connections

## Hydrogen Storage

```yaml
H2:
  geo_storage_nodes: ["Sichuan", "Chongqing", "Hubei", "Jiangxi", "Anhui", "Jiangsu", "Shandong", "Guangdong"]
```

- **`geo_storage_nodes`**: Provinces with geological hydrogen storage potential


## Solving Configuration

### General Options

```yaml
solving:
  options:
    formulation: kirchhoff
    load_shedding: false
    voll: 1e5
    noisy_costs: false
    min_iterations: 4
    max_iterations: 6
    clip_p_max_pu: 0.01
    skip_iterations: false
    track_iterations: false
```

- **`formulation`**: Network formulation (`kirchhoff` or `angles`)
- **`load_shedding`**: Allow unserved energy
- **`voll`**: Value of lost load (EUR/MWh)
- **`noisy_costs`**: Add noise to costs for degeneracy handling
- **`min_iterations`/`max_iterations`**: Iteration bounds for iterative solving
- **`clip_p_max_pu`**: Minimum capacity factor threshold
- **`skip_iterations`/`track_iterations`**: Iteration control

## Transmission Lines

```yaml
lines:
  line_length_factor: 1.25
  expansion:
    transmission_limit: vopt
    base_year: 2020
```

- **`line_length_factor`**: Line length increase factor, representing impossibility to build perfectly straight
- **`expansion.transmission_limit`**: Transmission expansion constraint
  - `[v]opt`: Optimal (unconstrained)
  - `v1.03`: Volume-constrained (3% increase limit PER YEAR not horizon)
  - `c1.03`: Cost-constrained (3% increase limit PER YEAR not horizon)
- **`expansion.base_year`**: Reference year for expansion limits (to calculate max exp). Should be year of input network topology file (which has capacities)

## Security Constraints

```yaml
security:
  line_margin: 70  # Max percent of line capacity
```

- **`line_margin`**: Security margin for transmission lines (% of capacity)

## Existing Capacities

```yaml
existing_capacities:
  add: True
  grouping_years: [1980,1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
  threshold_capacity: 1
  techs: ['coal','CHP coal', 'CHP gas', 'OCGT', 'CCGT', 'solar', 'solar thermal', 'onwind', 'offwind','coal boiler','ground heat pump','nuclear']
```

Configuration for incorporating existing power plant capacities:
- **`add`**: Include existing GEM capacities in the model. These are retired only if they reach end of life (determined based on the costs tech config)
- **`grouping_years`**: Years for capacity grouping
- **`threshold_capacity`**: Minimum capacity threshold
- **`techs`**: Technologies to include from existing capacity data

## Region Configuration

```yaml
fetch_regions:
  simplify_tol: 0.5
```

- **`simplify_tol`**: Tolerance for region boundary simplification

## Input/Output Settings

```yaml
io:
  nc_compression:
    level: 4
    zlib: True
```

Controls compression settings for NetCDF output files:
- **`level`**: Compression level (0-9, higher = more compression)
- **`zlib`**: Enable zlib compression

## TECHNOLOGY CONFIG
The below are in `config/technology_config.yaml`

## Transmission Efficiency

```yaml
transmission_efficiency:
  DC:
    efficiency_static: 0.98
    efficiency_per_1000km: 0.977
  H2 pipeline:
    efficiency_static: 1
    efficiency_per_1000km: 0.979
    compression_per_1000km: 0.019
```

Defines transmission efficiency parameters for different carriers & technologies:

### DC Transmission
- **`efficiency_static`**: Base efficiency for DC transmission lines (98%)
- **`efficiency_per_1000km`**: Distance-dependent efficiency factor per 1000 km (97.7% per 1000 km)

### Hydrogen Pipeline
- **`efficiency_static`**: Base efficiency for hydrogen pipelines (100% - no static losses)
- **`efficiency_per_1000km`**: Distance-dependent efficiency factor per 1000 km (97.9% per 1000 km)
- **`compression_per_1000km`**: Energy required for compression per 1000 km (1.9% of transported energy)

The total efficiency for transmission links is calculated as:
```
Total efficiency = efficiency_static × (efficiency_per_1000km)^(distance_km/1000)
```

## Combined Heat and Power (CHP) Parameters

```yaml
chp_parameters:
  eff_th: 0.5304
```
CHP is treated either as a pre-defined values or using a back pressure coeficient (see DK Energy catalogue). For coal CHP use the back pressure (variable ratio). For gas CHP use pre-defined parameter.

**TODO** this is legacy, check implentation is OK. CHP often runs heat first in China but code seems to maintain electric eff

## Solar Technology Parameters

```yaml
solar_cf_correction: 0.85
```

- **`solar_cf_correction`**: Correction factor applied to solar capacity factors (85%)

This factor accounts for various real-world effects that reduce solar Thermal performance compared to theoretical values. **NOT APPLIED TO PV**

## Heat Pump Configuration

```yaml
time_dep_hp_cop: True
```

- **`time_dep_hp_cop`**: Enable time-dependent coefficient of performance (COP) for heat pumps

When enabled, heat pump efficiency varies with ambient temperature conditions throughout the year.

## Thermal Energy Storage

```yaml
water_tanks:
  tes_tau:
    decentral: 3. # days
    central: 180 # days
```

Standing loss parameters for thermal energy storage (water tanks):
- **`decentral`**: Time constant for decentralized thermal storage (3 days)
- **`central`**: Time constant for centralized thermal storage (180 days)

The time constant (tau) determines the rate of thermal losses.

## Electricity Sector Configuration
Partially legacy and to be revised

```yaml
electricity:
  max_hours:
    battery: 6
    H2: 168
  min_charge:
    battery: 0.1 # fraction of e_nom
```

### Storage Parameters
- **`max_hours.battery`**: Maximum storage duration for batteries (6 hours)
- **`max_hours.H2`**: Maximum storage duration for hydrogen storage (168 hours = 1 week)
- **`min_charge.battery`**: Minimum state of charge for batteries (10% of nominal energy capacity)

## Hydroelectric Power

```yaml
hydro:
  hydro_capital_cost: True
  marginal_cost:
    reservoir: 0
  PHS_max_hours: 24 # hours
```

- **`hydro_capital_cost`**: Include capital costs for hydroelectric plants
- **`marginal_cost.reservoir`**: Marginal cost of storage (defaults to zeo)
- **`PHS_max_hours`**: Maximum storage duration for pumped hydro storage (24 hours)



## Fossil Fuel Ramping Constraints

Operational ramping constraints for fossil fuel power plants:
 
```yaml
fossil_ramps:
  tech:
    ramp_limit_up: 0.5 # fraction of p_nom per hour
    ramp_limit_down: 0.5 # fraction of p_nom per hour
```

- **`ramp_limit_up`**: Maximum upward ramping rate (50% of nominal capacity per hour)
- **`ramp_limit_down`**: Maximum downward ramping rate (50% of nominal capacity per hour)


## Nuclear Reactor Parameters
Instead of solving unit commitment problem, which is computationally expensive, can stylise baseload generation.
The upper limit reflects planned and unplanned outages

```yaml
nuclear_reactors:
  p_max_pu: 0.88 # fraction of p_nom, after IEAE
  p_min_pu: 0.7 # fraction of p_nom
```
Operational constraints for nuclear power plants:
- **`p_max_pu`**: Maximum power output (% of nominal capacity)
- **`p_min_pu`**: Minimum power output (% of nominal capacity)


## Economic Parameters

```yaml
costs:
  discountrate: 0.06
  social_discount_rate: 0.02
  USD2013_to_EUR2013: 0.9189
  marginal_cost:
    hydro: 0.
  pv_utility_fraction: 1
```

### Discount Rates
- **`discountrate`**: Financial discount rate for investment decisions (6%)
- **`social_discount_rate`**: Social discount rate for welfare analysis (2%)

The financial discount rate is used for technology investment decisions, while the social discount rate is applied for broader economic impact assessments.

### Currency Conversion
- **`USD2013_to_EUR2013`**: Exchange rate from USD to EUR for 2013 prices (0.9189 EUR/USD) - used for technoeconomic conversion. Weakpoint, only one currency.
Also tech costs are now in Euro2015 from DK EA

### Solar PV Configuration
- **`pv_utility_fraction`**: Fraction of solar PV that is utility-scale (100%)

This parameter distinguishes between utility-scale and residential/distributed solar installations, affecting cost assumptions and grid integration characteristics.

