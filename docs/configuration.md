# Configuration Reference

This is documentation for the PyPSA-China configuration (`config/default_config.yaml` & `config/technology_config.yaml`). The configuration file controls various aspects of the PyPSA-China energy system modeling workflow.

## Table of Contents

- [Run Configuration](#run-configuration)
- [File Paths](#file-paths)
- [Scenario Configuration](#scenario-configuration)
- [CO2 Scenarios](#co2-scenarios)
- [Time Snapshots](#time-snapshots)
- [Logging](#logging)
- [Feature Toggles](#feature-toggles)
- [Atlite Weather Data](#atlite-weather-data)
- [Renewable Energy Technologies](#renewable-energy-technologies)
- [Heat Demand](#heat-demand)
- [Technology Configuration](#technology-configuration)
- [Reporting](#reporting)
- [Bus and Carrier Configuration](#bus-and-carrier-configuration)
- [Technology Categories](#technology-categories)
- [Component Configuration](#component-configuration)
- [Hydro Dams](#hydro-dams)
- [Hydrogen Storage](#hydrogen-storage)
- [Grid Topology](#grid-topology)
- [Solving Configuration](#solving-configuration)
- [Transmission Lines](#transmission-lines)
- [Security Constraints](#security-constraints)
- [Existing Capacities](#existing-capacities)
- [Region Configuration](#region-configuration)
- [Input/Output Settings](#inputoutput-settings)

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

## Renewable Energy Technologies (Atlite)

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

## Sector & component Switches 

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

This configuration provides a comprehensive setup for modeling the Chinese power system with PyPSA-China. Adjust parameters according to your specific modeling requirements and computational resources.


