# Good to know (misc info to be sorted later)


## Time resolution
- the time step can be chosen in discrete hours. The time step applies to all snapshots, it is not currently possible to have different time meshes for subsets of snapshots.
- the time step simply choses 1 from every $$\delta t$$. For example 4 hours will take 0:00,4:00,8:00,12:00,16:00,20:00
- the time step weight should be chosen such that $$n_{steps}*n_{weights} = n_{modelled_years}*8760$$
- leap days are currently dropped

A better time meshing should be developed at some point.

## Links vs Generators
Conventional Gas plants, CHP & BECSS are implemented as links. For example, OCGT consists of a generator with carrier gas that produces gas at the gas bus, with the fuel costs as marginal costs & a link from the gas bus to AC. CHP gas follows a similar approach with heat as output in addition. 

Other thermal powerplants such as coal and gas+ccs are implemented as simple generators. The mixture of both leads to complications in the post-processing.

## Reporting

### Link capacities
In pypsa the capacities for a link `bus0->bus1` correspond to the max power at `bus0`. Reported nameplate capacities for AC, typically refer to the AC power. An option has been added to use the AC power in the statistics.

Similarly, the pypsa lossy transport link implementation (e.g. H2 pipelines or HVDC) requires a forward and reversed link. The reversed link is typically fictitious and only added for convenience. 

### Carrier groups
- 