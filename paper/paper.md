---
title: "PyPSA-China: an open-source model of the energy transition in China" 


tags:
  - Python
  - optimization
  - energy systems
  - China

date: 15 August 2025


# Purely financial (such as being named on an award) and organizational (such as general supervision of a research group) contributions are not considered sufficient for co-authorship of JOSS submissions, but active project direction and other forms of non-code contributions are. The authors themselves assume responsibility for deciding who should be credited with co-authorship, and co-authors must always agree to be listed. In addition, co-authors agree to be accountable for all aspects of the work, and to notify JOSS if any retraction or correction of mistakes are needed after publication.

authors:
  - name: Ivan Ramirez
    affiliation: '1'
    contributions: development, project direction, code review, documentation, validation

  - name: Yanlei Zhu
    affiliation: '2' # also 1?
    contributions: development, code review, documentation#TODO

  - name: Falko Ueckerdt
    affiliation: '1'
    contributions: project direction

  - name: Adrian Odenweller
    affiliation: '1'
    contributions: development (IAM coupling)

  - name: Chen Chris Gong
    affiliation: '1'
    contributions: #TODO

  - name: Robert Pietzcker
    affiliation: '1'
    contributions: validation

  - name: Gunnar Luderer
    affiliation: '1'
    contributions: #TODO

affiliations:
  - index: 1
    name: Potsdam Institute for Climate Impact Research, Potsdam, Germany
  - index: 2
    name: College of Environmental Sciences and Engineering, Peking University, Beijing, China
---


# Summary
<!-- Required section 
A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
-->

Open-source energy system models are key tools for fostering transparent and productive discussions on how the energy transition should proceed. However, no such model exists specifically for China. To adress this gap we present **PyPSA-China-PIK** - a fully open-source, open-data model of China's electricity sector. The model minimises the operation and investment costs of supplying electricity and other energy carriers subject to user-set constraints. These constraints can include  environmental limits - such as carbon prices or emisisons budgets - as well as operational reserve margins to ensure a realiable supply.

Importantly, PyPSA-China runs at a high spatio-temporal resolution, enabling accurate assessment of the benefits and challenges of high renewable energy penetration. The presented version builds upon a previously published model but incorporates significant upgrades and has been restructured to support ease of use and community development. Uniquely, the model can also be coupled with the REMIND Integrated Assessment Model [@baumstarkREMIND2021], enhancing the plausibility of long-term transition pathways.

# Statement of need
<!-- Required section -->
Internationally agreed emission reduction targets and the rapidly decreasing costs of renewable energy generation and storage mean that electricity systems around the world are poised for major transformation. Nowhere is this truer than in China, the world's largest electricity producer, largest greenhouse gas emitter and fastest increasing renewable capacity. As in all other countries, China’s energy transformation involves myriad actors with different motivations and interests. Finding cost-efficient solutions, transparently assessing risks and determining how -and if - likely winners and losers should be compensated requires an open debate. 

Yet there few, if any, open-source tools available to inform the debate in China and verify claims based on close-sourced results. High-quality studies have used expensive commercial software (e.g. PLEXOS [@abhyankarAchieving80Carbonfree2022] ) or proprietary code (sometimes available on request ) and data []. 

`PyPSA-China` stands out due to its open-source philosophy and focus on China [@zhouMultienergySystemHorizon2024]. We introduce a restructured version PyPSA-China-PIK, aimed at collaborative development and modelling, that supports coupling to macro-economic models, and with significant upgrades. 
<!-- In what terms to talk about coupling ? -->


<!--- RESEARCH APPLICATION 
- need obvious research application 
 
- Your software should be a significant contribution to the available open source software that either enables some new research challenges to be addressed or makes addressing research challenges significantly better (e.g., faster, easier, simpler).

-->

# Model Overview
PyPSA-China (PIK) is best understood as a workflow built the Python Power System Analysis ([PyPSA](https://pypsa.readthedocs.io/en/stable/)) modeling framework[@PyPSA]. The workflow collects relevant energy systems data, then formulates it into a PyPSA least-cost optimization problem and finally analyses the solution. [Framework described a number of times]. The cost minimization is separate and can be performed by a number of standalone solvers. The workflow is orchestrated by the `snakemake` management system and is highly modular. All execution options are controlled by `yaml` configuration files. 

Officially available data for load (demand) is limited to Provincial and autonomous region resolution. This fixes the spatial network resolution to 32 nodes, which is adequate for long-term capacity expansion planning but not for analysing high-voltage AC power flows. Time resolution is flexible down to 1 hour. 

[docs/workflow]
Figure 1 the workflow

# Functionalities

The core PyPSA-China PIK functionalities are:
- Optimisation of electricity generation, transmisision and storage capacity expansion. Least-cost dispatch.
- Modelling of renewable availability and potential based on historical weather data using the `atlite` package [@atlite2021]. Renewable resources can be aggregated by capacity factor at sub-node resolution, which is essential for large provinces such as Inner Mongolia.
- Modelling of hydroelectricity based on atlite and dam cascades[@liuRoleHydroPower2019].
- Co-optimisation of electricity and heat. Heat storage, production and combined heat-and-power generation can be included.
- electric vehicles ? @beijingzyl to add the non-flex implementation?
- Operational reserves margins scaling with VRE supply and load. This allows for the optimisation of contigencies.
- Reporting of supply & demand time series, yearly costs, energy balances, capacities and more. 

## Pathways
An advanced feature is using a transformation pathway from the REMIND macro-economic integrated assessment model [@baumstarkREMIND2021]

# Validation
- China dispatch is not based on merit order
- show 2020 data? do we want to add a PM2.5 limit for some regions to dispatch gas?
- good agreement, key difference is gas as noted by others
- this is because gas dispatch is set by limitations on PM2.5 for heating

# Related work <!-- (include?) -->
- pypsa-eur
- pypsa-earth
# Acknowledgements
Energy Foundation China grant <ID> #TODO