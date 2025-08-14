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

- A fully open-source, open-data model of the Chinese electricity sector. 
- The model minimises operation and investment costs of supplying present or future demand under user-set constraints.
- Constraints can include maximum environental impacts, for example represented by carbon price or budget, operational reserve margins ensuring adequate supply as well as custom 
- It runs at sufficient resolution to accurately model the benefits/impacts of high Renewable penetration 

# Statement of need
<!-- Required section -->

<!--- POSSIBLY too political   -->

- Energy/power systems around the world are set for major upheaval/transformations. This is due to the falling costs of renewable and the imperative of abating green-house-gas emissions (or national commitments). (Maybe add general trend of electrification)
- There are also ongoing and intense debates about the role of markets and their optimal structure.
- Yet given suprisingly few open models, which gives outsized influence to incumbents and/or interest groups.

- Particulary true of China: coal exit, increasing consumption per capita and likely change in relative consumption of industry and households. Debate on role of Ultra-high voltage lines and competing influences of provincial, private and central actors.

- Pathways

- Existing packages: pypsa-china (Xiaowei), PyPSA-earth, others not-open source. GridPath
- substantial development on previous model of China, with easier and better code that works out of the box, new features such as renewable grades and 

<!--- RESEARCH APPLICATION 
- need obvious research application 
 
- Your software should be a significant contribution to the available open source software that either enables some new research challenges to be addressed or makes addressing research challenges significantly better (e.g., faster, easier, simpler).

 -->

#

# Acknowledgements
Energy Foundation China grant <ID> #TODO