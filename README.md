# Extended Joint Probability Method for Compound Flooding

This repository contains code and data supporting the paper:

**Mark S. Bartlett, Nathan Geldner, Hugh J. Roberts, Zach Cobell, Brett McMann,
Luis Partida, Ovel Diaz, David R. Johnson, Hanbeen Kim, Gabriele Villarini,
Shubhra Misra, Muthukumar Narayanaswamy.** *Extending the Joint Probability
Method to Compound Flooding: Statistical Delineation of Transition Zones and
Design Event Selection.* arXiv preprint arXiv:2511.03871v2 (under review,
*Water Resources Research*).

## Overview

Compound flooding from the combined effects of extreme storm surge, rainfall, and
river flows poses significant hazards to infrastructure and communities—as
demonstrated by the August 2016 Louisiana flood and Hurricane Isaac (2012). This
repository contains the code for the data analysis and figures documenting the
pilot study that demonstrates a probabilistic extension of the Joint Probability
Method (JPM). The extension links the probabilistic foundation of coastal surge
hazard analysis to hydrologic drivers, enabling quantification of the compound
flood-depth distribution, statistical delineation of compound flood transition
zones (CFTZs), probabilistic flood-depth attribution, and response-based design
event selection.

### Key Innovations

1. **Unified Probabilistic Framework**: Derives the probability distribution of
   the compound flood response from both coastal and hydrologic drivers (not just
   driver co-occurrence), for tropical and non-tropical storms, within a single
   probabilistic structure.

2. **Statistical Transition Zone Delineation**: Defines compound flood transition
   zones from the flood-depth exceedance-probability distribution across annual
   exceedance probabilities (AEPs), rather than from a single event.

3. **Probabilistic Flood-Depth Attribution**: Decomposes each AEP flood depth into
   its hydrologic and coastal-surge contributions, quantifying where and how
   strongly each mechanism drives the compound response.

4. **Response-Based Design Storm Selection**: Identifies design storms by
   conditioning directly on the flood response, yielding storms that produce a
   target AEP flood depth—moving beyond selection based on driver likelihoods
   alone.

5. **Hydrologic-Coastal Coupling**: Represents storm arrivals as a marked Poisson
   process and carries stochastic rainfall fields, antecedent soil moisture,
   storage capacity, and baseflow alongside coastal storm-surge dynamics.

### Theoretical Foundation

The extended JPM builds on stochastic ecohydrology, where storm arrivals are
modeled as a marked Poisson process to derive analytical probabilistic
descriptions of watershed states and fluxes. The methodology extends the
traditional JPM storm variables **x**<sub>JPM</sub> = {*x*<sub>l</sub>,
*c*<sub>p</sub>, θ, *R*<sub>max</sub>, *v*<sub>f</sub>} to include:

- **Rainfall fields** **r**(*t*): Spatially and temporally varying precipitation
  during storm events
- **Antecedent soil moisture** *s*: Pre-storm watershed wetness
- **Storage capacity** *w*: Available soil-water storage governing runoff
  generation
- **Baseflow** *q*<sub>b</sub>: Antecedent river discharge influencing fluvial
  flood potential

This extension enables probabilistic characterization of the flood response
across the full spectrum of compound flooding mechanisms: coastal surge, pluvial
(rainfall-driven), and fluvial (river-driven) flooding.

## Repository Structure

```
extended-JPM/
├── notebooks/                                   # Annual Exceedance Probability (AEP) raster generation
│   ├── AEP_rasters_TC_and_overall.py           # Combined tropical cyclone and overall AEP
│   ├── TC_event_AEP_fluvial_rasters.py         # Fluvial (river) flood component AEP
│   ├── TC_event_AEP_pluvial_rasters.py         # Pluvial (rainfall) flood component AEP
│   └── TC_event_AEP_storm_surge_rasters.py     # Storm surge component AEP
│
├── reports/                                     # Analysis outputs and publication materials
│   ├── data/                                    # Processed data and results
│   │   ├── CFTZ_outline_10_yr.geojson          # 10-year return period CFTZ boundary
│   │   ├── CFTZ_outline_50_yr.geojson          # 50-year return period CFTZ boundary
│   │   ├── CFTZ_outline_100_yr.geojson         # 100-year return period CFTZ boundary
│   │   ├── CFTZ_outline_500_yr.geojson         # 500-year return period CFTZ boundary
│   │   ├── flood_zones_FeaturesToJSON.geojson  # Flood zone delineations
│   │   ├── FigureA2_Data.txt                   # Data for appendix figure A2
│   │   ├── Fig_A3_Data_18TCs34GagesUncertaintyQuantification.xlsx
│   │   └── HMS_re_calibration.xlsx             # HMS model recalibration parameters
│   │
│   └── figures/                                 # Scripts and PDFs for paper figures
│       ├── Compound-Flooding.pdf                 # Compound flooding illustration
│       ├── Fig. A2.py                            # Appendix figure A2 generation
│       ├── Fig. A3.py                            # Appendix figure A3 generation
│       ├── Fig. A114.py                          # Appendix figure A1 and A4 generation
│       ├── Figures5,6,7,8,9,10,11,12,and16.ipynb # Main text figures 5-12, 16
│       └── Figures13,14,15.ipynb                 # Main text figures 13-15
│
├── src/                                         # Source code modules
│   └── data/
│       └── __init__.py
│
└── README.md
```

## Methodology

### Extended JPM Formulation

The extended JPM formulates the annual maximum flood-depth cumulative
distribution function (CDF) by:

1. **Storm Frequency Integration**: Combining tropical and non-tropical storm
   arrival rates, weighted by their climatological frequencies, into a single
   annual flood-depth distribution.

2. **Joint Probability Structure**: Integrating the flood response over the joint
   probability distribution of meteorological drivers (storm-surge parameters and
   stochastic rainfall fields) and antecedent hydrologic state (soil moisture,
   storage capacity, and baseflow).

3. **Stochastic Process Representation**: Representing storm arrivals as a marked
   Poisson process—the shared foundation of the JPM and stochastic
   ecohydrology—and describing the hydrologic drivers as stochastic states with
   explicit probabilistic structure, connecting the event-scale JPM to the
   long-term stochastic hydrology literature.

The mathematical framework enables:
- **Statistical CFTZ delineation** based on flood-depth exceedance probabilities
  across annual exceedance probabilities (AEPs)
- **Probabilistic flood-depth attribution** decomposing each AEP depth into
  hydrologic and coastal-surge contributions
- **Response-based design storm selection** that conditions directly on the flood
  response to identify storms producing a target AEP flood depth
- **Quantification of compound interactions** that increase flood hazard relative
  to single-driver analysis

### Computational Workflow

The notebooks implement the extended JPM through the following computational steps:

1. **Component AEP Calculation** (`TC_event_AEP_*_rasters.py`): Generate spatially-explicit annual exceedance probability rasters for each flood mechanism (storm surge, pluvial, fluvial) by integrating over the joint distribution of relevant drivers

2. **Combined AEP Synthesis** (`AEP_rasters_TC_and_overall.py`): Integrate tropical cyclone and non-tropical contributions to produce overall flood depth exceedance probability surfaces

3. **Visualization and Analysis** (figures notebooks under the reports folder): Generate publication-quality figures showing spatial patterns of compound flood risk, transition zone extents, and mechanism contributions

## Case Study: Lake Maurepas, Louisiana

The methodology is demonstrated for the coastal region around Lake Maurepas,
Louisiana, where results show:

- **CFTZ Extent**: The statistically defined compound flood transition zone is
  more than double the area of prior event-based delineations (2038 km² vs.
  938 km²), demonstrating the importance of characterizing the full flood-depth
  distribution rather than a single event.

- **Compound Interactions**: Compound processes increase flood depths by up to
  0.7 m relative to the maximum of the individually simulated pluvial-, fluvial-,
  and coastal-only responses, with spatial variation in the relative importance
  of surge, pluvial, and fluvial mechanisms.

- **Probabilistic Hazard Characterization**: Produces annual-exceedance-probability
  (AEP) flood-depth maps that account for the joint occurrence of multiple drivers
  and their nonlinear interactions.

- **Flood-Depth Attribution**: Decomposes each AEP flood depth into its hydrologic
  and coastal-surge contributions, showing a continuous transition from
  hydrologically dominated flooding upstream to surge-dominated flooding near the
  coast, with the greatest attribution variability inside the CFTZ.

- **Response-Based Design Storms**: Shows that design storms conditioned on the
  flood response differ systematically from those selected on driver likelihoods
  alone, and identifies multiple equiprobable design storms for a target AEP depth.

## Installation

### Requirements

A comprehensive `requirements.txt` will be added in future versions.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Mark-S-Bartlett/extended-JPM.git
cd extended-JPM
```

## Citation

If you use this code or methodology in your research, please cite:

**Extended JPM Paper:**
```bibtex
@article{bartlett2025compound,
  title={Extending the Joint Probability Method to Compound Flooding: Statistical Delineation of Transition Zones and Design Event Selection},
  author={Bartlett, Mark S. and Narayanaswamy, Muthu and Geldner, Nicholas and Toro, Valeria N. and Begmohammadi, Arezoo and Rivera-Hern{\'a}ndez, Xochitl and Munroe, Robert and Cultra, Elizabeth and Colten, Craig E. and Brody, Samuel D. and Porporato, Amilcare},
  journal={arXiv preprint arXiv:2511.03871},
  year={2025}
}
```

**Stochastic Ecohydrology Foundation:**
```bibtex
@article{bartlett2025stochastic,
  title={Stochastic Ecohydrological Perspective on Semi-Distributed Rainfall–Runoff Dynamics},
  author={Bartlett, Mark S. and Cultra, Elizabeth and Geldner, Nicholas and Porporato, Amilcare},
  journal={Water Resources Research},
  year={2025}
}
```
## Contact

For questions about the code, methodology, or data:

- **Mark S. Bartlett**: Mark.Bartlett@gmail.com
- **Muthu Narayanaswamy**: mnarayanaswamy@thewaterinstitute.org

