# Extended Joint Probability Method for Compound Flooding

This repository contains code and data supporting the paper:

**Mark S. Bartlett, Nathan Geldner, Zach Cobell, Luis Partida, Ovel Diaz, David R. Johnson, Hanbeen Kim, Brett McMann, Gabriele Villarini, Shubra Misra, Hugh J. Roberts, Muthukumar Narayanaswamy** *Extending the Joint Probability Method to Compound Flooding: Statistical Delineation of Transition Zones and Design Event Selection.* arXiv preprint arXiv:2511.03871v2.

## Overview

Compound flooding from the combined effects of extreme storm surge, rainfall, and river flows poses significant risks to infrastructure and communities—as demonstrated by hurricanes Isaac and Harvey. This repository contains the code for the data analysis and figures documenting the pilot study demonstrating a formal extension of the Joint Probability Method (JPM). This extemsion includes linking the foundation of coastal surge risk analysis to incorporate hydrologic drivers for quantifying compound flood risk, statistically delineating compound flood transition zones (CFTZs), and determining design events.

### Key Innovations

1. **Unified Probabilistic Framework**: Integrates the likelihood of the compound flood response from coastal drivers and hydrologic drivers (not just driver co-occurrence) for both tropical and non-tropical storms within a single probabilistic structure

2. **Statistical Transition Zone Delineation**: Provides statistical definition of compound flood transition zones based on exceedance probabilities across multiple return periods, rather than event-specific analysis

3. **Design Storm Selection**: Enables systematic identification of design storms that produce specified return period flood depths, moving beyond design based solely on driver likelihoods

4. **Hydrologic-Coastal Coupling**: Incorporates rainfall fields, antecedent soil moisture, and baseflow as stochastic marked Poisson processes alongside coastal storm surge dynamics

### Theoretical Foundation

The extended JPM builds on foundational concepts from stochastic ecohydrology, where storm processes are modeled as marked Poisson processes to derive analytical probabilistic descriptions of watershed states and fluxes. The methodology extends traditional JPM storm variables **x**<sub>JPM</sub> = {*x*<sub>l</sub>, *c*<sub>p</sub>, θ, *R*<sub>max</sub>, *v*<sub>f</sub>} to include:

- **Rainfall fields** *R*(*x*, *y*, *t*): Spatially and temporally varying precipitation during storm events
- **Antecedent soil moisture** *s*<sub>0</sub>: Pre-storm watershed wetness conditions
- **Baseflow** *Q*<sub>b</sub>: Initial river discharge influencing fluvial flood potential

This extension enables probabilistic characterization of flood response across the full spectrum of compound flooding mechanisms: coastal surge, pluvial (rainfall-driven), and fluvial (river-driven) flooding.

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
│       ├── Compound-Flooding.pdf               # Compound flooding illustration
│       ├── Fig. A2.py                          # Appendix figure A2 generation
│       ├── Fig. A3.py                          # Appendix figure A3 generation
│       ├── Figs. 5. 6, 7, 8, 9. 13.ipynb       # Main text figures 5-9, 13
│       ├── Figures10,11,12.ipynb               # Main text figures 10-12
│       ├── Non-tropical-Contributation.pdf     # Non-tropical storm contribution
│       ├── transects.pdf                       # Cross-sectional flood transects
│       ├── transition_compare.pdf              # CFTZ comparison analysis
│       ├── transition_panels.pdf               # Multi-panel CFTZ visualization
│       └── transition_percent_panels.pdf       # Percentage-based CFTZ analysis
│
├── src/                                         # Source code modules
│   └── data/
│       └── __init__.py
│
└── README.md
```

## Methodology

### Extended JPM Formulation

The extended JPM formulates the annual maximum flood depth cumulative distribution function (CDF) by:

1. **Storm Frequency Integration**: Combining tropical and non-tropical storm arrival rates with appropriate probability weighting based on climatological frequencies

2. **Joint Probability Structure**: Integrating flood response models over the joint probability distribution of meteorological (storm surge parameters, rainfall) and hydrologic variables (antecedent soil moisture, baseflow)

3. **Stochastic Process Representation**: Treating hydrologic drivers as stochastic variables with explicit probabilistic structure derived from marked Poisson process theory, connecting to the broader stochastic hydrology literature

The mathematical framework enables:
- **Statistical CFTZ delineation** based on exceedance probabilities for multiple return periods
- **Design storm identification** that produces target return period flood depths
- **Quantification of compound interactions** that modify flood risk compared to single-driver analysis

### Computational Workflow

The notebooks implement the extended JPM through the following computational steps:

1. **Component AEP Calculation** (`TC_event_AEP_*_rasters.py`): Generate spatially-explicit annual exceedance probability rasters for each flood mechanism (storm surge, pluvial, fluvial) by integrating over the joint distribution of relevant drivers

2. **Combined AEP Synthesis** (`AEP_rasters_TC_and_overall.py`): Integrate tropical cyclone and non-tropical contributions to produce overall flood depth exceedance probability surfaces

3. **CFTZ Delineation** (data processing): Identify transition zones where compound interactions significantly modify flood depths compared to single-driver scenarios, at multiple return periods (10, 50, 100, 500 years)

4. **Visualization and Analysis** (figures notebooks): Generate publication-quality figures showing spatial patterns of compound flood risk, transition zone extents, and mechanism contributions

## Case Study: Lake Maurepas, Louisiana

The methodology is demonstrated for the coastal region around Lake Maurepas, Louisiana, where results show:

- **CFTZ Extent**: More than double the area of prior event-specific delineations, demonstrating the importance of statistical rather than single-event analysis

- **Compound Interactions**: Increase flood depths by up to 0.7 meters (2.25 feet) compared to coastal-only scenarios, with spatial variation in the relative importance of surge, pluvial, and fluvial mechanisms

- **Multi-Hazard Characterization**: Provides return period flood depth maps that properly account for the joint occurrence of multiple drivers and their interactions

- **Design Storm Insights**: Identifies that design storms for compound flooding differ systematically from those based on driver likelihoods alone

## Installation

### Requirements

The code requires standard Python scientific computing packages and geospatial libraries:

```
numpy
pandas
matplotlib
scipy
geopandas
rasterio
shapely
jupyter
```

Additional requirements:
- Python 3.8 or higher
- Sufficient memory for raster processing (16+ GB recommended)
- Geospatial data processing capabilities

A comprehensive `requirements.txt` will be added in future versions.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Mark-S-Bartlett/extended-JPM.git
cd extended-JPM
```

2. Install Python dependencies:
```bash
pip install numpy pandas matplotlib scipy geopandas rasterio shapely jupyter
```

3. Verify installation by opening a Jupyter notebook:
```bash
jupyter notebook
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

