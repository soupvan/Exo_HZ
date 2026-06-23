# Exo_HZ
Final project for UMD AOSC 247: Analyzing NASA Exoplanet Archive data to identify potentially habitable Earth-like exoplanets with atmospheres.

# Project description:
Based on the Project Description you submitted earlier, your final coding project and short 10-minute presentation will address a scientific question in your field of study, and should include reading in of actual data, analysis, and presentation of 4 or more visual results (plots, maps, movies, etc.). The coding effort should be about the same as 4 homework assignments, plus acquiring your data. If you are already conducting research, this is a great opportunity to translate an aspect of your work into Python. Your presentation should include your research question, results, challenges you encountered in coding your project with Python, and sections of code you are particularly happy with.

# Finding Earth-Like Exoplanets

Final project I chose for AOSC 247 Scientific Programming: Python, at the University of Maryland.

For this project, I used data from the NASA Exoplanet Archive and Exoplanet.eu to look for exoplanets that are most similar to Earth and determine whether any could potentially support liquid water or any atmosphere at all.

The project combines and analyzes data entries from thousands of confirmed and candidate exoplanets and their host stars, then narrows the search down to confirmed rocky planets in the habitable zone.

## What the project does

- Merges / cleans data from multiple exoplanet catalogs
- Creates an H-R diagram of exoplanet host stars
- Classifies planets by density 
- Finds rocky planets located in their star's habitable zone
- Calculates an Earth Similarity Index for each candidate
- Ranks the most Earth-like planets
- Investigates atmospheric data for the highest-ranked planet with available transmission spectroscopy observation data

## Results

Out of thousands of exoplanets, the analysis identified 34 rocky planets located within their habitable zones. These planets were ranked using an Earth Similarity Index based on radius, density, and equilibrium temperature.

The project ultimately focused on TRAPPIST-1 e, one of the most Earth-like known exoplanets with available transmission spectrum data. I used TRAPPIST-1 b as a reference spectrum to help remove stellar contamination for TRAPPIST-1 e and examine possible atmospheric absorption features.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy

## Data Sources

- NASA Exoplanet Archive
- Exoplanet.eu

This project gave me experience working with large astronomical datasets, cleaning and merging real-world scientific data, building visualizations, and general exoplanet research.
