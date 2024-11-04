# OBSERVATION TOOLS

The OBSERVATIONS subpackage provides routines and tools for working with observational galaxy survey data, specifically from the Galaxy and Mass Assembly (GAMA) and Legacy Survey of Space and Time (LSST) surveys. This module supports tasks such as data conversion and trimming, specifically to make the integration of observational data into analysis workflows smoother and more efficient. As new data from the Vera Rubin Observatory’s LSST become available, additional routines will be developed to work with this type of data. Developed and maintained by Daniel Gallego, Lianys Feliciano, and Charlotte Olsen. 

This subpackage includes:

    1. GAMA:
        - The GAMA subdirectory contains tutorials and tools designed for processing data from the Galaxy And Mass Assembly (GAMA) survey. Current resources focus on preparing GAMA data for further analysis by converting coordinates and trimming data to relevant sections.
        - Core Functions:
            - gama_functions.py: Contains functions for processing GAMA data, including utilities for coordinate conversions and data trimming.
        - Tutorials:
            - Convert_to_Euclidean_Coordinates.ipynb: This tutorial provides a guide for converting GAMA survey data into Euclidean coordinates, a common requirement for spatial analyses and data integration.
            - Trimming_Data.ipynb: This tutorial explains how to trim GAMA data to isolate specific regions or features of interest, ensuring efficient use of data in analysis workflows.

    2. LSST:
        - The LSST subdirectory is intended to support future work with data from the Vera Rubin Observatory’s Legacy Survey of Space and Time (LSST). While this section is currently under development, planned resources will focus on handling, analyzing, and visualizing LSST data within the context of cosmic web and galaxy evolution studies.
    
    