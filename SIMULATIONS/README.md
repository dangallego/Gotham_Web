# Simulation Tools

**Developers**: Lianys Feliciano, Janvi Madhani, Sneha Nair. 

This module includes tools and methods for working with simulation data, specifically for preparing and analyzing 3D data cubes. The tools are intended to streamline workflows in processing and interpreting data from simulations. This module includes tutorials and scripts aimed at guiding users through common simulation-related tasks, such as creating 3D grids and preparing data for DisPerSE.

The contents of this module include:

1. Core Package: This contains scripts and modules for performing various operations on simulation data. Key files include:

    - cube_class.py: Contains the Cube class, which provides methods for handling and processing 3D data cubes, such as loading, slicing, and saving data in formats compatible with DisPerSE and other tools.
    - cubes.py: A module with functions that operate on 3D cubes, including functions to prepare and manipulate data cubes.
    - treebricks.py: Provides utilities for handling hierarchical data structures, or "tree bricks," in simulations, useful for organizing and indexing cube data.

2. Tutorials: These notebooks illustrate practical uses of the SIMULATIONS package tools, demonstrating step-by-step processes for working with 3D data cubes and clusters.

    - Cubes_Tutorial.ipynb: This tutorial provides an overview of the Cube class and demonstrates useful operations such as loading and manipulating data cubes.
    - Intro_To_Clusters.ipynb: An introduction to working with clusters in simulation data, including identifying and analyzing cluster properties within data cubes.
    - Preparing_Cubes_For_Disperse.ipynb: A guide on formatting and preparing data cubes for use with DisPerSE, including instructions for setting up and verifying compatibility.

3. Shell Scripts: These template scripts show how to execute batch processing and job submission on supercomputers / clusters. 

    - get_fillaments.sh: A script for retrieving filament data from simulations, optimizing workflows by automating data retrieval.
    - submit_amr2cube.sh: A script to submit jobs for converting Adaptive Mesh Refinement (AMR) data into 3D cubes, intended for cluster environments.
    - jupyter-i.sh: This script supports the Jupyter notebook environment setup, ensuring smooth integration with cluster resources.

4. Sample Data: Example files to test and verify functionality within the package. These files provide initial data setups for users to work through tutorials and explore package capabilities.

    - sample_smoothed_cube.fr: A sample smoothed data cube file used as a reference in tutorials or testing workflows.
    - gascube_density_lim03_07_lma8.dat: A sample dataset containing gas density values over a 3D grid, ideal for working through density mapping and manipulation tasks.