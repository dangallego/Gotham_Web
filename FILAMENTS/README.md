## FILAMENT TOOLS

This module contains useful methods for working with filament data, in particular after having retrieved filament skeleton files from DisPerSE. These routines were developed by Lianys Feliciano, Daniel Gallego and Janvi Madhani. The list of available tutorials are: 

1. Extracting_Filaments: short lesson on how to extract filaments using the Infinity cluster (ie. the commands and which scripts to use), as well as how to read in the (.NDskl) skeleton files into Jupyter Notebooks. 

2. Fil_Tutorial: a tutorial of the *filament.py* package and the methods included within the Filament class. In regards to the class, this tutorial provides examples of: 

    a. Importing the class. 
    b. Utilizing import filaments method (*Filament.import_fils*).
    c. Working with filament dictionary data structure (what is returned initially from reading in filament skeletons).
    d. Obtaining critical points from the dictionary.
    e. Representing filament coordinates in a tabular way (using Pandas).
    f. Representing filament coordinates (tabular) as segment pairs, ie. as the coordinates that make up the segments of each filament.
    g. Obtaining filament lengths.
    h. Obtaining filament densities {currently WIP - bug in the code}.

3. Test-Lengths: in-depth and step-by-step guide for how the filament lengths are calculated in the Filament class methods. 

    