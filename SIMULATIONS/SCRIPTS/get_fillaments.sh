#
#!/bin/sh
#PBS -S /bin/sh
#PBS -N get_net
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=12:00:00


get_net -inp smooth_sys_.dat -tsi 680 ‘output folder’ 

mse snap_680_gasgrid.NDnet -outName system_2_filaments -periodicity 000 -nsig 3 -forceLoops -ppairs_ASCII -upSkl

skelconv system_2_filaments_s3.up.NDskl -smooth 5 -to NDskl_ascii