#
#!/bin/sh
#PBS -S /bin/sh
#PBS -N amr2cube
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=48:00:00

module () {
  eval $(/usr/bin/modulecmd bash $*)
}

export OMP_NUM_THREADS=64
module purge
module load intelpython/3-2021.4.0
module load openmpi/4.1.5-amdaocc3.0.0

#cd /home/madhani/ramses-3.0/utils/f90
cd /your/directory


#Variables for you to change 
type= 1 #the types are as follows 1: 2: 3: 4: 5: 
min= 0.3
max= 0.7
levelmax= 10 # the maximum level is 21

./amr2cube -inp /data7b/NewHorizon/OUTPUT_DIR/output_00970 -out name -typ type -xmi min -xma max -ymi min -yma max -zmi min -zma max -lma  levelmax



