#!/bin/sh
#PBS -S /bin/sh
#PBS -N jupyter_notebook
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=05:00:00
#PBS -e error.txt
#PBS -o output.txt


module () {
  eval $(/usr/bin/modulecmd bash $*)
}

export OMP_NUM_THREADS=64

module purge
module load intelpython/3-2022.2.1

# the directory you want to run your notebook from
cd /your/directory

# get tunneling info. Choose a random port
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="infinity01"
port_id="6000"

#conda init bash

# print tunneling instructions jupyter-log

echo -e "
#before submitting script, setup a password:
$ jupyter notebook --generate-config
$ jupyter notebook password

#Once complete, submit the job to infinity:
$ qsub jupyter.sh
#When the job starts, take a look at the output.txt file to see the SSH tunnel command needed

$ nano output.txt
#On your local machine, open a new terminal window, and then run the example commands given below.

# Example Command to create SSH tunnel:
$ ping -c 1 -s 999 infinity01.iap.fr #ping because it's Infinity
$ ssh -N -f -L localhost:8888:${node}:${port_id} ${user}@${cluster}.iap.fr #SSH. 
# Use a browser on your local machine to go to:
http://localhost:8888/

#If you are using an annoying browser like Chrome on Mac, make sure the localhost port you choose is a standard, typically allowed port. 
#8888, like in this example, is a good bet!
#You may get errors like ERR_UNSAFE_PORT. In this case add your port number to the list of allowed ports with this command (example for port 6666):
$/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --explicitly-allowed-ports=6666
# It might not be the end of your plight, since you might now get  an ERR_CONNECTION_RESET error. In this case check your VPN, reset your TCP/IP s$
# clear your cache, disable your proxy and pray the gods of internet.
"


jupyter notebook --no-browser --ip=${node}  --port=${port_id}

# keep it alive if necessary
sleep 36000
