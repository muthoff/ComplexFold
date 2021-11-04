## Get ComplexFold
git clone https://github.com/muthoff/ComplexFold
cd ComplexFold

## Get Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
# install in home and let initialize
conda update conda
conda install pip
echo "conda deactivate" >> ~/.bashrc
rm -r Anaconda3-2021.05-Linux-x86_64.sh


## Setup Anaconda
conda create -y --name af2 python==3.8
conda activate af2

conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 pdbfixer==1.7 numba
conda install -y -c bioconda hmmer==3.3.2 hhsuite kalign2

pip3 install -r requirements.txt
#pip3 install -r /usr/local/science/ComplexFold/requirements.txt
pip3 install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Update openmm 
CFDIR=$PWD
PYTHON=$(which python)
cd $(dirname $(dirname $PYTHON))/lib/python3.8/site-packages
patch -p0 < $CFDIR/docker/openmm.patch