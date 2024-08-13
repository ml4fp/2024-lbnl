git clone https://github.com/Alexanders101/SPANet

cd SPANet
rm -rf data/full_hadronic_ttbar/; ln -s /global/cfs/cdirs/ntrain1/attention/full_hadronic_ttbar/ data/full_hadronic_ttbar

module load pytorch/2.3.1
pip install --user numba opt_einsum "pytorch-lightning==2.4"