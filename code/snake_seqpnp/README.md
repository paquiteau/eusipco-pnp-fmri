# Benchmark of PnP methods for functional MRI 

This benchmark uses [SNAKE-fMRI](https://github.com/paquiteau/snake-fmri) for simulating fMRI k-space data. Then we use the deepinv library to reconstruct data, and we evaluate the reconstruction performances using the toolkit of SNAKE-fMRI. 

# Code description

`sequentialpnp.py` provides a SNAKE-fMRI reconstructor that is able to reconstruct data provided by a `data_loader` from SNAKE-fMRI. 

# Experiments 
## Simulation 

``` sh
snake-acq --config-name="scenario2-pnp" 
```

## Reconstructrion 


## Results Analysis 

```sh
python results_analysis.py output-folder
```


