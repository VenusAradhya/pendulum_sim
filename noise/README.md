# noise

## About

Data and code for seismic noise and sensor noise. Data for the former is in the file `2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv`, while data for the latter is in the file `aosem_noise.csv`.  Code to calculate the basics is in `asd_tools.py` - see the relevant docstrings and type hints. See `inspect_asd_tools.py` for example usage.

## Initial Usage

For initial, perhaps all, usage in this project it is recommended to set the keyword arguments `deterministic` to `True`, and `z_score` to `0` - both within the function `asd_from_asd_statistics`. This ensures that you will get the mean noise level on every run. Without this the noise level will, intentionally, fluctuate - this is an advanced feature for later training using a PPSD like method.