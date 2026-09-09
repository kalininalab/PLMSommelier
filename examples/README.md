# Example data

`fluorescence_sample.csv` is a deterministic 500-row subsample (400 train /
100 valid, matching the original split proportions) of the **TAPE
fluorescence** benchmark -- avGFP variants and their log-fluorescence, from
Rao et al., *Evaluating Protein Transfer Learning with TAPE* (NeurIPS 2019),
itself built from the deep mutational scan in Sarkisyan et al., *Local
fitness landscape of the green fluorescent protein* (Nature, 2016). It's here
only so the README's quickstart runs against real data with no download --
it's not a benchmark-scale dataset, and results from it aren't comparable to
the paper's or TAPE's numbers.

It exists purely so `plmsommelier suggest examples/fluorescence_sample.csv
facebook/esm2_t6_8M_UR50D --task regression` runs in about a minute on a CPU.
Use your own data (see the main README's "Input format" section) for
anything beyond a smoke test.

Regenerated with a fixed `random_state=42` subsample of the full
`fluorescence.csv` split used in this project's own benchmarking; see
`bench/prepare_data.py` (not shipped in the package) if you want the full
26.8k-row version.
