# Data

This project uses the public Wang2016 / Tsinghua SSVEP benchmark through MOABB.

I do not commit the raw public `.mat` cache because it is large. Running `scripts/run_wang2016_expansion.py` downloads/loads the data through MOABB and writes the compact final results table to `results/wang2016_expanded_model_comparison.csv`.

My private OpenBCI/OpenVEP pilot recordings are not part of this public repository. I used them to build the original pipeline and shape the question, but the reproducible results reported here come from the public Wang2016 benchmark.
