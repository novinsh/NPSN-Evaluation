## NPSN Evaluation with Energy score

This repository provides code for the evaluation of the models from the [NPSN paper](https://inhwanbae.github.io/publication/npsn/) with energy score. <br>
The NPSN original repository: https://github.com/InhwanBae/NPSN

**Motivation**: Multimodal Trajectory Predictions are often evaluated with "Minimum over N" (MoN) metrics. 
Many of the trajectory prediction models provide probabilistic outputs that should be evaluated with metrics that are appropriate for the evaluation of distributions.
We propose Energy Score as a strictly proper scoring rule to be used for evaluation of trajectory distribution evaluation.

By this evaluation, we wanted to show how differently Energy Score ranks models from the NPSN paper than the minFDE/minADE.
PS: minFDE/minADE are commonly used instances of the MoN family of metrics.



Our modifications to calculate energy score as an evaluation metric:
- `baselines/pecnet/bridge.py`
- `baselines/pecnet/utils.py`
- `npsn/__initi__.py`
- `npsn/utils.py`


New files added:
- `test_npsn_w_es.py`: a duplicate of `test_npsn.py` that integrates calculation of the energy score and saves the results under `results/` as `.pkl`
- `results/analyze_results.py`: compiles tables on the output of the `test_npsn_w_es.py`
- `test_es_implemenations.py`: we test our three implementations of energy score in `numba`, `pytorch` and `scipy` against `dcor` package. We use `scipy` implementation for pecnet and `pytorch` for npsn.
