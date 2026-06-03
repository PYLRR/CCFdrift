# CCFdrift

Clock drift of seismic stations estimated from ambient-noise cross-correlation
functions (CCFs) using the work from [Halbe et al. 2018](https://academic.oup.com/gji/article/214/3/2014/5038378). 
Built on [ObsPy](https://docs.obspy.org/), ir order to make it agnostic
to how the data is organized and juste use ObsPy client (SDS, ...).

Two steps:
1. `compute_ccf(...)` — daily CCFs for every station pair and every
   channel pair, written as one `.npz` per (pair, day). Store all in the specified directory.
2. `relative_drift(...)` — per-station drift relative to a chosen reference,
   with a consistency check and weighted least-squares
   network-wise clock drift estimations.

## Install

```bash
pip install git+https://github.com/PYLRR/CCFdrift.git  # to install online
```

## Compute the CCFs

```python
from itertools import combinations
from obspy.clients.filesystem.sds import Client
import CCFdrift

client = Client("/path/to/SDS_archive")
stations = ["staA", "staB", "staC"]

CCFdrift.compute_ccf(
    client=client,
    output_dir="/path/to/CCF",
    pairs=list(combinations(stations, 2)),
    components=("HDH", "EH1", "EH2", "EH3"),
    network="XX",
    start="2020-01-01",
    end="2021-06-01",
    n_workers=10,
)
```

Each (pair, day) is one file, making it possible to interrupt the process at any time. At the next run, elements already
processed will simpy be skipped.

## Estimate the relative drift

```python
import pandas as pd
import CCFdrift

res = CCFdrift.relative_drift(
    ccf_dir="/path/to/CCF",
    stations=["staA", "staB", "staC"],
    reference="staA",
    fs=50.0,
    rcf_period=(pd.Timestamp("2020-01-05"), pd.Timestamp("2020-04-01")),
    output_dir="/path/to/DRIFT/relative",   # optional, to write a csv
)

print(res.network)    # drifts
print(res.closure)    # triangle closures (consistency check)
print(res.chi2_red)   # chi 2 of the fit (consistency check)
```

The intermediate steps are also exposed as standalone functions
(`measure_all_pairs`, `triangle_closure`, `network_adjustment`, ...), so the
pipeline can be re-created.

## Notes

- The obtained measures are relative: once fixed, stations are expected to be consistent together but
a drift with respect to absolute time (UTC) likely still exists.
- For stations A and B, CCF(A,B) or CCF(B,A) is computed but not both, assuming the process symmetric.
