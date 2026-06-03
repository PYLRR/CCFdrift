from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
import os, tempfile
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime

from . import preprocessing

# @dataclass enables to inherit standard methods (such as __eq__) to handle the configuration parameters.
@dataclass
class CorrelationConfig:
    client: object  # any obspy client exposing get_waveforms(net, sta, loc, cha, t0, t1)
    network: str
    components: tuple[str, ...]
    location: str = "00"
    win_s: float = 3600.0
    overlap: float = 0.5
    target_fs: float = 50.0
    bp_fmin: float = 0.01
    bp_fmax: float = 10.0
    whiten_fmin: float = 0.05
    whiten_fmax: float = 0.5
    clip_n_sigma: float = 2.0
    max_gap_samples: int = 500
    max_lag_s: float = 800.0


def load_day(cfg: CorrelationConfig, station: str, channel: str,
             year: str, day: str) -> Trace | None:
    ''' Load the trace of a day.
    :param cfg: Configurations.
    :param station: Name of the station.
    :param channel: Name of the channel.
    :param year: Year at which data should be fetched.
    :param day: Day to be fetched as julian decimal day.
    :return: A trace containing the data of the given day.
    '''
    t0 = UTCDateTime(year=int(year), julday=int(day))
    try:
        st = cfg.client.get_waveforms(cfg.network, station, cfg.location,
                                      channel, t0, t0 + 86400.0)
    except Exception:
        return None
    if len(st) == 0:
        return None
    st.merge(fill_value=None)  # mask gaps for later filling
    return st[0]


def hourly_ccf(tr_a: Trace, tr_b: Trace, max_lag_s: float
               ) -> tuple[np.ndarray | None, np.ndarray | None]:
    ''' Perform the correlation of two one-hour traces using a multiplication in frequency domain.
    :param tr_a: First trace of the correlation.
    :param tr_b: Second trace of the correlation.
    :param max_lag_s: Maximum allowed lag time for the correlation.
    :return: The lags associated with the corresponding correlation values.
    '''
    # correlation by convolution (multiplication in the frequency domain)
    if tr_a.stats.sampling_rate != tr_b.stats.sampling_rate:
        print("Incompatible sampling rate; skipping CCF computation")
        return None, None
    fs = tr_a.stats.sampling_rate
    a = np.asarray(tr_a.data, dtype=np.float32)
    b = np.asarray(tr_b.data, dtype=np.float32)
    n = min(len(a), len(b))
    if n == 0:
        print("Empty data; skipping CCF computation")
        return None, None
    a = a[:n]
    b = b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        print("Zero-value data; skipping CCF computation")
        return None, None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    c_full = np.fft.irfft(A * np.conj(B), n=nfft) / (na * nb)
    maxlag_pts = int(round(max_lag_s * fs))
    pos = c_full[: maxlag_pts + 1]
    neg = c_full[-maxlag_pts:]
    c = np.concatenate((neg, pos)).astype(np.float32)
    lags = np.arange(-maxlag_pts, maxlag_pts + 1) / fs
    return lags, c


def daily_ccf_one_component_pair(cfg: CorrelationConfig,
                                 tr_a_day: Trace, tr_b_day: Trace
                                 ) -> tuple[np.ndarray | None,
                                            np.ndarray | None, int]:
    ''' Preprocess two one-day traces and correlate them on one-hour chunks.
    :param cfg: Configurations.
    :param tr_a_day: The first trace to correlate.
    :param tr_b_day: The second trace to correlate.
    :return: A triplet (lags, cc, n) where lags is the lags of the correlation, cc the associated correlations averaged
    on the whole day and n the number of one-hour chunk effectively used (chunks with large gaps being discarded).
    '''
    win_a = preprocessing.split_into_hourly_windows(tr_a_day, cfg.win_s, cfg.overlap)
    win_b = preprocessing.split_into_hourly_windows(tr_b_day, cfg.win_s, cfg.overlap)
    n_pairs = min(len(win_a), len(win_b))
    if n_pairs == 0:
        return None, None, 0
    stack = None
    nused = 0  # number of hourly windows actually used
    lags_out = None
    for i in range(n_pairs):
        wa = preprocessing.preprocess_window(
            win_a[i],
            max_gap_samples=cfg.max_gap_samples,
            bp_fmin=cfg.bp_fmin, bp_fmax=cfg.bp_fmax,
            target_fs=cfg.target_fs,
            clip_n_sigma=cfg.clip_n_sigma,
            whiten_fmin=cfg.whiten_fmin, whiten_fmax=cfg.whiten_fmax,
        )
        wb = preprocessing.preprocess_window(
            win_b[i],
            max_gap_samples=cfg.max_gap_samples,
            bp_fmin=cfg.bp_fmin, bp_fmax=cfg.bp_fmax,
            target_fs=cfg.target_fs,
            clip_n_sigma=cfg.clip_n_sigma,
            whiten_fmin=cfg.whiten_fmin, whiten_fmax=cfg.whiten_fmax,
        )
        if wa is None or wb is None:
            print("Empty window found; skipping window")
            continue
        s = max(wa.stats.starttime, wb.stats.starttime)
        e = min(wa.stats.endtime, wb.stats.endtime)
        if e - s < 0.5 * cfg.win_s:
            continue
        wa.trim(s, e, pad=False)
        wb.trim(s, e, pad=False)
        lags, c = hourly_ccf(wa, wb, cfg.max_lag_s)
        if c is None:
            print("CCF yielded None; skipping window")
            continue
        if stack is None:
            stack = np.zeros_like(c)
            lags_out = lags
        if len(c) != len(stack):
            print("Wrong CCF output size; skipping window")
            continue
        stack += c
        nused += 1
    if nused == 0:
        return None, None, 0
    return lags_out, (stack / nused).astype(np.float32), nused


def daily_ccf_all_components(cfg: CorrelationConfig,
                             station_a: str, station_b: str,
                             year: str, day: str) -> dict:
    ''' Given two stations, perform the correlations between all their channels for a given day.
    :param cfg: Configurations.
    :param station_a: Name of the first station whose data should be correlated.
    :param station_b: Name of the second station whose data should be correlated.
    :param year: Year for which data should be correlated.
    :param day: Julian decimal day of the year for which data should be correlated.
    :return: A dict giving the stations used, the date, the lags, the associated correlation values averaged over the
    day and the number of chunks used for correlations.
    '''
    out = {"station_a": station_a, "station_b": station_b,
           "year": year, "day": day,
           "lags": None, "ccfs": {}, "nwin": {}}
    traces_a = {ch: load_day(cfg, station_a, ch, year, day) for ch in cfg.components}
    traces_b = {ch: load_day(cfg, station_b, ch, year, day) for ch in cfg.components}
    for cha in cfg.components:
        if traces_a[cha] is None:
            continue
        for chb in cfg.components:
            if traces_b[chb] is None:
                continue
            key = f"{cha}-{chb}"
            lags, ccf, n = daily_ccf_one_component_pair(
                cfg, traces_a[cha], traces_b[chb]
            )
            if ccf is None:
                continue
            if out["lags"] is None:
                out["lags"] = lags
            out["ccfs"][key] = ccf
            out["nwin"][key] = n
    return out

# used as a copy of configurations for each process. The initialization is done in _init_worker.
# when a process is created, this is empty.
_WORKER_CFG: CorrelationConfig | None = None

def _init_worker(cfg: CorrelationConfig) -> None:
    ''' Gives the configurations to a given worker.
    :param cfg: Configurations to copy.
    :return: None
    '''
    global _WORKER_CFG  # note: global variables are only known by their process
    _WORKER_CFG = cfg


def _day_range(start, end, stride: int = 1) -> list[tuple[str, str]]:
    ''' Given two dates, gives the list of days between them. A stride may be given to reduce computation time.
    :param start: Start date. 
    :param end: End date.
    :param stride: Stride used in the list of days. This enables to reduced the number of days to process by evenly 
    discarding days, which may be useful to reduce computation time at the cost of accuracy.
    :return: A list of days between the given dates.
    '''
    t0 = UTCDateTime(start)
    t1 = UTCDateTime(end)
    days = []
    t = UTCDateTime(year=t0.year, julday=t0.julday)
    while t < t1:
        days.append((f"{t.year:04d}", f"{t.julday:03d}"))
        t = t + 86400.0
    return days[:: max(int(stride), 1)]


def _ccf_worker(job: tuple) -> tuple:
    ''' Assign the correlation of a given day between two stations to a worker.
    :param job: A tuple (test, ref, year, day, out_path) giving the two stations, the date and the path where the
    result should be saved.
    :return: A tuple (status, stations, year, day, n) where status may be "skip" if output file already exists,
    "err" if an exception occurred, void if no correlation was done, and ok if everything worked. n is then number of
    chunks used for correlations.
    '''
    test, ref, year, day, out_path = job
    cfg = _WORKER_CFG
    out = Path(out_path)
    if out.exists():
        return ("skip", f"{test}_{ref}", year, day, None)
    try:
        res = daily_ccf_all_components(cfg, test, ref, year, day)
    except Exception as exc:
        return ("err", f"{test}_{ref}", year, day, f"{type(exc).__name__}: {exc}")
    if not res["ccfs"]:
        return ("void", f"{test}_{ref}", year, day, None)
    keys = sorted(res["ccfs"].keys())
    arr = np.stack([res["ccfs"][k] for k in keys], axis=0)
    nwin = np.array([res["nwin"][k] for k in keys], dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(out.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(
                fh,
                lags=res["lags"].astype(np.float32),
                ccfs=arr.astype(np.float32),
                component_pairs=np.array(keys),
                nwin=nwin,
                date=f"{year}-{day}",
                station_a=test, station_b=ref,
            )
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ("ok", f"{test}_{ref}", year, day, len(keys))


def _print_progress(i: int, n: int, r: tuple, t_start: float) -> None:
    ''' Print the progress of the running CCF process.
    :param i: Current step.
    :param n: Total number of steps to process.
    :param r: A tuple (status, pair, year, day, info), see output of :func:`~_ccf_worker`.
    :param t_start: Time of computation start.
    :return: None.
    '''
    status, pair, year, day, info = r
    elapsed = time.time() - t_start
    eta = elapsed / i * (n - i)
    tail = f"npairs={info}" if status == "ok" else (info or "")
    print(f"[{i:5d}/{n}] {status.upper():4s} {pair} {year}.{day} {tail}"
          f"   (elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min)")


def compute_ccf(client, output_dir, pairs, components, network, start, end,
                location: str = "00", n_workers: int = 1,
                win_s: float = 3600.0, overlap: float = 0.5,
                target_fs: float = 50.0,
                bp_fmin: float = 0.01, bp_fmax: float = 10.0,
                whiten_fmin: float = 0.05, whiten_fmax: float = 0.5,
                clip_n_sigma: float = 2.0, max_gap_samples: int = 500,
                max_lag_s: float = 800.0,
                day_stride: int = 1, skip_existing: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    ''' Run the CCF for a whole dataset.
    :param client: An ObsPy client object enabling to handle various data structures (tsuch as SDS).
    :param output_dir: Output directory where CCF results should be written.
    :param pairs: Pairs of stations to correlate.
    :param components: Channels used for correlation.
    :param network: Name of the network of stations.
    :param start: Start time of the correlation.
    (note: stations whose record start later will simply be ignored until their recording start date)
    :param end: End time of the correlation.
    (note: stations whose record end earlier will simply be ignored starting from their recording end date)
    :param location: Location of the network. May be required for some data structure.
    :param n_workers: Number of processes to use. Default is single-process.
    :param win_s: Size of chunks used for correlation.
    :param overlap: Overlap between consecutive chunks, as a fraction of chunk size.
    :param target_fs: Sampling frequency used for correlation.
    :param bp_fmin: Minimum frequency of the bandpass filter.
    :param bp_fmax: Maximum frequency of the bandpass filter.
    :param whiten_fmin: Minimum frequency of the whitening filter.
    :param whiten_fmax: Maximum frequency of the whitening filter.
    :param clip_n_sigma: Multiplier of the signal standard deviation used as a clipping threshold to reduce the
    importance of transient events such as earthquakes.
    :param max_gap_samples: Maximum allowed size of a gap in data. A too large gap will cause the chunk to be discarded.
    :param max_lag_s: Maximum lag allowed when correlating two chunks.
    :param day_stride: Step between days used for correlation. Leave to one unless a small computation time is required.
    :param skip_existing: Skip files that were already written. Default to True.
    :param verbose: If verbose, print various information about the process. Default to True.
    :return: None.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CorrelationConfig(
        client=client, network=network,
        components=tuple(components), location=location,
        win_s=win_s, overlap=overlap, target_fs=target_fs,
        bp_fmin=bp_fmin, bp_fmax=bp_fmax,
        whiten_fmin=whiten_fmin, whiten_fmax=whiten_fmax,
        clip_n_sigma=clip_n_sigma, max_gap_samples=max_gap_samples,
        max_lag_s=max_lag_s,
    )

    pairs = [tuple(p) for p in pairs]
    days = _day_range(start, end, day_stride)

    jobs = []
    for test, ref in pairs:
        pdir = output_dir / f"{test}_{ref}"
        for year, day in days:
            out = pdir / f"ccf_{year}_{day}.npz"
            if skip_existing and out.exists():
                continue
            jobs.append((test, ref, year, day, str(out)))

    if verbose:
        print(f"Pairs: {len(pairs)} | days: {len(days)} | jobs to run: {len(jobs)}")

    counts = {"ok": 0, "skip": 0, "void": 0, "err": 0}
    if jobs:
        t_start = time.time()
        if n_workers and int(n_workers) > 1:
            with Pool(int(n_workers), initializer=_init_worker,
                      initargs=(cfg,)) as pool:
                for i, r in enumerate(pool.imap_unordered(_ccf_worker, jobs), 1):
                    counts[r[0]] += 1
                    if verbose:
                        _print_progress(i, len(jobs), r, t_start)
        else:
            _init_worker(cfg)
            for i, job in enumerate(jobs, 1):
                r = _ccf_worker(job)
                counts[r[0]] += 1
                if verbose:
                    _print_progress(i, len(jobs), r, t_start)
        if verbose:
            print(f"\nDone in {(time.time() - t_start) / 60:.1f} min | "
                  f"ok={counts['ok']} skip={counts['skip']} "
                  f"void={counts['void']} err={counts['err']}")