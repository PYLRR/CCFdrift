from __future__ import annotations

import numpy as np
from obspy import Stream, Trace


def split_into_hourly_windows(tr: Trace, win_s: float = 3600.0,
                              overlap: float = 0.5) -> list[Trace]:
    ''' Splits a trace in one-hour long chunks with an optional overlap.
    :param tr: Input trace.
    :param win_s: Chunk size in seconds, default being one hour.
    :param overlap: Overlap between two consecutive chunks, specified as a ratio in [0,1].
    :return: A list of chunks containing sub-traces.
    '''
    fs = tr.stats.sampling_rate
    npts_win = int(round(win_s * fs))
    step = int(round(npts_win * (1.0 - overlap)))
    starttime = tr.stats.starttime
    endtime = tr.stats.endtime
    out = []
    t0 = starttime
    while t0 + win_s <= endtime + 1e-9:
        sub = tr.slice(t0, t0 + win_s, nearest_sample=False)
        if sub.stats.npts >= npts_win - 1:  # check there is no gap
            out.append(sub)
        t0 = t0 + step / fs
    return out

def reject_or_fill_gaps(tr: Trace, max_gap_samples: int = 500) -> Trace | None:
    ''' Handle gaps in data, where gaps are marked as masked in the trace data.
    The gaps are filled with linear interpolated value if they are small enough, otherwise
    rejected.
    :param tr: Trace where gaps should be handled.
    :param max_gap_samples: Maximum allowed size for a gap to be filled. A larger gap will be rejected.
    :return: The trace with filled gaps if they all were small enough, None if a gap was rejected.
    '''
    data = np.asarray(tr.data, dtype=np.float64)
    # obspy trace merge naturally masks gaps; if already masked we use that otherwise we look for infinite values
    if not np.ma.isMaskedArray(data):
        nan_mask = ~np.isfinite(data)
        if not nan_mask.any():
            return tr
        mask = nan_mask
    else:
        mask = np.ma.getmaskarray(data)
        data = data.filled(np.nan)
    if not mask.any():
        return tr
    edges = np.diff(mask.astype(np.int8)) # edges = 1 at gap start, -1 at gap end and 0 otherwise
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts)) # window starts by a gap
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)])) # window ends by a gap
    for s, e in zip(starts, ends):
        if (e - s) > max_gap_samples:
            return None # gap too large for reasonable fix
    idx = np.arange(len(data))
    good = ~mask
    data[mask] = np.interp(idx[mask], idx[good], data[good]) # linear interp
    out = tr.copy()
    out.data = data.astype(np.float32)
    return out


def demean_detrend(tr: Trace) -> Trace:
    ''' Remove bias and linear trend in the data.
    :param tr: The trace to demean and detrend.
    :return: The demeaned and detrended trace.
    '''
    out = tr.copy()
    out.detrend("demean")
    out.detrend("linear")
    return out


def bandpass_broad(tr: Trace, fmin: float = 0.01, fmax: float = 10.0,
                   corners: int = 4) -> Trace:
    ''' Apply a cosine taper and a Butterworth-bandpass filter.
    :param tr: Trace to filter.
    :param fmin: Minimum frequency of the filter.
    :param fmax: Maximum frequency of the filter.
    :param corners: Order of Butterworth filter.
    :return:The filtered trace.
    '''
    out = tr.copy()
    nyq = 0.5 * out.stats.sampling_rate
    fmax_eff = min(fmax, 0.95 * nyq)
    out.taper(max_percentage=0.005, type="cosine")
    # zerophase enables to counter the dephasing caused by the filter.
    out.filter("bandpass", freqmin=fmin, freqmax=fmax_eff,
               corners=corners, zerophase=True)
    return out


def decimate_to(tr: Trace, target_fs: float = 50.0) -> Trace:
    ''' Decimate the signal as much as possible before resampling it to a wanted frequency.
    :param tr: The trace to decimate and resample.
    :param target_fs: The wanted sampling frequency.
    :return: The trace resampled at the wanted frequency.
    '''
    out = tr.copy()
    while out.stats.sampling_rate > target_fs + 1e-6: # iteratively decimate by up to a factor of 8
        ratio_int = int(round(out.stats.sampling_rate / target_fs))
        if ratio_int <= 1:  # no upsampling
            break
        step = ratio_int
        if step > 8:
            for d in (8, 7, 6, 5, 4, 3, 2):
                if ratio_int % d == 0:
                    step = d
                    break
            else: # executed if the ratio is greater than 8
                step = 8
        out.decimate(step, no_filter=False, strict_length=False)
    if abs(out.stats.sampling_rate - target_fs) > 1e-3: # resample to get the exact expected sampling rate
        out.resample(target_fs)
    return out


def amplitude_clip(tr: Trace, n_sigma: float = 2.0) -> Trace:
    ''' Clip a trace at a multiple of its standard deviation. This aims at reducing the importance of large transient
    signals such as earthquakes, given the objective is to correlate noise.
    :param tr: Trace to clip. The trace should be centered and detrended.
    :param n_sigma: The multiplier of the standard deviation used as a clip threshold.
    :return: The clipped trace.
    '''
    out = tr.copy()
    x = np.asarray(out.data, dtype=np.float64)
    s = x.std()
    if s > 0:
        np.clip(x, -n_sigma * s, n_sigma * s, out=x)
    out.data = x.astype(out.data.dtype, copy=False)
    return out


def spectral_whiten(tr: Trace, fmin: float = 0.05, fmax: float = 0.5,
                    pad: float = 0.01) -> Trace:
    ''' Whiten the signal, normalizing it frequency-wise in the wanted band. This is used to make all frequencies
    equally important at the correlation step.
    :param tr: Trace to whiten.
    :param fmin: Minimum frequency where whitening should be applied.
    :param fmax: Maximum frequency where whitening should be applied.
    :param pad: Size of a trapezoid-like taper before and after the whitening to make boundaries smoother, in Hz.
    :return: The whitened trace.
    '''
    out = tr.copy()
    x = np.asarray(out.data, dtype=np.float64)
    n = len(x)
    fs = out.stats.sampling_rate
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X)
    mag[mag == 0] = 1.0
    Y = X / mag
    # now apply a taper around the wanted frequencies
    band = np.zeros_like(f)
    f1, f2 = fmin - pad, fmin
    f3, f4 = fmax, fmax + pad
    rise = (f >= f1) & (f < f2)
    flat = (f >= f2) & (f <= f3)
    fall = (f > f3) & (f <= f4)
    if (f2 - f1) > 0:
        band[rise] = 0.5 * (1 - np.cos(np.pi * (f[rise] - f1) / (f2 - f1)))
    band[flat] = 1.0
    if (f4 - f3) > 0:
        band[fall] = 0.5 * (1 + np.cos(np.pi * (f[fall] - f3) / (f4 - f3)))
    Y *= band
    y = np.fft.irfft(Y, n=n)
    out.data = y.astype(out.data.dtype, copy=False)
    return out


def one_bit_normalize(tr: Trace) -> Trace:
    ''' Apply one bit normalization to trace, clipping it to -1 and +1.
    :param tr: Trace to normalize.
    :return: The normalize trace containing only -1 and +1 values.
    '''
    out = tr.copy()
    out.data = np.sign(np.asarray(out.data)).astype(np.float32)
    return out


def preprocess_window(tr: Trace,
                      max_gap_samples: int = 500,
                      bp_fmin: float = 0.01, bp_fmax: float = 10.0,
                      target_fs: float = 50.0,
                      clip_n_sigma: float = 2.0,
                      whiten_fmin: float = 0.05, whiten_fmax: float = 0.5,
                      ) -> Trace | None:
    ''' Preprocess a particular trace so that it is suitable for noise correlation.
    :param tr: The trace to preprocess.
    :param max_gap_samples: The maximum size allowed for a gap in data for the trace not to be rejected.
    :param bp_fmin: Minimum frequency of the bandpass filter.
    :param bp_fmax: Maximum frequency of the bandpass filter.
    :param target_fs: Resampling frequency.
    :param clip_n_sigma: Multiplier of sigma used to clip the signal and reduce the importance of large transient events such
    as earthquakes.
    :param whiten_fmin: Minimum frequency where whitening should be applied.
    :param whiten_fmax: Maximum frequency where whitening should be applied.
    :return: The preprocessed trace.
    '''
    tr = reject_or_fill_gaps(tr, max_gap_samples=max_gap_samples)
    if tr is None:
        return None
    tr = demean_detrend(tr)
    tr = bandpass_broad(tr, fmin=bp_fmin, fmax=bp_fmax)
    tr = decimate_to(tr, target_fs=target_fs)
    tr = amplitude_clip(tr, n_sigma=clip_n_sigma)
    tr = spectral_whiten(tr, fmin=whiten_fmin, fmax=whiten_fmax)
    tr = one_bit_normalize(tr)
    return tr