"""Deterministic synthetic providers for time-varying features.

These simulate APIs we don't have serving access to (GEOS-FP meteorology,
MODIS AOD). Values are deterministic functions of (location, timestamp) so
identical requests always produce identical predictions, and are calibrated
to the Delhi training distribution (2021-2025 station data):

    T_celsius 13..44, RH_pct 4..67, WS_ms 0.2..10.5, SLP_kpa 99.3..102.1,
    AOD_550 0..3.2, sun_zenith 21..57, sun_azimuth 99..158.

Sun geometry is real astronomy, computed at the Landsat overpass time
(10:30 IST) for the request date — matching how the model was trained.
"""

import numpy as np

# Monthly climatology from the training CSV (index 0 = January)
T_MONTHLY = np.array([18.2, 22.7, 28.2, 34.3, 37.6, 38.7, 37.7, 32.6, 32.3, 29.9, 24.3, 19.4])
RH_MONTHLY = np.array([30.5, 25.2, 23.1, 13.6, 18.1, 21.8, 33.2, 52.8, 45.0, 37.1, 34.1, 27.8])
WS_MONTHLY = np.array([5.0, 5.0, 4.8, 5.0, 3.7, 4.4, 4.6, 4.4, 4.3, 2.8, 3.1, 4.5])
SLP_MONTHLY = np.array([101.63, 101.52, 101.30, 100.73, 100.15, 100.01, 99.83, 100.21, 100.50, 101.07, 101.63, 101.73])
AOD_MONTHLY = np.array([0.44, 0.39, 0.46, 0.41, 0.63, 0.71, 0.69, 0.48, 0.46, 0.68, 1.05, 0.49])


def hash01(*arrays):
    """Deterministic pseudo-random floats in [0, 1) from numeric arrays."""
    acc = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for i, a in enumerate(arrays):
        acc = acc + np.asarray(a, dtype=np.float64) * (12.9898 + 17.17 * i)
    return np.modf(np.abs(np.sin(acc) * 43758.5453123))[0]


def solar_position(lat, lon, ts):
    """Sun azimuth/zenith (degrees) at 10:30 IST on ts's date — NOAA approximation.

    Trained features were extracted at Landsat overpass (~10:30 local), so we
    pin the hour and vary only with date and location.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    doy = ts.timetuple().tm_yday
    frac_year = 2.0 * np.pi / 365.0 * (doy - 1 + (10.5 - 12) / 24.0)

    decl = (
        0.006918 - 0.399912 * np.cos(frac_year) + 0.070257 * np.sin(frac_year)
        - 0.006758 * np.cos(2 * frac_year) + 0.000907 * np.sin(2 * frac_year)
        - 0.002697 * np.cos(3 * frac_year) + 0.00148 * np.sin(3 * frac_year)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(frac_year) - 0.032077 * np.sin(frac_year)
        - 0.014615 * np.cos(2 * frac_year) - 0.040849 * np.sin(2 * frac_year)
    )

    # 10:30 IST == 10.5h at UTC+5.5; true solar time depends on longitude
    tst = 10.5 * 60 + eqtime + 4 * lon - 60 * 5.5
    ha = np.radians(tst / 4.0 - 180.0)

    lat_r = np.radians(lat)
    cos_zen = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    zenith = np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))

    sin_zen = np.sin(np.radians(zenith))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = (np.sin(decl) - np.sin(lat_r) * cos_zen) / (np.cos(lat_r) * sin_zen)
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    azimuth = np.where(ha > 0, 360.0 - az, az)
    return azimuth, zenith


def _month_interp(table, ts):
    """Smoothly interpolate a monthly climatology table to ts's day of year."""
    doy = ts.timetuple().tm_yday
    pos = (doy - 15.0) / 30.44 % 12
    lo = int(np.floor(pos)) % 12
    hi = (lo + 1) % 12
    w = pos - np.floor(pos)
    return table[lo] * (1 - w) + table[hi] * w


def meteo(seed_idx, lat, lon, ts):
    """Synthetic GEOS-FP-like meteorology: seasonal + diurnal + spatial texture."""
    seed_idx = np.asarray(seed_idx, dtype=np.float64)
    unix_hour = np.float64(ts.timestamp() // 3600)
    hour_local = (ts.hour + ts.minute / 60.0 + 5.5) % 24  # IST

    diurnal = np.cos(2 * np.pi * (hour_local - 14.0) / 24.0)  # peaks at 2pm
    t_base = _month_interp(T_MONTHLY, ts)
    t = t_base + 4.5 * diurnal + 2.0 * (hash01(seed_idx, unix_hour) - 0.5)

    rh_base = _month_interp(RH_MONTHLY, ts)
    rh = rh_base - 8.0 * diurnal + 6.0 * (hash01(seed_idx, unix_hour, 2) - 0.5)
    rh = np.clip(rh, 4.0, 67.0)

    ws_base = _month_interp(WS_MONTHLY, ts)
    ws = ws_base * (0.7 + 0.6 * hash01(seed_idx, unix_hour, 3)) + 1.0 * diurnal
    ws = np.clip(ws, 0.2, 10.5)

    slp_base = _month_interp(SLP_MONTHLY, ts)
    slp = slp_base + 0.3 * (hash01(np.zeros_like(seed_idx), unix_hour, 4) - 0.5)
    slp = np.clip(slp, 99.3, 102.1)

    return ws, np.clip(t, 13.0, 44.0), slp, rh


def aod(seed_idx, lat, ts):
    """Synthetic MODIS-like AOD: regional daily value + mild spatial gradient."""
    seed_idx = np.asarray(seed_idx, dtype=np.float64)
    day = np.float64(ts.timestamp() // 86400)

    base = _month_interp(AOD_MONTHLY, ts)
    daily = base * (0.6 + 0.8 * float(hash01(np.array([day]), 5)[0]))
    spatial = 1.0 + 0.25 * (hash01(seed_idx, day) - 0.5)
    aod550 = np.clip(daily * spatial, 0.02, 3.16)

    aod470 = aod550 * 1.27  # mean training ratio AOD_470/AOD_550
    unc = np.clip(0.02 + 0.02 * hash01(seed_idx, day, 6), 0.003, 0.35)
    available = np.full_like(seed_idx, 1.0 if float(hash01(np.array([day]), 7)[0]) < 0.87 else 0.0)
    return aod550, aod470, unc, available
