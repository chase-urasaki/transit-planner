import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    get_sun,
)
import astropy.units as u

from datetime import timezone
from zoneinfo import ZoneInfo


# ============================================================
# EXOFOP ENDPOINTS
# ============================================================

EXOFOP_TOI_URL = (
    "https://exofop.ipac.caltech.edu/tess/"
    "download_toi.php?sort=toi&output=csv"
)

EXOFOP_TARGET_URL = (
    "https://exofop.ipac.caltech.edu/tess/"
    "target.php?id={tic}&json"
)


# ============================================================
# OBSERVATORY
# ============================================================

MAUNA_KEA = EarthLocation(
    lat=19.8206 * u.deg,
    lon=-155.4681 * u.deg,
    height=4205 * u.m,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _parse_target(target):
    """
    Parse TOI/TIC identifier.

    Examples
    --------
    "TOI 2048.01"  -> ("TOI", 2048.01)
    2048.01        -> ("TOI", 2048.01)

    "TIC 139474683" -> ("TIC", 139474683)
    139474683       -> ("TIC", 139474683)

    Bare integers are interpreted as TIC IDs.
    """

    if isinstance(target, str):

        s = target.strip().upper()

        if s.startswith("TOI"):

            value = (
                s.replace("TOI", "")
                 .replace("-", "")
                 .strip()
            )

            return "TOI", float(value)

        if s.startswith("TIC"):

            value = (
                s.replace("TIC", "")
                 .replace("-", "")
                 .strip()
            )

            # tolerate accidental TIC xxxxx.01
            value = value.split(".")[0]

            return "TIC", int(value)

        if "." in s:
            return "TOI", float(s)

        return "TIC", int(s)

    if isinstance(target, (float, np.floating)):
        if not float(target).is_integer():
            return "TOI", float(target)

    return "TIC", int(target)


def _find_column(df, possible_names):
    """Find a dataframe column flexibly."""

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # Exact match
    for name in possible_names:

        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    # Substring fallback
    for name in possible_names:

        key = name.strip().lower()

        for col in df.columns:

            if key in str(col).lower():
                return col

    raise KeyError(
        f"Could not find a column matching {possible_names}.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def _safe_float(value):
    """Convert a value to float, otherwise return NaN."""

    if value is None:
        return np.nan

    try:

        value = str(value).strip()

        if value == "":
            return np.nan

        return float(value)

    except (ValueError, TypeError):
        return np.nan


# ============================================================
# EXOFOP TIC JSON
# ============================================================

def get_exofop_tic_json(tic):
    """
    Download ExoFOP target JSON for a TIC.

    Does not require the target to have a TOI.
    """

    tic = int(tic)

    url = EXOFOP_TARGET_URL.format(
        tic=tic
    )

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    try:
        return response.json()

    except Exception as exc:

        raise ValueError(
            f"ExoFOP returned a response for TIC {tic}, "
            "but it was not valid JSON."
        ) from exc


# ============================================================
# TIC COORDINATES
# ============================================================

def _find_ra_dec(data):
    """
    Recursively find an RA/Dec pair in ExoFOP target JSON.
    """

    candidates = []

    def walk(obj, path="root"):

        if isinstance(obj, dict):

            lower = {
                str(k).lower(): k
                for k in obj
            }

            ra_key = None
            dec_key = None

            for lk, original in lower.items():

                if lk in [
                    "ra",
                    "ra_deg",
                    "radeg",
                ]:
                    ra_key = original

                if lk in [
                    "dec",
                    "decl",
                    "dec_deg",
                    "decdeg",
                ]:
                    dec_key = original

            if (
                ra_key is not None
                and dec_key is not None
            ):

                candidates.append(
                    (
                        path,
                        obj[ra_key],
                        obj[dec_key],
                    )
                )

            for key, value in obj.items():

                walk(
                    value,
                    f"{path}.{key}",
                )

        elif isinstance(obj, list):

            for i, value in enumerate(obj):

                walk(
                    value,
                    f"{path}[{i}]",
                )

    walk(data)

    if len(candidates) == 0:

        raise ValueError(
            "Could not identify RA/Dec in ExoFOP TIC JSON."
        )

    # Prefer shallow target-level metadata
    candidates.sort(
        key=lambda x: len(x[0])
    )

    _, ra, dec = candidates[0]

    try:

        ra_num = float(ra)
        dec_num = float(dec)

        if (
            0 <= ra_num <= 360
            and -90 <= dec_num <= 90
        ):

            return SkyCoord(
                ra=ra_num * u.deg,
                dec=dec_num * u.deg,
            )

    except (ValueError, TypeError):
        pass

    return SkyCoord(
        str(ra),
        str(dec),
        unit=(u.hourangle, u.deg),
    )


# ============================================================
# TIC CANDIDATE EPHEMERIS
# ============================================================

def _find_tic_candidate_ephemerides(data):
    """
    Find TIC candidate blocks containing the ExoFOP fields

        epoch
        epoch_e
        per
        per_e
        dur
        dur_e

    plus optional depth values.
    """

    candidates = []

    def walk(obj, path="root"):

        if isinstance(obj, dict):

            if (
                "epoch" in obj
                and "per" in obj
                and "dur" in obj
            ):

                epoch = _safe_float(
                    obj.get("epoch")
                )

                period = _safe_float(
                    obj.get("per")
                )

                duration = _safe_float(
                    obj.get("dur")
                )

                if (
                    np.isfinite(epoch)
                    and np.isfinite(period)
                    and np.isfinite(duration)
                    and period > 0
                ):

                    candidates.append(
                        {
                            "epoch": epoch,
                            "epoch_err": _safe_float(
                                obj.get("epoch_e")
                            ),

                            "period": period,
                            "period_err": _safe_float(
                                obj.get("per_e")
                            ),

                            "duration_hours": duration,
                            "duration_err_hours": _safe_float(
                                obj.get("dur_e")
                            ),

                            "depth_mmag": _safe_float(
                                obj.get("dep_m")
                            ),

                            "depth_mmag_err": _safe_float(
                                obj.get("dep_m_e")
                            ),

                            "depth_ppm": _safe_float(
                                obj.get("dep_p")
                            ),

                            "depth_ppm_err": _safe_float(
                                obj.get("dep_p_e")
                            ),

                            "source_path": path,

                            "raw": obj,
                        }
                    )

            for key, value in obj.items():

                walk(
                    value,
                    f"{path}.{key}",
                )

        elif isinstance(obj, list):

            for i, value in enumerate(obj):

                walk(
                    value,
                    f"{path}[{i}]",
                )

    walk(data)

    # Remove duplicate blocks
    unique = []

    for candidate in candidates:

        duplicate = False

        for other in unique:

            if (
                np.isclose(
                    candidate["period"],
                    other["period"],
                    atol=1e-10,
                )
                and
                np.isclose(
                    candidate["epoch"],
                    other["epoch"],
                    atol=1e-8,
                )
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(candidate)

    return unique


def list_tic_candidates(tic):
    """
    List all candidate ephemerides found for a TIC.
    """

    data = get_exofop_tic_json(
        tic
    )

    candidates = (
        _find_tic_candidate_ephemerides(
            data
        )
    )

    if len(candidates) == 0:

        raise ValueError(
            f"No candidate ephemerides found for TIC {tic}."
        )

    return pd.DataFrame(
        [
            {
                "candidate": i,
                "period_days": c["period"],
                "period_err_days": c["period_err"],
                "epoch": c["epoch"],
                "epoch_err_days": c["epoch_err"],
                "duration_hours": c["duration_hours"],
                "duration_err_hours": c["duration_err_hours"],
                "depth_ppm": c["depth_ppm"],
                "source_path": c["source_path"],
            }
            for i, c in enumerate(candidates)
        ]
    )


# ============================================================
# TOI INFO
# ============================================================

def _get_toi_info(
    toi,
    exofop_df=None,
):
    """
    Read TOI ephemeris from the current ExoFOP TOI table.
    """

    if exofop_df is None:

        df = pd.read_csv(
            EXOFOP_TOI_URL
        )

    else:

        df = exofop_df.copy()

    df.columns = df.columns.str.strip()

    toi_col = _find_column(
        df,
        ["TOI"],
    )

    period_col = _find_column(
        df,
        [
            "Period (days)",
            "Period",
        ],
    )

    epoch_col = _find_column(
        df,
        [
            "Epoch (BJD)",
            "Transit Epoch",
            "Epoch",
        ],
    )

    duration_col = _find_column(
        df,
        [
            "Duration (hours)",
            "Transit Duration",
            "Duration",
        ],
    )

    ra_col = _find_column(
        df,
        ["RA", "RA (J2000)"],
    )

    dec_col = _find_column(
        df,
        ["Dec", "DEC", "Dec (J2000)"],
    )

    values = pd.to_numeric(
        df[toi_col],
        errors="coerce",
    )

    matches = df[
        np.isclose(
            values,
            float(toi),
            atol=1e-5,
        )
    ]

    if len(matches) == 0:

        raise ValueError(
            f"TOI {toi} was not found in the ExoFOP TOI table."
        )

    row = matches.iloc[0]

    try:

        coord = SkyCoord(
            ra=float(row[ra_col]) * u.deg,
            dec=float(row[dec_col]) * u.deg,
        )

    except Exception:

        coord = SkyCoord(
            str(row[ra_col]),
            str(row[dec_col]),
            unit=(
                u.hourangle,
                u.deg,
            ),
        )

    def optional_col(names):

        try:
            return _find_column(
                df,
                names,
            )
        except KeyError:
            return None

    period_err_col = optional_col(
        [
            "Period (days) err",
            "Period Error",
            "Period err",
        ]
    )

    epoch_err_col = optional_col(
        [
            "Epoch (BJD) err",
            "Epoch Error",
            "Epoch err",
        ]
    )

    duration_err_col = optional_col(
        [
            "Duration (hours) err",
            "Duration Error",
            "Duration err",
        ]
    )

    return {
        "name": f"TOI {float(toi):g}",

        "period": float(
            row[period_col]
        ),

        "period_err": (
            _safe_float(
                row[period_err_col]
            )
            if period_err_col is not None
            else np.nan
        ),

        "epoch": float(
            row[epoch_col]
        ),

        "epoch_err": (
            _safe_float(
                row[epoch_err_col]
            )
            if epoch_err_col is not None
            else np.nan
        ),

        "duration_hours": float(
            row[duration_col]
        ),

        "duration_err_hours": (
            _safe_float(
                row[duration_err_col]
            )
            if duration_err_col is not None
            else np.nan
        ),

        "coord": coord,

        "source": "ExoFOP bulk TOI table",
    }


# ============================================================
# TIC INFO
# ============================================================

def _get_tic_info(
    tic,
    candidate=0,
):
    """
    Read target coordinates and TIC-only candidate ephemeris.
    """

    tic = int(tic)

    data = get_exofop_tic_json(
        tic
    )

    coord = _find_ra_dec(
        data
    )

    candidates = (
        _find_tic_candidate_ephemerides(
            data
        )
    )

    if len(candidates) == 0:

        raise ValueError(
            f"TIC {tic} exists in ExoFOP, but no candidate "
            "block containing epoch/per/dur was found."
        )

    if candidate >= len(candidates):

        raise IndexError(
            f"TIC {tic} has {len(candidates)} candidate "
            f"ephemeris block(s), but candidate={candidate} "
            "was requested.\n"
            "Use list_tic_candidates(tic) to inspect them."
        )

    c = candidates[candidate]

    return {
        "name": f"TIC {tic}",
        "tic": tic,
        "candidate": candidate,

        "period": c["period"],
        "period_err": c["period_err"],

        "epoch": c["epoch"],
        "epoch_err": c["epoch_err"],

        "duration_hours": c["duration_hours"],
        "duration_err_hours": c["duration_err_hours"],

        "depth_ppm": c["depth_ppm"],
        "depth_ppm_err": c["depth_ppm_err"],

        "depth_mmag": c["depth_mmag"],
        "depth_mmag_err": c["depth_mmag_err"],

        "coord": coord,

        "source": (
            "ExoFOP TIC target JSON: "
            + c["source_path"]
        ),
    }


# ============================================================
# EPHEMERIS OVERRIDE
# ============================================================

def _apply_ephemeris_override(
    info,
    ephemeris_override,
):
    """
    Replace ExoFOP ephemeris values with supplied values.

    The override may be partial.

    Allowed fields
    --------------
    period
    period_err
    epoch
    epoch_err
    duration_hours
    duration_err_hours
    """

    if ephemeris_override is None:
        return info

    info = info.copy()

    allowed = {
        "period",
        "period_err",
        "epoch",
        "epoch_err",
        "duration_hours",
        "duration_err_hours",
    }

    unknown = (
        set(ephemeris_override.keys())
        - allowed
    )

    if unknown:

        raise KeyError(
            "Unknown ephemeris override field(s): "
            f"{sorted(unknown)}\n"
            f"Allowed fields are: {sorted(allowed)}"
        )

    overridden = []

    for key, value in ephemeris_override.items():

        if value is not None:

            info[key] = float(value)
            overridden.append(key)

    # Validate critical values
    if (
        not np.isfinite(info["period"])
        or info["period"] <= 0
    ):
        raise ValueError(
            "The final orbital period must be positive and finite."
        )

    if not np.isfinite(
        info["epoch"]
    ):
        raise ValueError(
            "The final transit epoch must be finite."
        )

    if (
        not np.isfinite(
            info["duration_hours"]
        )
        or info["duration_hours"] <= 0
    ):
        raise ValueError(
            "The final transit duration must be positive and finite."
        )

    original_source = info.get(
        "source",
        "ExoFOP",
    )

    info["source"] = (
        original_source
        + "; ephemeris override applied to: "
        + ", ".join(overridden)
    )

    info["ephemeris_override"] = (
        ephemeris_override.copy()
    )

    return info


# ============================================================
# UNIFIED LOADER
# ============================================================

def get_exofop_ephemeris(
    target,
    candidate=0,
    exofop_df=None,
    ephemeris_override=None,
):
    """
    Load target information and optionally replace the
    ExoFOP ephemeris with an updated one.
    """

    target_type, target_id = (
        _parse_target(target)
    )

    if target_type == "TOI":

        info = _get_toi_info(
            target_id,
            exofop_df=exofop_df,
        )

    else:

        info = _get_tic_info(
            target_id,
            candidate=candidate,
        )

    return _apply_ephemeris_override(
        info,
        ephemeris_override,
    )


# ============================================================
# BJD_TDB -> UTC
# ============================================================

def _bjd_tdb_to_utc(
    bjd,
    coord,
    location,
    iterations=3,
):
    """
    Convert BJD_TDB approximately to observatory UTC by
    reversing the barycentric light-travel correction.
    """

    bjd = np.asarray(
        bjd,
        dtype=float,
    )

    t = Time(
        bjd,
        format="jd",
        scale="tdb",
        location=location,
    )

    for _ in range(iterations):

        correction = (
            t.light_travel_time(
                coord,
                kind="barycentric",
            )
        )

        t = Time(
            bjd
            - correction.to_value(u.day),
            format="jd",
            scale="tdb",
            location=location,
        )

    return t.utc


# ============================================================
# TRANSIT WINDOWS
# ============================================================

def get_transit_windows(
    target,
    start_date,
    end_date,
    location=MAUNA_KEA,
    timezone_name="Pacific/Honolulu",
    candidate=0,
    baseline_hours=1.5,
    exofop_df=None,
    ephemeris_override=None,
):
    """
    Generate all predicted transit windows in a date range.

    ephemeris_override may contain any subset of:

        period
        period_err
        epoch
        epoch_err
        duration_hours
        duration_err_hours

    Epoch may be either full BJD_TDB or BTJD.
    Values < 100000 are assumed to be BTJD and 2457000 is added.
    """

    info = get_exofop_ephemeris(
        target=target,
        candidate=candidate,
        exofop_df=exofop_df,
        ephemeris_override=ephemeris_override,
    )

    coord = info["coord"]

    period = info["period"]

    period_err = info.get(
        "period_err",
        np.nan,
    )

    t0 = info["epoch"]

    t0_err = info.get(
        "epoch_err",
        np.nan,
    )

    duration_hours = (
        info["duration_hours"]
    )

    # --------------------------------------------------------
    # BTJD -> BJD
    # --------------------------------------------------------

    if t0 < 100000:
        t0 += 2457000.0

    # Store actual full BJD used
    info["epoch_bjd_tdb_used"] = t0

    # --------------------------------------------------------
    # Requested interval
    # --------------------------------------------------------

    range_start = Time(
        f"{start_date} 00:00:00",
        scale="utc",
    )

    range_end = Time(
        f"{end_date} 23:59:59",
        scale="utc",
    )

    n_start = int(
        np.floor(
            (
                range_start.tdb.jd
                - t0
            )
            / period
        )
    ) - 2

    n_end = int(
        np.ceil(
            (
                range_end.tdb.jd
                - t0
            )
            / period
        )
    ) + 2

    epoch_numbers = np.arange(
        n_start,
        n_end + 1,
    )

    mid_bjd = (
        t0
        + epoch_numbers * period
    )

    half_duration_days = (
        duration_hours
        / 48.0
    )

    start_bjd = (
        mid_bjd
        - half_duration_days
    )

    end_bjd = (
        mid_bjd
        + half_duration_days
    )

    # --------------------------------------------------------
    # Convert to observatory UTC
    # --------------------------------------------------------

    mid_utc = _bjd_tdb_to_utc(
        mid_bjd,
        coord,
        location,
    )

    start_utc = _bjd_tdb_to_utc(
        start_bjd,
        coord,
        location,
    )

    end_utc = _bjd_tdb_to_utc(
        end_bjd,
        coord,
        location,
    )

    keep = (
        (mid_utc.jd >= range_start.jd)
        &
        (mid_utc.jd <= range_end.jd)
    )

    epoch_numbers = (
        epoch_numbers[keep]
    )

    mid_bjd = mid_bjd[keep]

    mid_utc = mid_utc[keep]
    start_utc = start_utc[keep]
    end_utc = end_utc[keep]

    if len(mid_utc) == 0:

        raise ValueError(
            "No predicted transits occur within the requested date range."
        )

    # --------------------------------------------------------
    # Timing uncertainty
    #
    # Assumes T0 and P errors are independent.
    # --------------------------------------------------------

    if (
        np.isfinite(t0_err)
        and np.isfinite(period_err)
    ):

        mid_err_days = np.sqrt(
            t0_err**2
            +
            (
                epoch_numbers
                * period_err
            )**2
        )

    else:

        mid_err_days = np.full(
            len(epoch_numbers),
            np.nan,
        )

    # --------------------------------------------------------
    # Datetimes
    # --------------------------------------------------------

    local_zone = ZoneInfo(
        timezone_name
    )

    start_dt_utc = (
        start_utc.to_datetime(
            timezone=timezone.utc
        )
    )

    mid_dt_utc = (
        mid_utc.to_datetime(
            timezone=timezone.utc
        )
    )

    end_dt_utc = (
        end_utc.to_datetime(
            timezone=timezone.utc
        )
    )

    start_local = [
        x.astimezone(local_zone)
        for x in start_dt_utc
    ]

    mid_local = [
        x.astimezone(local_zone)
        for x in mid_dt_utc
    ]

    end_local = [
        x.astimezone(local_zone)
        for x in end_dt_utc
    ]

    # --------------------------------------------------------
    # Baseline observing window
    # --------------------------------------------------------

    baseline_days = (
        baseline_hours
        / 24.0
    )

    obs_start = Time(
        start_utc.jd - baseline_days,
        format="jd",
        scale="utc",
    )

    obs_end = Time(
        end_utc.jd + baseline_days,
        format="jd",
        scale="utc",
    )

    obs_start_dt = (
        obs_start.to_datetime(
            timezone=timezone.utc
        )
    )

    obs_end_dt = (
        obs_end.to_datetime(
            timezone=timezone.utc
        )
    )

    obs_start_local = [
        x.astimezone(local_zone)
        for x in obs_start_dt
    ]

    obs_end_local = [
        x.astimezone(local_zone)
        for x in obs_end_dt
    ]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "target": info["name"],

            "epoch_number":
                epoch_numbers,

            "date_local": [
                x.date()
                for x in mid_local
            ],

            "transit_start_utc":
                start_dt_utc,

            "midtransit_utc":
                mid_dt_utc,

            "transit_end_utc":
                end_dt_utc,

            "transit_start_local":
                start_local,

            "midtransit_local":
                mid_local,

            "transit_end_local":
                end_local,

            "obs_start_utc":
                obs_start_dt,

            "obs_end_utc":
                obs_end_dt,

            "obs_start_local":
                obs_start_local,

            "obs_end_local":
                obs_end_local,

            "midtransit_bjd_tdb":
                mid_bjd,

            "midtransit_err_days":
                mid_err_days,

            "midtransit_err_hours":
                mid_err_days * 24,

            "midtransit_err_min":
                mid_err_days * 24 * 60,
        }
    )

    result.attrs["ephemeris"] = info
    result.attrs["timezone"] = timezone_name
    result.attrs["location"] = location

    return result


# ============================================================
# VISIBILITY
# ============================================================

def add_visibility(
    transits,
    location=None,
    min_altitude=20,
    n_samples=150,
):
    """
    Add target and Sun visibility diagnostics.
    """

    df = transits.copy()

    # pandas .copy() normally preserves attrs,
    # but save explicitly just to be safe.
    attrs = transits.attrs.copy()

    info = attrs["ephemeris"]
    coord = info["coord"]

    if location is None:
        location = attrs["location"]

    transit_min_alt = []
    transit_mid_alt = []
    transit_max_alt = []

    obs_min_alt = []

    sun_mid_alt = []
    obs_max_sun_alt = []

    full_transit_visible = []
    full_obs_window_visible = []

    for _, row in df.iterrows():

        # Transit interval
        t1 = Time(
            row["transit_start_utc"]
        )

        t2 = Time(
            row["transit_end_utc"]
        )

        transit_times = Time(
            np.linspace(
                t1.jd,
                t2.jd,
                n_samples,
            ),
            format="jd",
            scale="utc",
            location=location,
        )

        transit_frame = AltAz(
            obstime=transit_times,
            location=location,
        )

        transit_alt = (
            coord.transform_to(
                transit_frame
            )
            .alt
            .deg
        )

        # Baseline + transit
        o1 = Time(
            row["obs_start_utc"]
        )

        o2 = Time(
            row["obs_end_utc"]
        )

        obs_times = Time(
            np.linspace(
                o1.jd,
                o2.jd,
                n_samples,
            ),
            format="jd",
            scale="utc",
            location=location,
        )

        obs_frame = AltAz(
            obstime=obs_times,
            location=location,
        )

        obs_alt = (
            coord.transform_to(
                obs_frame
            )
            .alt
            .deg
        )

        obs_sun = (
            get_sun(obs_times)
            .transform_to(obs_frame)
            .alt
            .deg
        )

        # Midpoint
        tmid = Time(
            row["midtransit_utc"]
        )

        mid_frame = AltAz(
            obstime=tmid,
            location=location,
        )

        mid_alt = (
            coord.transform_to(
                mid_frame
            )
            .alt
            .deg
        )

        mid_sun = (
            get_sun(tmid)
            .transform_to(mid_frame)
            .alt
            .deg
        )

        transit_min_alt.append(
            np.min(transit_alt)
        )

        transit_mid_alt.append(
            mid_alt
        )

        transit_max_alt.append(
            np.max(transit_alt)
        )

        obs_min_alt.append(
            np.min(obs_alt)
        )

        sun_mid_alt.append(
            mid_sun
        )

        obs_max_sun_alt.append(
            np.max(obs_sun)
        )

        full_transit_visible.append(
            np.all(
                transit_alt >= min_altitude
            )
        )

        full_obs_window_visible.append(
            np.all(
                obs_alt >= min_altitude
            )
        )

    df["alt_min_transit_deg"] = (
        transit_min_alt
    )

    df["alt_mid_deg"] = (
        transit_mid_alt
    )

    df["alt_max_transit_deg"] = (
        transit_max_alt
    )

    df["alt_min_obs_window_deg"] = (
        obs_min_alt
    )

    df["sun_alt_mid_deg"] = (
        sun_mid_alt
    )

    df["sun_alt_max_obs_window_deg"] = (
        obs_max_sun_alt
    )

    df["full_transit_visible"] = (
        full_transit_visible
    )

    df["full_obs_window_visible"] = (
        full_obs_window_visible
    )

    df.attrs = attrs

    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_transit_windows(
    transits,
    location=None,
    timezone_name=None,
    panel_hours=10,
    min_altitude=20,
    ncols=2,
    figsize_per_panel=(7, 4),
    show_timing_uncertainty=True,
):
    """
    Plot all transit opportunities.

    Background
    ----------
    black       : astronomical night
    dark blue   : astronomical twilight
    medium blue : nautical twilight
    light blue  : civil twilight
    pale blue   : daytime

    Other
    -----
    orange      : transit
    red dashed  : mid-transit
    cyan dotted : requested baseline edges
    yellow      : minimum altitude
    """

    df = transits

    info = df.attrs[
        "ephemeris"
    ]

    coord = info[
        "coord"
    ]

    if location is None:

        location = df.attrs[
            "location"
        ]

    if timezone_name is None:

        timezone_name = df.attrs[
            "timezone"
        ]

    local_zone = ZoneInfo(
        timezone_name
    )

    n_events = len(df)

    if n_events == 0:

        raise ValueError(
            "Transit dataframe is empty."
        )

    nrows = int(
        np.ceil(
            n_events / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_panel[0]
            * ncols,

            figsize_per_panel[1]
            * nrows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    twilight_colors = {
        0: "#dcecf4",  # day
        1: "#86b9df",  # civil
        2: "#477fba",  # nautical
        3: "#193f70",  # astronomical twilight
        4: "#000000",  # night
    }

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        ax = axes[i]

        mid = Time(
            row["midtransit_utc"]
        )

        half_panel_days = (
            panel_hours / 48.0
        )

        times = Time(
            np.linspace(
                mid.jd
                - half_panel_days,

                mid.jd
                + half_panel_days,

                600,
            ),
            format="jd",
            scale="utc",
            location=location,
        )

        frame = AltAz(
            obstime=times,
            location=location,
        )

        altitude = (
            coord.transform_to(frame)
            .alt
            .deg
        )

        sun_alt = (
            get_sun(times)
            .transform_to(frame)
            .alt
            .deg
        )

        dt = times.to_datetime(
            timezone=timezone.utc
        )

        # ----------------------------------------------------
        # TWILIGHT BACKGROUND
        # ----------------------------------------------------

        category = np.zeros(
            len(times),
            dtype=int,
        )

        category[
            (sun_alt <= 0)
            &
            (sun_alt > -6)
        ] = 1

        category[
            (sun_alt <= -6)
            &
            (sun_alt > -12)
        ] = 2

        category[
            (sun_alt <= -12)
            &
            (sun_alt > -18)
        ] = 3

        category[
            sun_alt <= -18
        ] = 4

        segment_start = 0

        for j in range(
            1,
            len(category) + 1,
        ):

            if (
                j == len(category)
                or
                category[j]
                != category[segment_start]
            ):

                ax.axvspan(
                    dt[segment_start],
                    dt[j - 1],
                    color=twilight_colors[
                        category[
                            segment_start
                        ]
                    ],
                    linewidth=0,
                    zorder=0,
                )

                segment_start = j

        # ----------------------------------------------------
        # TARGET ALTITUDE
        # ----------------------------------------------------

        ax.plot(
            dt,
            altitude,
            color="white",
            linewidth=2.2,
            zorder=5,
        )

        # Transit
        ax.axvspan(
            row["transit_start_utc"],
            row["transit_end_utc"],
            color="orange",
            alpha=0.60,
            zorder=3,
        )

        # Mid-transit
        ax.axvline(
            row["midtransit_utc"],
            color="red",
            linestyle="--",
            linewidth=1.4,
            zorder=6,
        )

        # Baseline edges
        ax.axvline(
            row["obs_start_utc"],
            color="cyan",
            linestyle=":",
            linewidth=1.2,
            zorder=6,
        )

        ax.axvline(
            row["obs_end_utc"],
            color="cyan",
            linestyle=":",
            linewidth=1.2,
            zorder=6,
        )

        # Altitude threshold
        ax.axhline(
            min_altitude,
            color="yellow",
            linestyle=":",
            linewidth=1.0,
            zorder=6,
        )

        # ----------------------------------------------------
        # Y AXIS
        # ----------------------------------------------------

        ax.set_ylim(
            0,
            90,
        )

        ax.set_ylabel(
            "Altitude [deg]"
        )

        ax.grid(
            alpha=0.15,
            color="white",
        )

        # ----------------------------------------------------
        # TIME TICKS
        # ----------------------------------------------------

        ax.xaxis.set_major_locator(
            mdates.HourLocator(
                interval=1
            )
        )

        # Bottom: local time
        def local_formatter(
            x,
            pos=None,
        ):

            dt_utc = mdates.num2date(
                x,
                tz=timezone.utc,
            )

            dt_local = (
                dt_utc.astimezone(
                    local_zone
                )
            )

            return dt_local.strftime(
                "%H:%M"
            )

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                local_formatter
            )
        )

        local_mid = row[
            "midtransit_local"
        ]

        local_timezone_label = (
            local_mid.tzname()
            or timezone_name
        )

        ax.set_xlabel(
            local_timezone_label
        )

        # ----------------------------------------------------
        # TOP AXIS: UTC
        # ----------------------------------------------------

        top = ax.twiny()

        top.set_xlim(
            ax.get_xlim()
        )

        ticks = ax.get_xticks()

        top.set_xticks(
            ticks
        )

        top.set_xticklabels(
            [
                mdates.num2date(
                    tick,
                    tz=timezone.utc,
                ).strftime(
                    "%H:%M"
                )
                for tick in ticks
            ]
        )

        top.set_xlabel(
            "UTC"
        )

        # ----------------------------------------------------
        # TITLE — EXPLICIT TIMEZONE
        # ----------------------------------------------------

        title = (
            f"{info['name']} — "
            f"{local_mid:%Y-%m-%d} "
            f"{local_timezone_label}"
        )

        if (
            show_timing_uncertainty
            and np.isfinite(
                row[
                    "midtransit_err_hours"
                ]
            )
        ):

            err_hr = row[
                "midtransit_err_hours"
            ]

            if err_hr < 1:

                title += (
                    "\n"
                    r"$\sigma_{T_c}$ = "
                    f"{err_hr * 60:.0f} min"
                )

            else:

                title += (
                    "\n"
                    r"$\sigma_{T_c}$ = "
                    f"{err_hr:.1f} hr"
                )

        ax.set_title(
            title,
            pad=28,
        )

    # Hide unused axes
    for j in range(
        n_events,
        len(axes),
    ):

        axes[j].axis(
            "off"
        )

    fig.suptitle(
        f"{info['name']} transit visibility",
        fontsize=16,
        y=1.01,
    )

    fig.tight_layout()

    return fig


# ============================================================
# ALL-IN-ONE WRAPPER
# ============================================================

def plot_exofop_transits(
    target,
    start_date,
    end_date,
    location=MAUNA_KEA,
    timezone_name="Pacific/Honolulu",
    candidate=0,
    baseline_hours=1.5,
    panel_hours=10,
    min_altitude=20,
    ncols=2,
    exofop_df=None,
    ephemeris_override=None,
    show_timing_uncertainty=True,
):
    """
    Full target -> ephemeris -> transit windows -> visibility -> plot
    workflow.

    Parameters
    ----------
    target
        e.g.
            "TIC 139474683"
            "TOI 2048.01"

    ephemeris_override : dict, optional
        Any subset of:

            {
                "period": ...,
                "period_err": ...,
                "epoch": ...,
                "epoch_err": ...,
                "duration_hours": ...,
                "duration_err_hours": ...
            }

        Supplied values replace ExoFOP values.

    Returns
    -------
    transits : pandas.DataFrame
    fig : matplotlib.figure.Figure
    """

    transits = get_transit_windows(
        target=target,
        start_date=start_date,
        end_date=end_date,
        location=location,
        timezone_name=timezone_name,
        candidate=candidate,
        baseline_hours=baseline_hours,
        exofop_df=exofop_df,
        ephemeris_override=(
            ephemeris_override
        ),
    )

    transits = add_visibility(
        transits,
        location=location,
        min_altitude=min_altitude,
    )

    fig = plot_transit_windows(
        transits,
        location=location,
        timezone_name=timezone_name,
        panel_hours=panel_hours,
        min_altitude=min_altitude,
        ncols=ncols,
        show_timing_uncertainty=(
            show_timing_uncertainty
        ),
    )

    info = transits.attrs[
        "ephemeris"
    ]

    print(
        f"Target: {info['name']}"
    )

    print(
        f"Ephemeris source: "
        f"{info['source']}"
    )

    print(
        f"P = {info['period']:.10f} d"
    )

    if np.isfinite(
        info.get(
            "period_err",
            np.nan,
        )
    ):

        print(
            f"σP = "
            f"{info['period_err']:.10f} d"
        )

    print(
        f"T0 = "
        f"{info['epoch_bjd_tdb_used']:.8f} "
        f"BJD_TDB"
    )

    if np.isfinite(
        info.get(
            "epoch_err",
            np.nan,
        )
    ):

        print(
            f"σT0 = "
            f"{info['epoch_err']:.8f} d"
        )

    print(
        f"Duration = "
        f"{info['duration_hours']:.4f} hr"
    )

    return transits, fig