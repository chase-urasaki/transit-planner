# ============================================================
# TESS ROTATION + GYROINTERP QUICK-LOOK PIPELINE
# ============================================================
#%%
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

from astropy.timeseries import LombScargle
from gyrointerp import gyro_age_posterior, get_summary_statistics


# ============================================================
# 1. Download TESS light curves
# ============================================================

def get_tess_lightcurve(tic_id):
    """
    Download, normalize, and stitch all available TESS SPOC
    light curves for a given TIC.

    Returns
    -------
    lc : lightkurve.LightCurve
        Stitched, normalized TESS light curve.
    """

    print(f"\nSearching TESS SPOC light curves for TIC {tic_id}...")

    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author="SPOC"
    )

    print(search)

    if len(search) == 0:
        raise RuntimeError(
            f"No TESS SPOC light curves found for TIC {tic_id}."
        )

    print(f"\nFound {len(search)} light-curve products.")
    print("Downloading...")

    collection = search.download_all(
        quality_bitmask="default",
        flux_column="pdcsap_flux"
    )

    if collection is None or len(collection) == 0:
        raise RuntimeError("Download failed or returned no light curves.")

    # Normalize each sector individually
    normalized = lk.LightCurveCollection(
        [lc.normalize() for lc in collection]
    )

    # Stitch all sectors together
    lc = normalized.stitch()

    # Basic cleanup
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=5)

    print(f"Final light curve contains {len(lc)} cadences.")

    return lc


# ============================================================
# 2. Mask planetary transits
# ============================================================

def mask_multiple_transits(
    lc,
    planets,
    duration_factor=1.5
):
    """
    Mask transits from one or more planets.

    Each planet dictionary must contain:

        name
        period      [days]
        t0          [BJD or BTJD]
        duration    [hours]

    Full BJD epochs are automatically converted to TESS BTJD.
    """

    time = lc.time.value

    keep = np.ones(len(time), dtype=bool)

    print("\nMasking transits:")

    for planet in planets:

        name = planet.get("name", "Planet")
        period = planet["period"]
        t0 = planet["t0"]
        duration_hours = planet["duration"]

        # ----------------------------------------------------
        # Convert BJD -> BTJD if necessary
        #
        # TESS:
        # BTJD = BJD - 2457000
        # ----------------------------------------------------

        if t0 > 1_000_000:

            t0_btjd = t0 - 2457000.0

            print(
                f"  {name}: converting "
                f"BJD {t0:.6f} -> BTJD {t0_btjd:.6f}"
            )

        else:

            t0_btjd = t0

        duration_days = duration_hours / 24.0

        # ----------------------------------------------------
        # Distance from nearest transit center
        # ----------------------------------------------------

        dt = (
            (time - t0_btjd + 0.5 * period) % period
            - 0.5 * period
        )

        # ----------------------------------------------------
        # Transit mask
        # ----------------------------------------------------

        in_transit = (
            np.abs(dt)
            <
            0.5 * duration_days * duration_factor
        )

        keep &= ~in_transit

        print(
            f"  {name}: masked "
            f"{np.sum(in_transit)} cadences"
        )

    print(
        f"\nKept {np.sum(keep)} / {len(keep)} cadences "
        f"({100*np.mean(keep):.1f}%)"
    )

    return lc[keep]


# ============================================================
# 3. Lomb-Scargle rotation search
# ============================================================

def find_rotation_period(
    lc,
    min_period=0.5,
    max_period=30,
    flatten_window=5.0
):
    """
    Flatten the TESS light curve and calculate a Lomb-Scargle
    periodogram.

    Parameters
    ----------
    flatten_window : float
        Approximate detrending window in DAYS.

    Returns
    -------
    flat_lc
    periods
    power
    best_period
    """

    # --------------------------------------------------------
    # Determine cadence
    # --------------------------------------------------------

    cadence = np.nanmedian(
        np.diff(lc.time.value)
    )

    window_length = int(
        flatten_window / cadence
    )

    # Lightkurve needs an odd window
    if window_length % 2 == 0:
        window_length += 1

    # Minimum sensible value
    window_length = max(window_length, 5)

    print(
        f"\nMedian cadence = "
        f"{cadence * 24 * 60:.2f} minutes"
    )

    print(
        f"Flattening window = "
        f"{flatten_window:.1f} days "
        f"({window_length} cadences)"
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flat_lc = lc.flatten(
        window_length=window_length,
        break_tolerance=5
    )

    time = flat_lc.time.value
    flux = flat_lc.flux.value

    good = (
        np.isfinite(time)
        &
        np.isfinite(flux)
    )

    time = time[good]
    flux = flux[good]

    # --------------------------------------------------------
    # Lomb-Scargle
    # --------------------------------------------------------

    frequency, power = LombScargle(
        time,
        flux
    ).autopower(
        minimum_frequency=1 / max_period,
        maximum_frequency=1 / min_period,
        samples_per_peak=10
    )

    periods = 1 / frequency

    best_period = periods[
        np.argmax(power)
    ]

    print(
        f"\nStrongest Lomb-Scargle period: "
        f"{best_period:.3f} days"
    )

    return (
        flat_lc,
        periods,
        power,
        best_period
    )


# ============================================================
# 4. gyrointerp age
# ============================================================

def calculate_gyro_age(
    prot,
    teff,
    prot_err=None,
    teff_err=None,
    max_age=2600
):
    """
    Calculate gyrochronology age using gyrointerp.

    Parameters
    ----------
    prot : float
        Rotation period [days].

    teff : float
        Effective temperature [K].

    prot_err : float, optional
        Rotation-period uncertainty [days].

    teff_err : float, optional
        Effective-temperature uncertainty [K].

    max_age : float
        Maximum age considered [Myr].
    """

    print("\nRunning gyrointerp...")
    print(f"P_rot = {prot:.3f} days")
    print(f"Teff  = {teff:.0f} K")

    # For quick-look purposes, 500 grid points is plenty.
    age_grid = np.linspace(
        0,
        max_age,
        500
    )

    age_posterior = gyro_age_posterior(
        prot,
        teff,
        Prot_err=prot_err,
        Teff_err=teff_err,
        age_grid=age_grid
    )

    # gyrointerp's own summary-statistics function
    stats = get_summary_statistics(
        age_grid,
        age_posterior
    )

    age = stats["median"]
    age_plus = stats["+1sigma"]
    age_minus = stats["-1sigma"]

    print("\n----------------------------")
    print("Gyrochronology")
    print("----------------------------")

    print(
        f"Age = {age:.0f} "
        f"+{age_plus:.0f} "
        f"-{age_minus:.0f} Myr"
    )

    return {
        "age_myr": age,
        "age_plus": age_plus,
        "age_minus": age_minus,
        "age_grid": age_grid,
        "age_posterior": age_posterior
    }


# ============================================================
# 5. Full analysis
# ============================================================

def analyze_rotation(
    tic_id,
    planets,
    teff,
    prot_err=None,
    teff_err=None,
    min_rotation_period=0.5,
    max_rotation_period=30,
    flatten_window=5.0,
    duration_factor=1.5,
    plot=True
):
    """
    Full quick-look pipeline:

        TIC
         |
         v
       TESS
         |
         v
    mask transits
         |
         v
    Lomb-Scargle
         |
         v
       P_rot
         |
         v
     gyrointerp
         |
         v
        Age
    """

    # ========================================================
    # Download
    # ========================================================

    lc = get_tess_lightcurve(
        tic_id
    )

    # ========================================================
    # Mask transits
    # ========================================================

    masked_lc = mask_multiple_transits(
        lc,
        planets,
        duration_factor=duration_factor
    )

    # ========================================================
    # Rotation
    # ========================================================

    (
        flat_lc,
        periods,
        power,
        prot
    ) = find_rotation_period(
        masked_lc,
        min_period=min_rotation_period,
        max_period=max_rotation_period,
        flatten_window=flatten_window
    )

    # ========================================================
    # Gyro age
    # ========================================================

    gyro = calculate_gyro_age(
        prot,
        teff,
        prot_err=prot_err,
        teff_err=teff_err
    )

    # ========================================================
    # Final result
    # ========================================================

    print("\n")
    print("=" * 45)
    print("FINAL RESULT")
    print("=" * 45)

    print(f"TIC       : {tic_id}")
    print(f"Teff      : {teff:.0f} K")
    print(f"P_rot     : {prot:.3f} d")

    print(
        f"Gyro age  : "
        f"{gyro['age_myr']:.0f} "
        f"+{gyro['age_plus']:.0f} "
        f"-{gyro['age_minus']:.0f} Myr"
    )

    print("=" * 45)

    # ========================================================
    # Diagnostic plots
    # ========================================================

    if plot:

        # ----------------------------------------------------
        # Transit-masked LC
        # ----------------------------------------------------

        plt.figure(figsize=(12, 4))

        plt.scatter(
            masked_lc.time.value,
            masked_lc.flux.value,
            s=2
        )

        plt.xlabel("Time [BTJD]")
        plt.ylabel("Normalized Flux")

        plt.title(
            f"TIC {tic_id} — Transit-Masked TESS Light Curve"
        )

        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------
        # Periodogram
        # ----------------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            periods,
            power
        )

        plt.axvline(
            prot,
            linestyle="--",
            label=f"P = {prot:.3f} d"
        )

        # Also show 2P and P/2 because rotation harmonics
        # are common in spotted stars.

        if 2 * prot <= max_rotation_period:

            plt.axvline(
                2 * prot,
                linestyle=":",
                alpha=0.5,
                label=f"2P = {2*prot:.3f} d"
            )

        if prot / 2 >= min_rotation_period:

            plt.axvline(
                prot / 2,
                linestyle=":",
                alpha=0.5,
                label=f"P/2 = {prot/2:.3f} d"
            )

        plt.xlabel("Period [days]")
        plt.ylabel("Lomb-Scargle Power")

        plt.xlim(
            min_rotation_period,
            max_rotation_period
        )

        plt.legend()

        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------
        # Rotation-folded LC
        # ----------------------------------------------------

        folded = flat_lc.fold(
            period=prot
        )

        plt.figure(figsize=(10, 5))

        plt.scatter(
            folded.phase.value,
            folded.flux.value,
            s=3,
            alpha=0.5
        )

        plt.xlabel("Rotational Phase")
        plt.ylabel("Normalized Flux")

        plt.title(
            f"TIC {tic_id}: "
            f"$P_{{rot}} = {prot:.3f}$ d"
        )

        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------
        # Gyro age posterior
        # ----------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            gyro["age_grid"],
            gyro["age_posterior"]
        )

        plt.axvline(
            gyro["age_myr"],
            linestyle="--",
            label=(
                f"{gyro['age_myr']:.0f} "
                f"+{gyro['age_plus']:.0f} "
                f"-{gyro['age_minus']:.0f} Myr"
            )
        )

        plt.xlabel("Age [Myr]")
        plt.ylabel("Posterior Probability Density")

        plt.title(
            f"TIC {tic_id} — gyrointerp"
        )

        plt.legend()

        plt.tight_layout()
        plt.show()

    # ========================================================
    # Return everything
    # ========================================================

    return {

        "tic": tic_id,

        "teff": teff,

        "rotation_period_days": prot,

        "gyro_age_myr": gyro["age_myr"],
        "gyro_age_plus": gyro["age_plus"],
        "gyro_age_minus": gyro["age_minus"],

        "lightcurve": lc,
        "masked_lightcurve": masked_lc,
        "flattened_lightcurve": flat_lc,

        "periods": periods,
        "power": power,

        "age_grid": gyro["age_grid"],
        "age_posterior": gyro["age_posterior"]
    }


# ============================================================
# TARGET INFORMATION
# ============================================================

tic_id = 303204165

teff = 5991  # K

planets = [

    {
        "name": "TOI-4647.01",
        "period": 8.834,            # days
        "t0": 2459451.542506,      # BJD
        "duration": 3.014           # hours
    },

    {
        "name": "TOI-4647.02",
        "period": 11.070,           # days
        "t0": 2459451.391313,      # BJD
        "duration": 4.492           # hours
    }

]


# ============================================================
# RUN
# ============================================================

result = analyze_rotation(

    tic_id=tic_id,

    planets=planets,

    teff=teff,

    # Optional uncertainties.
    # If None, gyrointerp defaults to ~1% in Prot and 100 K in Teff.
    prot_err=None,
    teff_err=None,

    # Rotation search range
    min_rotation_period=0.5,
    max_rotation_period=15,

    # Don't make this too short relative to P_rot
    flatten_window=5.0,

    duration_factor=1.5,

    plot=True
)
#%%
#%%
masked_lc = result["masked_lightcurve"]
# %%
def compare_flattening_windows(
    masked_lc,
    windows=(None, 5, 10, 20),
    min_period=0.5,
    max_period=15
):
    """
    Compare Lomb-Scargle periodograms for different
    light-curve detrending windows.

    None = no flattening.
    """

    plt.figure(figsize=(12, 6))

    results = {}

    for window in windows:

        # ----------------------------------------------
        # Prepare light curve
        # ----------------------------------------------

        if window is None:

            test_lc = masked_lc
            label = "No flattening"

        else:

            cadence = np.nanmedian(
                np.diff(masked_lc.time.value)
            )

            window_length = int(window / cadence)

            if window_length % 2 == 0:
                window_length += 1

            test_lc = masked_lc.flatten(
                window_length=window_length,
                break_tolerance=5
            )

            label = f"{window} d flatten"

        # ----------------------------------------------
        # Lomb-Scargle
        # ----------------------------------------------

        time = test_lc.time.value
        flux = test_lc.flux.value

        good = (
            np.isfinite(time)
            &
            np.isfinite(flux)
        )

        time = time[good]
        flux = flux[good]

        frequency, power = LombScargle(
            time,
            flux
        ).autopower(
            minimum_frequency=1/max_period,
            maximum_frequency=1/min_period,
            samples_per_peak=10
        )

        periods = 1 / frequency

        best_period = periods[
            np.argmax(power)
        ]

        results[window] = best_period

        print(
            f"{label:20s}: "
            f"P = {best_period:.3f} d"
        )

        # Normalize so we're comparing peak locations,
        # not absolute LS power
        plt.plot(
            periods,
            power / np.nanmax(power),
            label=f"{label}: {best_period:.2f} d"
        )

    plt.xlabel("Period [days]")
    plt.ylabel("Normalized Lomb-Scargle Power")

    plt.xlim(
        min_period,
        max_period
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
# %%
masked_lc = result["masked_lightcurve"]

period_comparison = compare_flattening_windows(
    masked_lc,
    windows=(None, 5, 10, 20),
    min_period=0.5,
    max_period=15
)
# %%
