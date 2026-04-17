"""
run_catalog_pipeline.py
---------------
Main script: runs the chunk pipeline over the UNIONS ShapePipe catalog,
counts objects passing all cuts, and builds histograms for every column.

Histogram groups (each saved to a separate figure):
    1.  Sky position & weight         RA, Dec, w_iv
    2.  Photometry (SExtractor)       mag, snr, MAGERR_AUTO, MAG_WIN, MAGERR_WIN
    3.  Flux (SExtractor)             FLUX_AUTO, FLUXERR_AUTO, FLUX_APER, FLUXERR_APER
    4.  Size & shape (SExtractor)     FLUX_RADIUS, FWHM_IMAGE, FWHM_WORLD
    5.  Ellipticities (uncal + PSF)   e1_uncal, e2_uncal, e1_PSF, e2_PSF, fwhm_PSF
    6.  Flags & epochs                IMAFLAGS_ISO, FLAGS, NGMIX_MCAL_FLAGS,
                                      NGMIX_MOM_FAIL, N_EPOCH, NGMIX_NGMIX_N_EPOCH,
                                      NGMIX_FLAGS_*, TILE_ID
    7.  NGMIX ellipticities           NGMIX_ELL_*
    8.  NGMIX flux                    NGMIX_FLUX_*, NGMIX_FLUX_ERR_*
    9.  NGMIX size (T)                NGMIX_T_*, NGMIX_T_ERR_*, NGMIX_Tpsf_*
    10. Photo-z (BPZ)                 Z_B, Z_B_MIN, Z_B_MAX, T_B
    11. GAaP magnitudes               MAG_GAAP_0p7_*
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catalog_pipeline import (
    CutCounter, Histogram, HistogramGroup, HealpixStats,
    chunk_runner_serial, summarize_serial,
    chunk_runner_mpi, summarize_mpi,
)
from slide_tools import make_slides
import healpy as hp
from pathlib import Path
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAT_PATH   = "/arc/projects/unions/lensing/ShapePipe/v1.6.x/unions_shapepipe_comprehensive_struc_ugriz_2024_v1.6.c.1.hdf5"
output_dir = "output_mpi_wTcut_ugri/"
CACHE_PATH = output_dir+"pipeline_cache.hdf5"
resume=False
chunk_size = 1_000_000
nchunks    = None       # set e.g. nchunks=5 for a quick test, None for full run
use_mpi = True

# ---------------------------------------------------------------------------
# Cut definitions
# ---------------------------------------------------------------------------

cut_defs = [
    ("FLAGS == 0",               lambda x: x["FLAGS"] == 0),
    ("NGMIX_MCAL_FLAGS == 0",    lambda x: x["NGMIX_MCAL_FLAGS"] == 0),
    ("NGMIX_FLAGS_NOSHEAR == 0", lambda x: x["NGMIX_FLAGS_NOSHEAR"] == 0),
    ("N_EPOCH >= 2",             lambda x: x["N_EPOCH"] >= 2),
    ("mag > 15",                 lambda x: x["mag"] > 15),
    ("mag < 30",                 lambda x: x["mag"] < 30),
    ("snr > 10",                 lambda x: x["snr"] > 10),
    ("snr < 500",                lambda x: x["snr"] < 500),
    ("T_ratio >= 0.707",         lambda x: x["NGMIX_T_NOSHEAR"] / x["NGMIX_Tpsf_NOSHEAR"] >= 0.707),
    ("T_ratio <= 3.0",           lambda x: x["NGMIX_T_NOSHEAR"] / x["NGMIX_Tpsf_NOSHEAR"] <= 3.0),
    ("Z_B >= 0.0",               lambda x: x["Z_B"] >= 0.0),
    ("Z_B <= 3.0",               lambda x: x["Z_B"] <= 3.0),
    ("MAG_GAAP_0p7_u != -99.",   lambda x: x["MAG_GAAP_0p7_u"] != -99.),
    ("MAG_GAAP_0p7_g != -99.",   lambda x: x["MAG_GAAP_0p7_g"] != -99.),
    ("MAG_GAAP_0p7_r != -99.",   lambda x: x["MAG_GAAP_0p7_r"] != -99.),
    ("MAG_GAAP_0p7_i != -99.",   lambda x: x["MAG_GAAP_0p7_i"] != -99.),
    #("MAG_GAAP_0p7_z(2) != -99.",lambda x: (x["MAG_GAAP_0p7_z"] != -99.)|(x["MAG_GAAP_0p7_z2"] != -99.) ),
]

# ---------------------------------------------------------------------------
# quantities I want maps for
# ---------------------------------------------------------------------------

HEALPIX_FIELDS = [
 'w_iv',
 'mag',
 'snr',
 'e1_uncal',
 'e2_uncal',
 'FLUX_RADIUS',
 'FWHM_IMAGE',
 'FWHM_WORLD',
 'MAGERR_AUTO',
 'MAG_WIN',
 'MAGERR_WIN',
 'FLUX_AUTO',
 'FLUXERR_AUTO',
 'FLUX_APER',
 'FLUXERR_APER',
 'NGMIX_T_NOSHEAR',
 'NGMIX_Tpsf_NOSHEAR',
 'TILE_ID',
 'IMAFLAGS_ISO',
 'FLAGS',
 'NGMIX_MCAL_FLAGS',
 'NGMIX_MOM_FAIL',
 'N_EPOCH',
 'NGMIX_N_EPOCH',
 'NGMIX_ELL_NOSHEAR_0',
 'NGMIX_ELL_NOSHEAR_1',
 'NGMIX_FLUX_NOSHEAR',
 'NGMIX_FLUX_ERR_NOSHEAR',
 'NGMIX_T_ERR_NOSHEAR',
 'e1_PSF',
 'e2_PSF',
 'fwhm_PSF',
 'patch',
 'Z_B',
 'Z_B_MIN',
 'Z_B_MAX',
 'T_B',
 'MAG_GAAP_0p7_u',
 'MAG_GAAP_0p7_g',
 'MAG_GAAP_0p7_r',
 'MAG_GAAP_0p7_i',
 'MAG_GAAP_0p7_z',
 'MAG_GAAP_0p7_z2'
]

# ---------------------------------------------------------------------------
# Histogram definitions
# Grouped by physical meaning. Each entry:
#   (field, bins, range, log)
# log=True  → histogram of log10(|value|); good for quantities spanning decades
# ---------------------------------------------------------------------------

# Each group is (group_title, filename_stem, [(field, bins, range, log), ...])
HISTOGRAM_GROUPS = [

    ("Sky position & weight", "hist_position", [
        ("RA",   100, (0.0,   360.0), False),
        ("Dec",  100, (-20.0,  90.0), False),
        ("w_iv", 50,  (0, 5),        False ),   
    ]),

    ("Photometry", "hist_photometry", [
        ("mag",        50, (15.0, 30.0),  False),
        ("snr",        50, (10.0, 500.0), False),
        ("MAGERR_AUTO",50, (0.0,  1.0),   False),
        ("MAG_WIN",    50, (15.0, 30.0),  False),
        ("MAGERR_WIN", 50, (0.0,  1.0),   False),
    ]),

    ("Flux (SExtractor)", "hist_flux_sex", [
        ("FLUX_AUTO",     50, (0, np.log10(60000)), True),
        ("FLUXERR_AUTO",  50, (0, np.log10(100)), True),
        ("FLUX_APER",     50, (0, np.log10(6000)), True),
        ("FLUXERR_APER",  50, (0, np.log10(30)), True),
    ]),

    ("Size & shape (SExtractor)", "hist_size_sex", [
        ("FLUX_RADIUS",  50, (0.0, 20.0),  False),
        ("FWHM_IMAGE",   50, (0.0, 20.0),  False),
        ("FWHM_WORLD",   50, (0.0, 0.005), False),
    ]),

    ("Ellipticities & PSF shape", "hist_ellip", [
        ("e1_uncal",  50, (-1.0, 1.0), False),
        ("e2_uncal",  50, (-1.0, 1.0), False),
        ("e1_PSF",    50, (-1.0, 1.0), False),
        ("e2_PSF",    50, (-1.0, 1.0), False),
        ("fwhm_PSF",  50, (0.0,  2.0), False),
    ]),

    ("Flags & epochs", "hist_flags", [
        ("IMAFLAGS_ISO",     20, (0,  20), False),
        ("FLAGS",            10, (0,  10), False),
        ("NGMIX_MCAL_FLAGS", 10, (0,  10), False),
        ("NGMIX_MOM_FAIL",   10, (0,  10), False),
        ("N_EPOCH",          20, (0,  20), False),
        ("NGMIX_N_EPOCH",    20, (0,  20), False),
        ("NGMIX_FLAGS_NOSHEAR", 10, (0, 10), False),
        ("NGMIX_FLAGS_1M",   10, (0,  10), False),
        ("NGMIX_FLAGS_1P",   10, (0,  10), False),
        ("NGMIX_FLAGS_2M",   10, (0,  10), False),
        ("NGMIX_FLAGS_2P",   10, (0,  10), False),
        ("TILE_ID",          50, (0, 700),      False),
    ]),

    ("NGMIX ellipticities", "hist_ngmix_ell", [
        ("NGMIX_ELL_NOSHEAR_0",   50, (-1.0, 1.0), False),
        ("NGMIX_ELL_NOSHEAR_1",   50, (-1.0, 1.0), False),
        ("NGMIX_ELL_1M_0",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_1M_1",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_1P_0",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_1P_1",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_2M_0",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_2M_1",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_2P_0",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_2P_1",        50, (-1.0, 1.0), False),
        ("NGMIX_ELL_PSFo_NOSHEAR_0", 50, (-1.0, 1.0), False),
        ("NGMIX_ELL_PSFo_NOSHEAR_1", 50, (-1.0, 1.0), False),
        ("NGMIX_ELL_ERR_NOSHEAR_0",  50, (0.0,  0.5), False),
        ("NGMIX_ELL_ERR_NOSHEAR_1",  50, (0.0,  0.5), False),
    ]),

    ("NGMIX flux", "hist_ngmix_flux", [
        ("NGMIX_FLUX_NOSHEAR",    50, (0, np.log10(30000)), True),
        ("NGMIX_FLUX_1M",         50, (0, np.log10(30000)), True),
        ("NGMIX_FLUX_1P",         50, (0, np.log10(30000)), True),
        ("NGMIX_FLUX_2M",         50, (0, np.log10(30000)), True),
        ("NGMIX_FLUX_2P",         50, (0, np.log10(30000)), True),
        ("NGMIX_FLUX_ERR_NOSHEAR",50, (0, np.log10(1000)), True),
        ("NGMIX_FLUX_ERR_1M",     50, (0, np.log10(1000)), True),
        ("NGMIX_FLUX_ERR_1P",     50, (0, np.log10(1000)), True),
        ("NGMIX_FLUX_ERR_2M",     50, (0, np.log10(1000)), True),
        ("NGMIX_FLUX_ERR_2P",     50, (0, np.log10(1000)), True),
    ]),

    ("NGMIX size (T)", "hist_ngmix_T", [
        ("NGMIX_T_NOSHEAR",    50, (0.0, 5.0),  False),
        ("NGMIX_T_1M",         50, (0.0, 5.0),  False),
        ("NGMIX_T_1P",         50, (0.0, 5.0),  False),
        ("NGMIX_T_2M",         50, (0.0, 5.0),  False),
        ("NGMIX_T_2P",         50, (0.0, 5.0),  False),
        ("NGMIX_T_ERR_NOSHEAR",50, (0.0, 1.0),  False),
        ("NGMIX_T_ERR_1M",     50, (0.0, 1.0),  False),
        ("NGMIX_T_ERR_1P",     50, (0.0, 1.0),  False),
        ("NGMIX_T_ERR_2M",     50, (0.0, 1.0),  False),
        ("NGMIX_T_ERR_2P",     50, (0.0, 1.0),  False),
        ("NGMIX_Tpsf_NOSHEAR", 50, (0.0, 1.0),  False),
        ("NGMIX_Tpsf_1M",      50, (0.0, 1.0),  False),
        ("NGMIX_Tpsf_1P",      50, (0.0, 1.0),  False),
        ("NGMIX_Tpsf_2M",      50, (0.0, 1.0),  False),
        ("NGMIX_Tpsf_2P",      50, (0.0, 1.0),  False),
    ]),

    ("Photo-z (BPZ)", "hist_photoz", [
        ("Z_B",     50, (0.0, 3.5), False),
        ("Z_B_MIN", 50, (0.0, 3.5), False),
        ("Z_B_MAX", 50, (0.0, 3.5), False),
        ("T_B",     50, (0.0, 8.0), False),
    ]),

    ("GAaP magnitudes", "hist_gaap", [
        ("MAG_GAAP_0p7_u",  50, (15.0, 32.0), False),
        ("MAG_GAAP_0p7_g",  50, (15.0, 32.0), False),
        ("MAG_GAAP_0p7_r",  50, (15.0, 32.0), False),
        ("MAG_GAAP_0p7_i",  50, (15.0, 32.0), False),
        ("MAG_GAAP_0p7_z",  50, (15.0, 32.0), False),
        ("MAG_GAAP_0p7_z2", 50, (15.0, 32.0), False),
    ]),
]

# ------------
# make output directory
# ------------
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build flat processor list
# ---------------------------------------------------------------------------

cut_counter = CutCounter(cut_defs)

# collect all unique histogram fields and build the hist processors
ALL_HIST_FIELDS = []
ranges = {}
logs   = {}
bins   = {}
for _, _, fields in HISTOGRAM_GROUPS:
    for (f, b, rng, log) in fields:
        if f not in ALL_HIST_FIELDS:
            ALL_HIST_FIELDS.append(f)

        bins[f] = b
        if rng is not None:
            ranges[f] = rng
        if log:
            logs[f] = True
hist_processor_masked = HistogramGroup(
    fields=ALL_HIST_FIELDS,
    bins=bins,
    ranges=ranges,
    logs=logs,
    masked=True,
    tag="_withcuts",
)
hist_processor_nocuts = HistogramGroup(
    fields=ALL_HIST_FIELDS,
    bins=bins,
    ranges=ranges,
    logs=logs,
    masked=False,
    tag="_nocuts",
)

# build the healpix map processors
nside = 512
hp_stats_masked = HealpixStats(
    nside=nside,
    fields=HEALPIX_FIELDS,
    masked=True,
    tag="_withcuts"
)
hp_stats_nocuts = HealpixStats(
    nside=nside, 
    fields=HEALPIX_FIELDS,
    masked=False,
    tag="_nocuts"
)

processors = [cut_counter, hist_processor_masked, hist_processor_nocuts, hp_stats_masked, hp_stats_nocuts]

# ---------------------------------------------------------------------------
# select mpi or serial
# ---------------------------------------------------------------------------

if use_mpi:
    chunk_runner = chunk_runner_mpi
    summarize = summarize_mpi
else:
    chunk_runner = chunk_runner_serial
    summarize = summarize_serial

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

chunk_runner(
    cat_path   = CAT_PATH,
    cache_path = CACHE_PATH,
    cut_defs   = cut_defs,
    processors = processors,
    chunk_size = chunk_size,
    nchunks    = nchunks,       # set e.g. nchunks=5 for a quick test, None for full run
    resume     = resume,
)

summarize(CACHE_PATH, processors, output_dir=output_dir)

# ---------------------------------------------------------------------------
# plots and postprocessing only run on rank=0
# ---------------------------------------------------------------------------

if use_mpi: 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
else:
    rank = 0
    
if rank == 0:
    
    # ---------------------------------------------------------------------------
    # Print output of cut counter
    # ---------------------------------------------------------------------------
    
    cut_counter.print_summary()
    
    # ---------------------------------------------------------------------------
    # Plot — histograms: one figure per group
    # ---------------------------------------------------------------------------
    
    def plot_group(hist_proc, title, stem, fields, prefix="hist", plot_dir="./"):
        n = len(fields)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3 * nrows),
            constrained_layout=True
        )
    
        axes = [axes] if n == 1 else list(axes.flat)
    
        for ax, (field, *_) in zip(axes, fields):
            hist_proc.plot(
                field,
                ax=ax,
                log_y=False,
                color="steelblue",
                edgecolor="none",
            )
    
        for ax in axes[n:]:
            ax.set_visible(False)

        title += f" {hist_proc.tag}"
        fig.suptitle(title, fontsize=12, fontweight="bold")
        outfile = plot_dir + f"/{prefix}_{stem}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
    
        print(f"[plot] saved {outfile}")
    
    histograms_computed = np.array([isinstance(p,Histogram) or isinstance(p,HistogramGroup) for p in processors]).any()
    if histograms_computed:
        for (title, stem, fields) in HISTOGRAM_GROUPS:
            plot_group(hist_processor_masked, title, stem, fields, "hist_withcuts", output_dir)
            plot_group(hist_processor_nocuts, title, stem, fields, "hist_nocuts", output_dir)
    
    # ---------------------------------------------------------------------------
    # Plot — healpix maps
    # ---------------------------------------------------------------------------
    
    def plot_healpix_means(
        hp_stats,
        fields,
        prefix="hp_mean",
        ra_range=None,   # (ra_min, ra_max) in degrees
        dec_range=None,  # (dec_min, dec_max) in degrees
        rotate=False,
         plot_dir="./"
    ):
        fields = ["counts"] + fields
        for field in fields:
            if field == "counts":
                m = hp_stats.get_counts_map()
            else:
                m = hp_stats.get_mean_map(field)
    
            if (m==hp.UNSEEN).all(): #map is empty
                print(f"[plot_healpix_means] Map {field} is empty")
                continue
    
            vmin = np.nanpercentile(m[m!=hp.UNSEEN], 1)
            vmax = np.nanpercentile(m[m!=hp.UNSEEN], 99)
    
            if rotate:
                rot=[-180, 0]
            else:
                rot=None
    
            if ra_range is not None and dec_range is not None:
                if rotate:
                    lonra = [ra_range[0]-180, ra_range[1]-180]
                else:
                    lonra = [ra_range[0], ra_range[1]]
                latra = [dec_range[0], dec_range[1]]
    
                hp.cartview(
                    m,
                    lonra=lonra,
                    latra=latra,
                    rot=rot,
                    title=f"Mean {field} {hp_stats.tag}",
                    unit=field,
                    cmap="viridis",
                    min=vmin,
                    max=vmax,
                )
            else:
                hp.mollview(
                    m,
                    title=f"Mean {field} {hp_stats.tag}",
                    rot=rot,
                    unit=field,
                    cmap="viridis",
                    min=vmin,
                    max=vmax,
                )
    
            outfile = plot_dir + f"/{prefix}_{field}.png"
            plt.savefig(outfile, dpi=150)
            plt.close()
            print(f"[plot] saved {outfile}")
    
    ra_range = [90,300]
    dec_range = [20,90]
    plot_healpix_means(
        hp_stats_masked, 
        HEALPIX_FIELDS, 
        prefix="hp_mean_withcuts_zoom", 
        ra_range=ra_range, 
        dec_range=dec_range, 
        rotate=True,
        plot_dir=output_dir, 
    )
    plot_healpix_means(
        hp_stats_nocuts, 
        HEALPIX_FIELDS, 
        prefix="hp_mean_nocuts_zoom", 
        ra_range=ra_range, 
        dec_range=dec_range, 
        rotate=True,
        plot_dir=output_dir, 
    )
    ra_range = None
    dec_range = None
    plot_healpix_means(
        hp_stats_masked, 
        HEALPIX_FIELDS, 
        prefix="hp_mean_withcuts", 
        ra_range=ra_range, 
        dec_range=dec_range, 
        rotate=True,
        plot_dir=output_dir, 
    )
    plot_healpix_means(
        hp_stats_nocuts, 
        HEALPIX_FIELDS, 
        prefix="hp_mean_nocuts", 
        ra_range=ra_range, 
        dec_range=dec_range, 
        rotate=True,
        plot_dir=output_dir, 
    )

    make_slides(output_dir)

    
    print("\nAll done.")