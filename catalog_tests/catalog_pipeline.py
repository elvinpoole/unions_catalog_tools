"""
catalog_pipeline.py
-------------------
Chunk-based processing pipeline for large HDF5 catalogs.

Main entry points:
    chunk_runner(...)      - run processors over the full catalog
    summarize(...)         - reduce chunk results into final summaries
    extract_columns(...)   - stream cut-passing rows to a new HDF5

Processor interface (subclass BaseProcessor):
    name                   - str, used as HDF5 group key in cache
    process(x, mask)       - called per chunk; returns dict of arrays/scalars
    reduce(chunk_results)  - called once; receives list of dicts, one per chunk
"""

import numpy as np
import h5py
from abc import ABC, abstractmethod
from pathlib import Path
import healpy as hp

# ---------------------------------------------------------------------------
# Base processor
# ---------------------------------------------------------------------------

class BaseProcessor(ABC):
    """
    Subclass this to define a new per-chunk computation.

    process(x, mask) should be cheap and return only small arrays or scalars
    — never a full-dataset-length array.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string key; used as the HDF5 group name in the cache file."""
        ...

    @abstractmethod
    def process(self, x: np.ndarray, mask: np.ndarray) -> dict:
        """
        Process one chunk.

        Parameters
        ----------
        x    : structured numpy array, one chunk of raw catalog data
        mask : boolean array, shape (len(x),), True = passes all cuts

        Returns
        -------
        dict mapping str -> np.ndarray or scalar.
        All values must be numpy-serialisable.
        """
        ...

    @abstractmethod
    def reduce(self, chunk_results: list[dict]):
        """
        Combine results from all chunks into a final summary.

        Parameters
        ----------
        chunk_results : list of dicts returned by process(), one per chunk,
                        in chunk order.

        Returns
        -------
        Whatever makes sense for this processor (stored on self for later use).
        """
        ...


# ---------------------------------------------------------------------------
# Concrete processor: CutCounter
# ---------------------------------------------------------------------------

class CutCounter(BaseProcessor):
    """
    Counts how many objects pass each individual cut, and how many pass all.

    Parameters
    ----------
    cut_defs : list of (name: str, func: callable)
        Each func takes a chunk array x and returns a boolean mask.
    """

    def __init__(self, cut_defs: list):
        self.cut_defs = cut_defs
        self.summary_ = None

    @property
    def name(self) -> str:
        return "CutCounter"

    def process(self, x: np.ndarray, mask: np.ndarray) -> dict:
        n = len(x)
        counts = np.zeros(len(self.cut_defs), dtype=np.int64)
        for i, (_, func) in enumerate(self.cut_defs):
            counts[i] = np.count_nonzero(func(x))
        return {
            "individual_counts": counts,
            "all_pass_count":    np.int64(np.count_nonzero(mask)),
            "total_seen":        np.int64(n),
        }

    def reduce(self, chunk_results: list[dict]):
        total_seen        = sum(r["total_seen"]        for r in chunk_results)
        all_pass          = sum(r["all_pass_count"]    for r in chunk_results)
        individual_counts = sum(r["individual_counts"] for r in chunk_results)

        self.summary_ = {
            "cut_names":         [name for name, _ in self.cut_defs],
            "individual_counts": individual_counts,
            "all_pass_count":    all_pass,
            "total_seen":        total_seen,
        }
        return self.summary_

    def print_summary(self):
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")
        s = self.summary_
        N = s["total_seen"]
        print("\n=== CutCounter Summary ===")
        for name, count in zip(s["cut_names"], s["individual_counts"]):
            print(f"  {name:35s}: {count:10d} / {N}   ({100*count/N:.4f}% pass)")
        print(f"  {'ALL CUTS':35s}: {s['all_pass_count']:10d} / {N}   "
              f"({100*s['all_pass_count']/N:.4f}% pass)")


# ---------------------------------------------------------------------------
# Concrete processor: Histogram
# ---------------------------------------------------------------------------

class Histogram(BaseProcessor):
    """
    Accumulates a histogram of one field for objects that pass all cuts.

    Parameters
    ----------
    field   : str               - column name in the catalog
    bins    : int or array-like - passed to np.histogram
    range   : (float, float) or None - passed to np.histogram
    masked  : bool              - if True (default), histogram only masked rows
    log     : bool              - if True, take log10 of values before binning
                                  (useful for flux/size columns spanning decades)
    """

    def __init__(self, field: str, bins: int = 50, range=None,
                 masked: bool = True, log: bool = False):
        self.field  = field
        self.bins   = bins
        self.range  = range
        self.masked = masked
        self.log    = log
        self.summary_ = None

    @property
    def name(self) -> str:
        return f"Histogram_{self.field}"

    def process(self, x: np.ndarray, mask: np.ndarray) -> dict:
        values = x[self.field][mask] if self.masked else x[self.field]
        values = values.astype(np.float64)
        if self.log:
            values = np.log10(values[values > 0])
        counts, edges = np.histogram(values, bins=self.bins, range=self.range)
        return {
            "counts": counts.astype(np.int64),
            "edges":  edges.astype(np.float64),
        }

    def reduce(self, chunk_results: list[dict]):
        total_counts = sum(r["counts"] for r in chunk_results)
        edges = chunk_results[0]["edges"]
        self.summary_ = {"counts": total_counts, "edges": edges}
        return self.summary_

    def plot(self, ax=None, log_y: bool = False, **kwargs):
        import matplotlib.pyplot as plt
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")
        counts  = self.summary_["counts"]
        edges   = self.summary_["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        if ax is None:
            _, ax = plt.subplots()
        ax.bar(centers, counts, width=np.diff(edges), **kwargs)
        xlabel = f"log10({self.field})" if self.log else self.field
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.set_title(self.field, fontsize=9)
        if log_y:
            ax.set_yscale("log")
        return ax

# ---------------------------------------------------------------------------
# Concrete processor: Healpix statistics
# ---------------------------------------------------------------------------

class HealpixStats(BaseProcessor):
    """
    Accumulates per-healpix statistics:
        - counts per pixel
        - sum of selected fields per pixel

    Parameters
    ----------
    nside  : int
        Healpix nside
    fields : list[str]
        Columns to accumulate sums for
    masked : bool
        If True, only use rows passing mask
    lonlat : bool
        If True, assumes RA/Dec in degrees
    """

    def __init__(self, nside: int, fields: list[str],
                 masked: bool = True, lonlat: bool = True,
                tag: str = ""):
        self.nside  = nside
        self.fields = fields
        self.masked = masked
        self.lonlat = lonlat
        self.tag = tag
        
        self.npix = hp.nside2npix(nside)
        self.summary_ = None

    @property
    def name(self) -> str:
        return f"HealpixStats_nside{self.nside}{self.tag}"

    def process(self, x: np.ndarray, mask: np.ndarray) -> dict:
        if self.masked:
            x = x[mask]
    
        if len(x) == 0:
            return {
                "pix": np.array([], dtype=np.int64),
                "count": np.array([], dtype=np.int64),
                **{f"sum_{f}": np.array([], dtype=np.float64)
                   for f in self.fields}
            }
    
        ra  = x["RA"]
        dec = x["Dec"]
    
        pix = hp.ang2pix(self.nside, ra, dec, lonlat=self.lonlat)
    
        # compress to unique pixels
        unique_pix, inv = np.unique(pix, return_inverse=True)
    
        count = np.bincount(inv).astype(np.int64)
    
        result = {
            "pix": unique_pix,
            "count": count,
        }
    
        for f in self.fields:
            vals = x[f].astype(np.float64)
            sums = np.bincount(inv, weights=vals)
            result[f"sum_{f}"] = sums
    
        return result

    def reduce(self, chunk_results: list[dict]):
        # We only make the full healpix array here
        # We have 1 full healpix array per field
        total_count = np.zeros(self.npix, dtype=np.int64)
        total_sums  = {
            f: np.zeros(self.npix, dtype=np.float64)
            for f in self.fields
        }
    
        for r in chunk_results:
            pix   = r["pix"]
            count = r["count"]
    
            total_count[pix] += count
    
            for f in self.fields:
                total_sums[f][pix] += r[f"sum_{f}"]
    
        self.summary_ = {
            "count": total_count,
            "sums": total_sums,
        }
        return self.summary_

    def get_mean_map(self, field: str):
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")

        count = self.summary_["count"]
        sums  = self.summary_["sums"][field]

        mean = np.zeros_like(sums, dtype=np.float64)
        mask = count > 0
        mean[mask] = sums[mask] / count[mask]
        mean[~mask] = np.nan

        return mean


# ---------------------------------------------------------------------------
# HDF5 cache helpers (private)
# ---------------------------------------------------------------------------

def _chunk_key(ichunk: int) -> str:
    return f"chunk_{ichunk:06d}"


def _save_chunk_result(cache_file: h5py.File, group_name: str,
                       ichunk: int, result: dict):
    key = _chunk_key(ichunk)
    grp = cache_file.require_group(f"{group_name}/{key}")
    for k, v in result.items():
        v = np.asarray(v)
        if k in grp:
            del grp[k]
        grp.create_dataset(k, data=v)


def _load_chunk_result(cache_file: h5py.File, group_name: str,
                       ichunk: int) -> dict:
    key = _chunk_key(ichunk)
    grp = cache_file[f"{group_name}/{key}"]
    return {k: grp[k][()] for k in grp}


def _save_mask_chunk(cache_file: h5py.File, ichunk: int, mask: np.ndarray):
    key = _chunk_key(ichunk)
    grp = cache_file.require_group("mask")
    if key in grp:
        del grp[key]
    grp.create_dataset(key, data=mask, compression="gzip", compression_opts=4)


def _completed_chunks(cache_file: h5py.File) -> set:
    if "meta/completed_chunks" not in cache_file:
        return set()
    return set(cache_file["meta/completed_chunks"][()])


def _mark_chunk_complete(cache_file: h5py.File, ichunk: int):
    completed = _completed_chunks(cache_file)
    completed.add(ichunk)
    arr = np.array(sorted(completed), dtype=np.int64)
    if "meta/completed_chunks" in cache_file:
        del cache_file["meta/completed_chunks"]
    cache_file.create_dataset("meta/completed_chunks", data=arr)
    cache_file.flush()


# ---------------------------------------------------------------------------
# chunk_runner
# ---------------------------------------------------------------------------

def chunk_runner(
    cat_path:     str,
    cache_path:   str,
    cut_defs:     list,
    processors:   list[BaseProcessor],
    chunk_size:   int  = 1_000_000,
    nchunks:      int  = None,
    resume:       bool = True,
    dataset_name: str  = "data",
    verbose:      bool = True,
):
    cache_path = Path(cache_path)

    if not resume and cache_path.exists():
        cache_path.unlink()
        if verbose:
            print(f"[runner] resume=False — deleted existing cache {cache_path}")
    elif not resume and not cache_path.exists():
        if verbose:
            print(f"[runner] running from scratch (resume=False)")
    elif resume and cache_path.exists():
        if verbose:
            print(f"[runner] running with resume=True using cache from {cache_path}")
    elif resume and not cache_path.exists():
        if verbose:
            print(f"[runner] resume=True but no cache file was found")

    with h5py.File(cat_path, "r") as cat, \
         h5py.File(cache_path, "a") as cache:

        dset = cat[dataset_name]
        N    = dset.shape[0]

        ichunk = 0
        for start in range(0, N, chunk_size):
            if nchunks is not None and ichunk >= nchunks:
                break

            end = min(start + chunk_size, N)
            key = _chunk_key(ichunk)

            procs_needed = [
                p for p in processors
                if f"{p.name}/{key}" not in cache
            ]

            if not procs_needed:
                if verbose:
                    print(f"[runner] Chunk {ichunk:6d} — skipping (all processors cached)")
                ichunk += 1
                continue

            x = dset[start:end]

            if f"mask/{key}" in cache:
                mask = cache[f"mask/{key}"][()]
            else:
                cut_masks = [func(x) for _, func in cut_defs]
                mask      = np.logical_and.reduce(cut_masks)
                _save_mask_chunk(cache, ichunk, mask)

            for proc in procs_needed:
                result = proc.process(x, mask)
                _save_chunk_result(cache, proc.name, ichunk, result)

            all_done = all(
                f"{p.name}/{key}" in cache
                for p in processors
            )
            if all_done:
                _mark_chunk_complete(cache, ichunk)

            if verbose:
                n_pass = np.count_nonzero(mask)
                print(f"[runner] Chunk {ichunk:6d}  rows {start}:{end}  "
                      f"pass={n_pass}/{end-start} "
                      f"({100*n_pass/(end-start):.2f}%)  "
                      f"ran {len(procs_needed)}/{len(processors)} processor(s)")

            ichunk += 1

    if verbose:
        print(f"[runner] Done. Cache written to {cache_path}")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def summarize(
    cache_path: str,
    processors: list[BaseProcessor],
    verbose:    bool = True,
):
    """
    Load all cached chunk results and call reduce() on each processor.
    """
    with h5py.File(cache_path, "r") as cache:
        completed = sorted(_completed_chunks(cache))
        if not completed:
            raise RuntimeError("No completed chunks found in cache.")

        for proc in processors:
            chunk_results = [
                _load_chunk_result(cache, proc.name, i) for i in completed
            ]
            proc.reduce(chunk_results)
            if verbose:
                print(f"[summarize] {proc.name} reduced over {len(completed)} chunks.")


# ---------------------------------------------------------------------------
# extract_columns
# ---------------------------------------------------------------------------

def extract_columns(
    cat_path:     str,
    cache_path:   str,
    columns:      list[str],
    out_path:     str,
    chunk_size:   int  = 1_000_000,
    dataset_name: str  = "data",
    verbose:      bool = True,
):
    """
    Stream cut-passing rows into a new HDF5 file, never loading full arrays.

    Parameters
    ----------
    cat_path     : source catalog HDF5
    cache_path   : cache HDF5 produced by chunk_runner (contains masks)
    columns      : list of column names to extract
    out_path     : output HDF5 path (overwritten if exists)
    chunk_size   : must match the chunk_size used in chunk_runner
    dataset_name : HDF5 dataset name inside cat_path
    """
    out_path = Path(out_path)

    with h5py.File(cat_path, "r") as cat, \
         h5py.File(cache_path, "r") as cache, \
         h5py.File(out_path, "w") as out:

        dset      = cat[dataset_name]
        N         = dset.shape[0]
        completed = sorted(_completed_chunks(cache))

        out_dsets = {}
        for col in columns:
            sample = dset[0:1][col]
            out_dsets[col] = out.create_dataset(
                col,
                shape=(0,) + sample.shape[1:],
                maxshape=(None,) + sample.shape[1:],
                dtype=sample.dtype,
                compression="gzip",
                compression_opts=4,
            )

        total_written = 0
        for ichunk in completed:
            start = ichunk * chunk_size
            end   = min(start + chunk_size, N)
            x     = dset[start:end]
            mask  = cache[f"mask/{_chunk_key(ichunk)}"][()]

            n_pass = np.count_nonzero(mask)
            if n_pass == 0:
                continue

            for col in columns:
                ds = out_dsets[col]
                ds.resize(total_written + n_pass, axis=0)
                ds[total_written : total_written + n_pass] = x[col][mask]

            total_written += n_pass

        if verbose:
            print(f"[extract] Wrote {total_written} rows × {len(columns)} columns "
                  f"to {out_path}")