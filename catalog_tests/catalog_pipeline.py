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
from mpi4py import MPI

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
    Single-field histogram processor 
    
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

    def save(self, out_path: str):
        """
        Save bin edges, centers, and counts to an HDF5 file.

        Parameters
        ----------
        out_path : path to output HDF5 file (overwritten if exists)
        """
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")
        counts  = self.summary_["counts"]
        edges   = self.summary_["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        with h5py.File(out_path, "w") as f:
            f.create_dataset("edges",   data=edges)
            f.create_dataset("centers", data=centers)
            f.create_dataset("counts",  data=counts)
            f.attrs["field"]  = self.field
            f.attrs["log"]    = self.log
            f.attrs["masked"] = self.masked


class HistogramGroup(BaseProcessor):
    """
    Multi-field histogram processor

    Parameters
    ----------
    fields : list[str]
        Column names
    bins : int (default for all fields)
    ranges : dict[str, (min, max)] or None
    logs : dict[str, bool] or None
    masked : bool
    """

    def __init__(
        self,
        fields: list[str],
        bins: dict[str, int],
        ranges: dict[str, tuple] | None = None,
        logs: dict[str, bool] | None = None,
        masked: bool = True,
        tag: str = "",
    ):
        self.fields = fields
        self.bins = bins
        self.ranges = ranges or {}
        self.logs = logs or {}
        self.masked = masked
        self.tag = tag

        self.summary_ = None

    @property
    def name(self) -> str:
        return f"HistogramGroup{self.tag}"

    def process(self, x: np.ndarray, mask: np.ndarray) -> dict:
        if self.masked:
            x = x[mask]

        if len(x) == 0:
            return {
                f"{f}_counts": np.zeros(self.bins[f], dtype=np.int64)
                for f in self.fields
            } | {
                f"{f}_edges": np.linspace(0, 1, self.bins[f] + 1)
                for f in self.fields
            }

        result = {}

        for f in self.fields:
            values = x[f].astype(np.float64)

            if self.logs.get(f, False):
                values = values[values > 0]
                values = np.log10(values)

            counts, edges = np.histogram(
                values,
                bins=self.bins[f],
                range=self.ranges.get(f, None),
            )

            result[f"{f}_counts"] = counts.astype(np.int64)
            result[f"{f}_edges"]  = edges.astype(np.float64)

        return result

    def reduce(self, chunk_results: list[dict]):
        summary = {}

        for f in self.fields:
            total_counts = sum(r[f"{f}_counts"] for r in chunk_results)
            edges = chunk_results[0][f"{f}_edges"]

            summary[f] = {
                "counts": total_counts,
                "edges": edges,
            }

        self.summary_ = summary
        return summary

    def plot(self, field: str, ax=None, log_y=False, **kwargs):
        import matplotlib.pyplot as plt

        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")

        data = self.summary_[field]
        counts = data["counts"]
        edges  = data["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])

        if ax is None:
            _, ax = plt.subplots()

        ax.bar(centers, counts, width=np.diff(edges), **kwargs)

        xlabel = f"log10({field})" if self.logs.get(field, False) else field

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.set_title(field, fontsize=9)

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
        mean[~mask] = hp.UNSEEN

        return mean

    def get_counts_map(self) -> np.ndarray:
        """
        Return the counts map as a full healpix array (length npix).
        Empty pixels are set to nan.
        """
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")
        m = self.summary_["count"].astype(np.float64)
        m[m == 0] = hp.UNSEEN
        return m

    @property
    def area_deg2(self) -> float:
        """
        Footprint area in square degrees: pixels with count >= 1 * pixel area.
        """
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")
        n_occupied  = np.count_nonzero(self.summary_["count"] >= 1)
        pix_area    = hp.nside2pixarea(self.nside, degrees=True)
        return n_occupied * pix_area

    def save(self, out_path: str):
        """
        Save the counts map and all mean maps to an HDF5 file.

        Datasets written
        ----------------
        counts/pix    : int64  pixel indices with count >= 1
        counts/values : int64  count values at those pixels
        means/<field>/pix    : int64  pixel indices with count >= 1
        means/<field>/values : float64 mean values at those pixels
        Attributes: nside, tag, area_deg2
        """
        if self.summary_ is None:
            raise RuntimeError("Call reduce() first.")

        occupied = self.summary_["count"] >= 1
        pix      = np.where(occupied)[0].astype(np.int64)

        with h5py.File(out_path, "w") as f:
            f.attrs["nside"]    = self.nside
            f.attrs["tag"]      = self.tag
            f.attrs["area_deg2"] = self.area_deg2

            grp = f.create_group("counts")
            grp.create_dataset("pix",    data=pix)
            grp.create_dataset("values", data=self.summary_["count"][pix])

            for field in self.fields:
                mean = self.get_mean_map(field)
                mgrp = f.require_group(f"means/{field}")
                mgrp.create_dataset("pix",    data=pix)
                mgrp.create_dataset("values", data=mean[pix])


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

def chunk_runner_serial(
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

def chunk_runner_mpi(
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

    assert not resume, "dont use resume with the MPI version"
            
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cache_path = Path(cache_path)
    cache_path_rank = cache_path.with_name(
        f"{cache_path.stem}_rank{rank}.hdf5"
    )

    if cache_path_rank.exists():
        cache_path_rank.unlink()
        if verbose:
            print(f"[runner] resume=False — deleted existing cache {cache_path_rank}")
    else:
        if verbose:
            print(f"[runner] running from scratch (resume=False)")

    if rank == 0 and verbose:
        print(f"[mpi] Running with {size} ranks")

    with h5py.File(cat_path, "r", swmr=True) as cat, \
         h5py.File(cache_path_rank, "a") as cache:

        dset = cat[dataset_name]
        N    = dset.shape[0]

        ichunk = 0
        for start in range(0, N, chunk_size):

            if nchunks is not None and ichunk >= nchunks:
                break

            # distribute chunks across ranks
            if ichunk % size != rank:
                ichunk += 1
                continue

            end = min(start + chunk_size, N)
            key = _chunk_key(ichunk)

            x = dset[start:end]

            # compute mask
            cut_masks = [func(x) for _, func in cut_defs]
            mask      = np.logical_and.reduce(cut_masks)

            _save_mask_chunk(cache, ichunk, mask)

            # run processors
            for proc in processors:
                result = proc.process(x, mask)
                _save_chunk_result(cache, proc.name, ichunk, result)

            if verbose:
                n_pass = np.count_nonzero(mask)
                print(f"[rank {rank}] chunk {ichunk:6d} "
                      f"{start}:{end} "
                      f"pass={n_pass}/{end-start} "
                      f"({100*n_pass/(end-start):.2f}%)")

            ichunk += 1

    comm.Barrier()

    if rank == 0 and verbose:
        print(f"[mpi] Done. Per-rank caches written like: {cache_path.stem}_rank*.hdf5")

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def summarize_serial(
    cache_path: str,
    processors: list[BaseProcessor],
    output_dir: str  = None,
    verbose:    bool = True,
):
    """
    Load all cached chunk results and call reduce() on each processor.

    Parameters
    ----------
    cache_path : HDF5 cache produced by chunk_runner
    processors : list of processors to reduce
    output_dir : if given, call save() on processors that support it and
                 write <output_dir>/<processor.name>.h5 for each
    verbose    : print progress
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

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for proc in processors:
            if not hasattr(proc, "save"):
                continue
            out_path = out_dir / f"{proc.name}.h5"
            proc.save(str(out_path))
            if verbose:
                print(f"[summarize] {proc.name} saved to {out_path}")

def summarize_mpi(
    cache_path: str,
    processors: list[BaseProcessor],
    output_dir: str  = None,
    verbose:    bool = True,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cache_path = Path(cache_path)

    # each rank loads its own cache
    cache_path_rank = cache_path.with_name(
        f"{cache_path.stem}_rank{rank}.hdf5"
    )

    with h5py.File(cache_path_rank, "r") as cache:
        #completed = sorted(_completed_chunks(cache))

        local_results = {}
        for proc in processors:
            #chunk_results = [
            #    _load_chunk_result(cache, proc.name, i) for i in completed
            #]
            
            #####
            if proc.name not in cache:
                local_results[proc.name] = []
                continue
        
            grp = cache[proc.name]
        
            chunk_keys = sorted(grp.keys())  # e.g. chunk_000000, chunk_000001, ...
        
            chunk_results = [
                {k: grp[key][k][()] for k in grp[key]}
                for key in chunk_keys
            ]
            #####
            
            local_results[proc.name] = chunk_results

    # gather all results to rank 0
    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return

    # merge all chunk results
    for proc in processors:
        all_chunk_results = []

        for rank_results in gathered:
            all_chunk_results.extend(rank_results[proc.name])

        proc.reduce(all_chunk_results)

        if verbose:
            print(f"[summarize] {proc.name} reduced over {len(all_chunk_results)} chunks.")

    # save outputs
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for proc in processors:
            if not hasattr(proc, "save"):
                continue
            out_path = out_dir / f"{proc.name}.h5"
            proc.save(str(out_path))
            if verbose:
                print(f"[summarize] {proc.name} saved to {out_path}")


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