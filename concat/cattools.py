"""
Functions and Classes that are useful for finding and concatenating files from the UNIONS survey.
"""

import fitsio as fio
import h5py 
import os
import time
import numpy as np 
import gc
import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u
import dask
import dask.array as da

example_tile = "UNIONS.316.239"
test_mode_max_tiles_per_batch = 8
test_mode_max_batches = 2

class ConCat:
    """
    A class to concatenate catalog data from UNIONS survey tiles into a single HDF5 file.

    Three primary files will be produced:
    - main catalog 
        - contains external links to the two files below, 
          plus an indexing group for matching between the two catalogs
    - photometry catalog 
        - All of the individual photometry tiles concatenated into one hdf5 file
        - native column names
    - shear catalog 
        - a ShapePipe catalog converted from fits to hdf5
        - native column names

    Can also produce files to be run through the DESC 2pt pipeline (TXPipe)
    All use external links to the above files where possible and TXPipe colnames:
    - TX shear catalog
    - TX photo-z catalog
    - TX photometry catalog
    """

    def __init__(self, bands, maxn, config, verbose=False):
        """
        Initializes the ConCat object by reading config and preparing output filename

        Args:
            bands (str): Photometric bands to use, e.g., 'ugri' or 'ugriz'.
            maxn (int or str): Maximum number of objects expected across all photometry catalogs, or 'auto' to figure it out automatically.
            config (dict): Configuration loaded from a YAML file
        """
        self.bands = bands
        self.maxn = maxn
        
        self.config = config
        self.phot_base_path = self.config.get("phot_base_path")
        self.phot_input_file = self.config.get("phot_input_file")
        self.shear_input_file = self.config.get("shear_input_file")
        self.phot_output_label = self.config.get("phot_output_label")
        self.shear_output_label = self.config.get("shear_output_label")
        self.run_label = self.config.get("run_label")
        self.selection = config.get('selection_label')
        self.test_mode = config.get('test_mode')
        self.do_phot_file = config.get('do_phot_file')
        self.do_shear_file = config.get('do_shear_file')
        self.do_main_file = config.get('do_main_file')
        self.do_txpipe_files = config.get('do_txpipe_files')
        self.parallel = config.get('parallel')
        self.use_batches = config.get('use_batches', True)
        self.batch_size = config.get('batch_size', 1000)
        if self.test_mode:
            print(f'Running in TEST MODE')


        self.missing_tiles = []

        self.tile_list = os.listdir(self.phot_base_path)
        print(f"{len(self.tile_list)} tiles found")

        self.today = str(datetime.date.today())
        
        self.make_file_names()
        
        if self.do_phot_file:
            #get the dtypes of the columns from an example tile
            f_example = fio.FITS(self.phot_base_path +"/"+ example_tile +"/"+ self.phot_input_file.format(tile=example_tile, bands=self.bands))
            self.example_dtype = f_example[1].get_rec_dtype()

        self.verbose = verbose

    def make_file_names(self):
        """
        If a stage is set to run, generate it's filename
        If a stage is not set to run look for an input name
        If no input name, use generated one
        """
        #generated file names (including today's date)
        phot_output_file = f"./{self.run_label}_PHOT_{self.phot_output_label}_"+self.phot_input_file.format(tile=self.today ,bands=self.bands) + ".h5"
        shear_output_file = f"./{self.run_label}_SHEAR_{self.shear_output_label}_{self.today}.h5"
        main_output_file = f"./{self.run_label}_MAIN_{self.shear_output_label}_{self.phot_output_label}_"+self.phot_input_file.format(tile=self.today ,bands=self.bands) + ".h5"

        if self.do_phot_file:
            self.phot_output_file = phot_output_file
        else:
            self.phot_output_file = self.config.get("input_phot_file", phot_output_file )
            
        if self.do_shear_file:
            self.shear_output_file = shear_output_file
        else:
            self.shear_output_file = self.config.get("input_shear_file", shear_output_file )
            
        if self.do_main_file:
            self.main_output_file = main_output_file
        else:
            self.main_output_file = self.config.get("input_main_file", main_output_file )
    
    def run(self):
        """
        Execute the pipeline
        """
        phot_cols,sp_cols = self.make_output_col_list()

        #concatenate the individual tile files, making basic cuts
        if self.do_phot_file:
            print(f'Will output photometry catalog to {self.phot_output_file}')

            if self.maxn == "auto":
                self.maxn = self.auto_compute_maxn(verbose=self.verbose)
                self.save_missing_tile_file()
            else:
                self.missing_tiles = []
            
            if self.parallel:
                self.select_and_save_photometry_parallel(phot_cols, use_batches=self.use_batches, batch_size=self.batch_size)
            else:   
                self.select_and_save_photometry_lowmem(phot_cols, verbose=self.verbose)
        
        else:
            print(f'Not generating photometry catalog')

        if self.do_shear_file:
            print(f'Will output shear catalog to {self.shear_output_file}')
            self.save_shear_catalog(sp_cols)
        else:
            print(f'Not generating shear catalog')
            
        if self.do_main_file:
            print(f'Will output main catalog to {self.main_output_file}')
            self.save_main_catalog()
        else:
            print(f'Not generating main catalog')
        
        # This will additionally save the files in the format needed to run through TXPipe (DESC 2pt pipeline)
        # TXPipe needs 3 files, photometry, photo-z and shear 
        if self.do_txpipe_files:
            print(f'Saving TXPipe catalogs with external links')
            self.save_txpipe_cats()

    def make_output_col_list(self):
        """
        Constructs the list of columns to extract from each tile catalog

        The columns to be saved are currently hardcoded and depend on 
        the selection and bands requested.
        
        Returns:
            cols (list): List of column names to extract from the photometry catalogs.
            sp_cols (list): List of column names to extract from the ShapePipe catalog
        """
        cols_basic = ["ALPHA_J2000", "DELTA_J2000", "MAG_AUTO", "Z_B", "Flag", "Tilename"]
        cols_ap_band = ["MAG_GAAP{ap}_{band}", "MAGERR_GAAP{ap}_{band}",]
        cols_band = ["MAG_LIM_{band}"]

        if self.bands == "ugriz":
            band_list = ["u", "g", "r", "i", "z", "z2"]
        elif self.bands == "ugri":
            band_list = ["u", "g", "r", "i"]

        if self.selection == "all":
            ap_list = [""] #when we select "all" objects i dont want to save the other apature cos it makes teh file too big
            cols_ap_band.append("FLAG_GAAP{ap}_{band}") #but i do want to save the flags if we are saving everything
            cols_band = [] # I also dont need the maglims for teh all selection
        else:
            ap_list = ["", "_0p7", "_1p0"]

        cols = []
        cols += cols_basic
        for band in band_list:
            for c in cols_band:
                cols += [c.format(band=band)]
            for ap in ap_list:  # Which GAaP apertures to include
                for c in cols_ap_band:
                    cols += [c.format(ap=ap, band=band)]
        
        print(f'Selecting {len(cols)} photometry columns: {cols}')

        sp_cols = ['RA', 'Dec', 'w_iv', 'mag', 'e1', 'e2', 'snr', 'e1_uncal', 'e2_uncal', 'FLUX_RADIUS', 'FWHM_IMAGE', 'FWHM_WORLD', 'MAGERR_AUTO', 'MAG_WIN', 'MAGERR_WIN', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'e1_leak_corrected', 'e2_leak_corrected']

        print(f'Selecting {len(sp_cols)} shear columns: {sp_cols}')
        
        return cols, sp_cols

    def auto_compute_maxn(self, verbose=False):
        """
        Compute the total number of objects in all of the photometry files

        We do this by loading the catalog file from each tile and getting just the nrows value
        
        If the files have to be read from disk the IO can be a little slow

        Returns:
            maxn: sum of nrows from each file
        """
        deltas = [] #timing info
        self.nrows_tot = [] #number of rows in each file
        for i, tile in enumerate(self.tile_list):
            if self.test_mode:
                if i >= test_mode_max_tiles_per_batch:
                    break
            if verbose and i%200 == 0 and i!=0 :
                print(f"{i}/{len(self.tile_list)} approx time remaining {time_left}s, mean of last 100 nrows {np.round(np.mean(self.nrows_tot[-99:]),1)}")
            
            s = time.time()
            filename = self.phot_input_file.format(tile=tile, bands=self.bands)
            filepath = self.phot_base_path +"/"+ tile +"/"+ filename
        
            #check that file exists
            if not os.path.exists(filepath):
                print(f"tile {tile} has no catalog")
                self.nrows_tot.append(0)
                self.missing_tiles.append(tile)
                continue
        
            with fio.FITS(filepath) as f:
                self.nrows_tot.append(f[1].get_nrows())
        
            e = time.time()
            delta = e-s
            deltas.append(delta)
            time_left = np.round((len(self.tile_list)-i)*np.mean(deltas),1)

        print(f"{np.sum(self.nrows_tot)} objects found")
        return np.sum(self.nrows_tot)

    def save_missing_tile_file(self):
        missing_tile_file = open(f"{self.run_label}_MISSINGTILES_{self.phot_output_label}_" + self.phot_input_file.format(tile="missing_tiles",bands=self.bands)+'.txt', 'w')
        missing_tile_file.write('\n'.join(self.missing_tiles))
        missing_tile_file.close()

    def select_and_save_photometry_lowmem(self, cols, verbose=False,):
        """
        Loops through all the catalog files (one for each tile)
        and saves "cols" to the hdf5 file

        Optimized for low memory use
        This is not parralelized and will be slow
        """

        self.nrows_masked = [] #number of objects from each file after masking
        self.nrows_unmasked = []

        #open the output photometry file
        with h5py.File(self.phot_output_file, "w") as output:
        
            #make an empty hdf5 group to save the catalogs to 
            grp = output.create_group("photometry")
            
            # make empty data sets for each column
            # length of data sets should be >= number of objects in final catalog
            for c in cols:
                if c=="Tilename":
                    string_type = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset(c, shape=(self.maxn,), maxshape=(None,), dtype=string_type)
                else:
                    col_dtype = self.example_dtype[0][c] #get from f_example
                    grp.create_dataset(c, shape=(self.maxn,), maxshape=(None,), dtype=col_dtype)        # 64-bit float
        
            deltas = [] #for timing info
            for i, tile in enumerate(self.tile_list):
                if self.test_mode:
                    if i >= test_mode_max_tiles_per_batch:
                        break
                if verbose and i%200 == 0 and i!=0 :
                    print(f"{i}/{len(self.tile_list)} approx time remaining {time_left}s, mean of last 100 nrows {np.round(np.mean(self.nrows_tot[-99:]),1)}")
                
                s = time.time()
                filename = self.phot_input_file.format(tile=tile, bands=self.bands)
                filepath = os.path.join(self.phot_base_path, tile, filename)
            
                #check that file exists
                if tile in self.missing_tiles:
                    print(f"skipping tile {tile} has no catalog")
                    continue
            
                with fio.FITS(filepath) as f:
                    if i == 0:
                        print('Columns found in first phot fits file')
                        print(f[1]._colnames)
                    mask = self.get_mask(f[1], self.selection)
                    nrows1 = np.sum(mask.astype('int'))
                    self.nrows_masked.append(nrows1)
                    self.nrows_unmasked.append(len(mask))
                    
                    start = sum(self.nrows_masked[:i])
                    end = sum(self.nrows_masked[:i+1])
                    for c in cols:
                        if c=="Tilename":
                            grp[c][start:end] = tile
                        else:
                            grp[c][start:end] = f[1][c].read()[mask]

                #timing stuff
                e = time.time()
                delta = e-s
                deltas.append(delta)
                time_left = np.round((len(self.tile_list)-i)*np.mean(deltas),1)

            #trim the excess from each column
            self.ntot_masked = np.sum(self.nrows_masked)
            for c in cols:
                grp[c].resize((self.ntot_masked,))

            # Add an ID column
            # Call it "thisfile" to make it clear this is not a universal ID
            grp.create_dataset("ID_thisfile", data=np.arange(self.ntot_masked) )

        #output.close()
        
        self.ntot_unmasked = np.sum(self.nrows_unmasked)
        print(f'{self.ntot_unmasked} objects total')
        print(f'{self.ntot_masked} objects passed the selected cuts')

    def select_and_save_photometry_parallel(self, cols, verbose=False, use_batches=False, batch_size=1000):
        """
        Loops through all the catalog files (one for each tile)
        and saves "cols" to the hdf5 file

        This is an attempt to parralelize the processing with dask 
        (memory usage will be much larger)

        use_batches: 
            will multiprocess the first <batch_size> tiles
            and save to file. then repeat for each batch
            batches reduce max memory usage
        
        """

        #open the output photometry file and set up the columns
        with h5py.File(self.phot_output_file, "w") as output:
            #make an empty hdf5 group to save the catalogs to 
            grp = output.create_group("photometry")

            # make empty data sets for each column
            # length of data sets should be >= number of objects in final catalog
            for c in cols:
                if c=="Tilename":
                    string_type = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset(c, shape=(self.maxn,), maxshape=(None,), dtype=string_type)
                else:
                    col_dtype = self.example_dtype[0][c] #get from f_example
                    grp.create_dataset(c, shape=(self.maxn,), maxshape=(None,), dtype=col_dtype)
        
        # Split the tile list into batches if requested
        if use_batches:
            batches = [self.tile_list[i:i + batch_size] for i in range(0, len(self.tile_list), batch_size)]
        else:
            batches = [self.tile_list]

        # Process each batch
        print(f"Processing tiles in {len(batches)} batches")
        nrows_batch = []
        for ibatch, batch_tile_list in enumerate(batches):
            print(f'batch {ibatch}')
            if self.test_mode:
                if ibatch >= test_mode_max_batches:
                    break
            
            ###### Parallelization starts here
            futures = [] #will contain a list of processes to be parallelized over tiles
            for i, tile in enumerate(batch_tile_list):
                if self.test_mode:
                    if i >= test_mode_max_tiles_per_batch:
                        break
                
                filename = self.phot_input_file.format(tile=tile, bands=self.bands)
                filepath = os.path.join(self.phot_base_path, tile, filename)
            
                #check that file exists
                if tile in self.missing_tiles:
                    print(f"skipping tile {tile} has no catalog")
                    continue
    
                futures.append(self._delayed_process_tile(tile, filepath, cols))
    
            #start the dask computation
            print('starting dask computation')
            s = time.time()
            results = dask.compute(*futures)
            e = time.time()
            print(f'Finished dask computation. took {e-s}s')
            
            # Concatenate all column data
            combined = {c: np.concatenate([r[c] for r in results if r[c].size > 0]) for c in cols}  
            
            nrows_batch.append(len(combined[cols[0]]))
            start = sum(nrows_batch[:ibatch])
            end = sum(nrows_batch[:ibatch+1])
            
            assert len(combined[cols[0]]) == end-start

            #add batch to group
            with h5py.File(self.phot_output_file, "r+") as output:
                for c in cols:
                    output[f"photometry/{c}"][start:end] = combined[c]

            #clean up memory
            del combined
            del results

        self.ntot_masked = sum(nrows_batch)
        print(f'{self.ntot_masked} objects passed the selected cuts')

        with h5py.File(self.phot_output_file, "r+") as output: 
            #resize the columns to the masked size
            for c in cols:
                output[f"photometry/{c}"].resize((self.ntot_masked,))
    
            # Add an ID column
            # Call it "thisfile" to make it clear this is not a universal ID
            output["photometry/"].create_dataset("ID_thisfile", data=np.arange(self.ntot_masked) )

    def _process_single_tile(self, tile, filepath, cols):
        """
        Load a fits file and output dict of the requested columns
        """
        try:
            with fio.FITS(filepath) as f:
                data = f[1]
                mask = self.get_mask(data, self.selection)

                rows = data.read_rows(np.where(mask)[0])
                
                out = {}
                for c in cols:
                    if c == "Tilename":
                        out[c] = np.full(sum(mask), tile, dtype='S20')
                    else:
                        out[c] = rows[c]
                del mask
                del rows
                return out
        except Exception as e:
            print(f"Skipping tile {tile} due to error: {e}")
            return {c: np.array([], dtype="f8") for c in cols}

    @dask.delayed
    def _delayed_process_tile(self, tile, filepath, cols):
        """
        Process the tile using dask delay. This method is lazy and 
        will not be executed until we run the compute()
        """
        return self._process_single_tile(tile, filepath, cols)
    
    def save_shear_catalog(self, cols):
        """
        Loads the shear catalog from fits and saves the requested columns to an hdf5 file
        """

        with h5py.File(self.shear_output_file, "w") as output:
            
            #make an empty hdf5 group to save the catalog to 
            grp = output.create_group("shear")

            sp_fits_table = fio.FITS(self.shear_input_file)[-1]

            for c in cols:
                print('shear col', c)
                if self.test_mode:
                    grp.create_dataset(c, data=sp_fits_table[c].read(rows=range(100000)) )
                else:
                    grp.create_dataset(c, data=sp_fits_table[c].read() )

    def save_main_catalog(self):
        """
        Make a main catalog hdf5 file with external links to the phot and shear catalogs
        and an index group that matches the phot and shear catalogs
        """
        with h5py.File(self.main_output_file, "w") as output:
            output['photometry'] = h5py.ExternalLink(self.phot_output_file, 'photometry')
            output['shear'] = h5py.ExternalLink(self.shear_output_file, 'shear')


            #make index arrays that match the two catalogs
            ra1 = output['photometry/ALPHA_J2000'][:]
            dec1 = output['photometry/DELTA_J2000'][:]

            ra2 = output['shear/RA'][:]
            dec2 = output['shear/Dec'][:]

            #match sky coordinates using astropy
            phot_coords = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
            shear_coords = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
            #match photometry catalog to shear
            idx_p2s, d2d_p2s, d3d_p2s = phot_coords.match_to_catalog_sky(shear_coords)
            #match shear catalog to photometry
            idx_s2p, d2d_s2p, d3d_s2p = shear_coords.match_to_catalog_sky(phot_coords)

            grp = output.create_group("index")
            grp.create_dataset("index_phot_to_shear", data=idx_p2s )
            grp.create_dataset("index_shear_to_phot", data=idx_s2p )
            grp.create_dataset("d2d_p2s", data=d2d_p2s.value )
            grp.create_dataset("d2d_s2p", data=d2d_s2p.value )

    def get_mask(self, fits_table, selection):
        """
        Get an array "mask" that corresponds to a particular selection of galaxies

        currently implemented:
            selection=='all'
                all objects
            selection=="flags_and_mags"
                a basic set of cuts that required FLAG_GAAP==0 and MAG_GAAP!=-99 for all bands
        """
        nrows = fits_table.get_nrows()

        if selection == 'all':
            return np.ones(nrows, dtype=bool)
            
        elif selection == 'flags_and_mags':
            # select only objects that have mag_flag=0 in all bands, and mag is not -99
            # for the ugriz catalog, only 1 of z and z2 has to pass (flag and mag)
            
            mask = np.ones(nrows, dtype=bool)

            selection_cols = [f"FLAG_GAAP_{b}" for b in self.bands]
            selection_cols += [f"MAG_GAAP_{b}" for b in self.bands]
            if "ugriz":
                selection_cols += ["FLAG_GAAP_z2", "MAG_GAAP_z2"]
                
            selection_data = fits_table[selection_cols].read()
            
            for band in "ugri":
                mask &= (selection_data[f"FLAG_GAAP_{band}"] == 0)
                mask &= (selection_data[f"MAG_GAAP_{band}"] != -99.) 
            if self.bands == "ugriz":
                mask &= np.logical_or((selection_data["FLAG_GAAP_z"] == 0), (selection_data["FLAG_GAAP_z2"] == 0) )
                mask &= np.logical_or((selection_data["MAG_GAAP_z"] != -99.), (selection_data["MAG_GAAP_z2"] != -99.)) #must have a measurement in either z col

            del selection_data

            return mask

    def save_txpipe_cats(self):
        """
        Save catalogs in TXPipe format
            - TX shear catalog
            - TX photo-z catalog
            - TX photometry catalog
        
        Use external links to the photometry and shear h5 files where possible 
        
        Column names use TXPipe conventions:
        """

        #Photometry file details
        tx_phot_output_file = f"./{self.run_label}_txpipe_photometry_{self.phot_output_label}" + self.phot_input_file.format(tile=self.today ,bands=self.bands) + ".h5"
        phot_col_dict = { #matching unions column names to txpipe column names
            "ALPHA_J2000":"ra", 
            "DELTA_J2000":"dec",
            "MAG_GAAP_u":"u_mag",
            "MAGERR_GAAP_u":"u_mag_err",
            "MAG_GAAP_g":"g_mag",
            "MAGERR_GAAP_g":"g_mag_err",
            "MAG_GAAP_r":"r_mag",
            "MAGERR_GAAP_r":"r_mag_err",
            "MAG_GAAP_i":"i_mag",
            "MAGERR_GAAP_i":"i_mag_err",
        }
        if self.bands == "ugriz":
            phot_col_dict["MAG_GAAP_z"] = "z_mag"
            phot_col_dict["MAGERR_GAAP_z"] = "z_mag_err"

        
        #Photo-z file
        tx_pz_output_file = f"./{self.run_label}_txpipe_photoz_{self.phot_output_label}" + self.phot_input_file.format(tile=self.today ,bands=self.bands) + ".h5"
        pz_col_dict = {
            "Z_B":"zb", 
        }

        #shear cat file
        tx_shear_output_file = f"./{self.run_label}_txpipe_shear_{self.shear_output_label}_{self.today}.h5"
        shear_col_dict = { 
            'RA':'ra',
            'Dec':'dec',
            'w_iv':'weight',
            'e1':'g1',
            'e2':'g2',
            #TODO: add T and SNR columns
        }
        
        with h5py.File(self.main_output_file, "r") as main_out:

            #Photometry file
            with h5py.File(tx_phot_output_file, "w") as tx_phot_out:
                grp = tx_phot_out.create_group("photometry")
                for col in phot_col_dict.keys():
                    if col in ["MAG_GAAP_z", "MAGERR_GAAP_z"]:
                        #For the z-columns we want to join z and z2
                        z1 = main_out[f'photometry/MAG_GAAP_z'][:]
                        z2 = main_out[f'photometry/MAG_GAAP_z2'][:]
                        grp.create_dataset(
                            phot_col_dict[col], 
                            data=np.where(
                                np.logical_or((z2==99.),(z2==-99.)),
                                main_out[f'photometry/{col}'][:],
                                main_out[f'photometry/{col}2'][:]),
                            )
                    else:
                        grp[phot_col_dict[col]] = h5py.ExternalLink(
                            self.phot_output_file, 
                            f'photometry/{col}'
                        )
                        

            #photometry file
            with h5py.File(tx_pz_output_file, "w") as tx_pz_out:
                grp = tx_pz_out.create_group("ancil")
                for col in pz_col_dict.keys():
                    grp[pz_col_dict[col]] = h5py.ExternalLink(
                            self.phot_output_file, 
                            f'photometry/{col}'
                        )

            #shear catlog
            with h5py.File(tx_shear_output_file, "w") as tx_shear_out:
                grp = tx_shear_out.create_group("shear")
                
                for col in shear_col_dict.keys():
                    #grp.create_dataset(
                    #    shear_col_dict[col], 
                    #    data=main_out[f'shear/{col}'][:],
                    #    )
                    grp[shear_col_dict[col]] = h5py.ExternalLink(
                            self.shear_output_file, 
                            f'shear/{col}'
                        )
                
            
                
                
                
                
            
        
            
                
                    
                        