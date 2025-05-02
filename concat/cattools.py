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

example_tile = "UNIONS.316.239"

class ConCat:
    """
    A class to concatenate catalog data from UNIONS survey tiles into a single HDF5 file.
    """

    def __init__(self, bands, maxn, config, verbose=False):
        """
        Initializes the ConCat object by reading config and preparing output filename

        Args:
            bands (str): Photometric bands to use, e.g., 'ugri' or 'ugriz'.
            maxn (int or str): Maximum number of tiles to process, or 'auto' to figure it out automatically.
            config (dict): Configuration loaded from a YAML file
            max_tiles (int or None): set to break all the loops after a few tiles (for testing)
        """
        self.bands = "ugriz"
        self.maxn = maxn
        
        self.config = config
        self.base_path = self.config.get("base_path")
        self.cat_file = self.config.get("cat_file")
        self.concat_output_label = self.config.get("concat_output_label")
        self.sp_output_label = self.config.get("sp_output_label")
        self.selection = config.get('selection_label')
        self.max_tiles = config.get('max_tiles')
        self.do_concat_files = config.get('do_concat_files')
        self.do_txpipe_files = config.get('do_txpipe_files')
        if self.max_tiles is not None:
            print(f'Limiting the catalog to the first {self.max_tiles} tiles found (for testing)')

        self.tile_list = os.listdir(self.base_path)
        print(f"{len(self.tile_list)} tiles found")

        self.outfile = f"./{self.concat_output_label}_"+self.cat_file.format(tile=str(datetime.date.today()) ,bands=self.bands) + ".h5"
        print(f'We will output the concatenated file to {self.outfile}')

        #get the dtypes of the columns from an example tile
        f_example = fio.FITS(self.base_path +"/"+ example_tile +"/"+ self.cat_file.format(tile=example_tile, bands=self.bands))
        self.example_dtype = f_example[1].get_rec_dtype()

        self.verbose = verbose
    
    def run(self):
        """
        Execute the concatenation process.
        """
        output_cols = self.make_output_col_list()

        #concatenate the individual tile files, making basic cuts
        if self.do_concat_files:
            if self.maxn == "auto":
                self.maxn = self.auto_compute_maxn(verbose=self.verbose)
                self.save_missing_tile_file()
            else:
                self.missing_tiles = []
            
            self.select_and_save(output_cols, verbose=self.verbose)

        # This will additionally save the files in the format needed to run through TXPipe (DESC 2pt pipeline)
        # TXPipe needs 3 files, photometry, photo-z and shear 
        if self.do_txpipe_files:
            self.save_txpipe_cats()

    def make_output_col_list(self):
        """
        Constructs the list of columns to extract from each tile catalog

        Returns:
            list: List of column names to extract from the input catalogs.
        """
        cols_basic = ["ALPHA_J2000", "DELTA_J2000", "MAG_AUTO", "Z_B", "Flag", "Tilename"]
        cols_ap_band = ["MAG_GAAP{ap}_{band}", "MAGERR_GAAP{ap}_{band}",]
        cols_band = ["MAG_LIM_{band}"]

        if self.bands == "ugriz":
            band_list = ["u", "g", "r", "i", "z", "z2"]
        elif self.bands == "ugri":
            band_list = ["u", "g", "r", "i"]

        cols = []
        cols += cols_basic
        for band in band_list:
            for c in cols_band:
                cols += [c.format(band=band)]
            for ap in ["", "_0p7", "_1p0"]:  # Which GAaP apertures to include
                for c in cols_ap_band:
                    cols += [c.format(ap=ap, band=band)]
        
        print(f'Selecting {len(cols)} columns: {cols}')
        return cols

    def auto_compute_maxn(self, verbose=False):
        """
        Load catalog file from each tile and get nrows
        If the files have to be read from disk the IO can be a little slow

        Returns:
            maxn: sum of nrows from each file
        """
        deltas = [] #timing info
        self.nrows_tot = [] #number of rows in each file
        self.missing_tiles = [] #keep track of the number of missing files
        for i, tile in enumerate(self.tile_list):
            if self.max_tiles is not None:
                if i >= self.max_tiles:
                    break
            if verbose and i%200 == 0 and i!=0 :
                print(f"{i}/{len(self.tile_list)} approx time remaining {time_left}s, mean of last 100 nrows {np.round(np.mean(self.nrows_tot[-99:]),1)}")
            
            s = time.time()
            filename = self.cat_file.format(tile=tile, bands=self.bands)
            filepath = self.base_path +"/"+ tile +"/"+ filename
        
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
        missing_tile_file = open(f"{self.concat_output_label}" + self.cat_file.format(tile="missing_tiles",bands=self.bands)+'.txt', 'w')
        missing_tile_file.write('\n'.join(self.missing_tiles))
        missing_tile_file.close()

    def select_and_save(self, cols, verbose=False,):

        self.nrows_masked = [] #number of objects from each file after masking
        self.nrows_unmasked = []

        #open the file
        with h5py.File(self.outfile, "w") as output:
        
            #make an empty hdf5 group to save the catalogs to 
            grp = output.create_group("catalog")
            
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
                if self.max_tiles is not None:
                    if i >= self.max_tiles:
                        break
                if verbose and i%200 == 0 and i!=0 :
                    print(f"{i}/{len(self.tile_list)} approx time remaining {time_left}s, mean of last 100 nrows {np.round(np.mean(self.nrows_tot[-99:]),1)}")
                
                s = time.time()
                filename = self.cat_file.format(tile=tile, bands=self.bands)
                filepath = self.base_path +"/"+ tile +"/"+ filename
            
                #check that file exists
                if tile in self.missing_tiles:
                    print(f"skipping tile {tile} has no catalog")
                    continue
            
                with fio.FITS(filepath) as f:
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

        self.ntot_unmasked = np.sum(self.nrows_unmasked)
        print(f'{self.ntot_unmasked} objects total')
        print(f'{self.ntot_masked} objects passed the selected cuts')

    def get_mask(self, fits_table, selection):

        if selection == 'all':
            return np.ones(fits_table.get_nrows()).astype('bool')
            
        elif selection == 'flags_and_mags':
            # select only objects that have mag_flag=0 in all bands, and mag is not -99
            # for the ugriz catalog only 1 of z and z2 has to pass (flag and mag)
            #mask = (fits_table['Flag'].read() == 0) #we might not want flag=0, too restrictive
            mask = np.ones(fits_table.get_nrows()).astype('bool')
            for band in "ugri":
                mask *= (fits_table[f"FLAG_GAAP_{band}"].read() == 0)
                mask *= (fits_table[f"MAG_GAAP_{band}"].read() != -99.) 
                #mask *= (fits_table[f"MAG_GAAP_{band}"].read() != 99.) #+99 are non detections, but fine to include
            if self.bands == "ugriz":
                mask *= np.logical_or((fits_table["FLAG_GAAP_z"].read() == 0), (fits_table["FLAG_GAAP_z2"].read() == 0) )
                z1 = fits_table["MAG_GAAP_z"].read()
                z2 = fits_table["MAG_GAAP_z2"].read()
                mask *= np.logical_or((z1 != -99.), (z2 != -99.)) #must have a measurement in either z col
                #mask *= np.logical_and((z1 !=  99.), (z2 !=  99.)) #+99 are non detections, but fine to include

            return mask

    def save_txpipe_cats(self):

        #Photometry file details
        phot_name = f"./txpipe_photometry_{self.concat_output_label}_" + self.cat_file.format(tile=str(datetime.date.today()) ,bands=self.bands) + ".h5"
        phot_col_dict = {
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
        pz_name = f"./txpipe_photoz_{self.concat_output_label}_" + self.cat_file.format(tile=str(datetime.date.today()) ,bands=self.bands) + ".h5"
        pz_col_dict = {
            "Z_B":"zb", 
        }
        
        with h5py.File(self.outfile, "r") as main_out:

            #Photometry file
            with h5py.File(phot_name, "w") as phot_out:
                grp = phot_out.create_group("photometry")
                for col in phot_col_dict.keys():
                    #join the two z columns
                    if col in ["MAG_GAAP_z", "MAGERR_GAAP_z"]:
                        z1 = main_out[f'catalog/MAG_GAAP_z'][:]
                        z2 = main_out[f'catalog/MAG_GAAP_z2'][:]
                        grp.create_dataset(
                            phot_col_dict[col], 
                            data=np.where(
                                np.logical_or((z2==99.),(z2==-99.)),
                                main_out[f'catalog/{col}'][:],
                                main_out[f'catalog/{col}2'][:]),
                            )
                    else:
                        grp.create_dataset(
                            phot_col_dict[col], 
                            data=main_out[f'catalog/{col}'][:],
                            )

            #photometry file
            with h5py.File(pz_name, "w") as pz_out:
                grp = pz_out.create_group("ancil")
                for col in pz_col_dict.keys():
                    grp.create_dataset(
                        pz_col_dict[col], 
                        data=main_out[f'catalog/{col}'][:],
                        )
                
                
                
                
            
        
            
                
                    
                        