import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import geopandas as gpd
import os.path
#fig = plt.figure()

# TODO integrate constants with repo/config files/snakefile
NATURAL_EARTH_RESOLUTION = '10m' # 1:10m scale
# first administration level
NATURAL_EARTH_DATA_SET = 'admin_1_states_provinces'
CRS = 4326 # WGS84
DEFAULT_OUTPATH_SHP = "../data/province_shapes/CHN_adm1.shp"

def make_province_shape_file(country_iso2_code = 'CN', output_file= DEFAULT_OUTPATH_SHP):
    """fetch the province/state level (1st admin level) from the NATURAL_EARTH data store and make a file

    Args:
        country_iso2_code (str, optional): the country code (iso_a2) for which
          provincial records will be extracted. None will not filter (untestetd) Defaults to 'CN'
        output_file (str, optional): The default output location. Defaults to DEFAULT_OUTPATH_SHP.
    """    

    shpfilename = shpreader.natural_earth(resolution=NATURAL_EARTH_RESOLUTION,
                                    category='cultural',
                                    name=NATURAL_EARTH_DATA_SET)
    reader = shpreader.Reader(shpfilename)
    print("downloaded succesfully")
    provinces_states = reader.records()

    def filter_country_code(records:object, target_iso_a2_code = 'CN')->list:
        """filter provincial/state (admin level 1) records for one country

        Args:
            records (shpreader.Reader.records): the records object from cartopy shpreader for natural earth dataset
            target_iso_a2_code (str, optional): the country code (iso_a2) for which provincial records will be extracted. Defaults to 'CN'.

        Returns:
            list: records list
        """    
        results = []
        for rec in records:
            if rec.attributes['iso_a2'] == target_iso_a2_code:
                results.append(rec)
        
        return results
    
    #TODO test with none
    if country_iso2_code is not None:
        provinces_states = filter_country_code(provinces_states, target_iso_a2_code=country_iso2_code)

    gdf = gpd.GeoDataFrame(geometry=[r.geometry for r in res])
    gdf.set_crs(epsg=CRS, inplace=True)  # WGS84
    gdf.to_file(os.path.abspath(output_file))

make_province_shape_file()