#!/usr/bin/env python3
"""
Trim WRF coordinates for CorrDiff ensemble visualization.
Simple script to extract and save trimmed coordinate arrays.
"""

import numpy as np
import xarray as xr
import netCDF4 as nc


def trim_coordinates(wrf_coord_path, trim_pixels, output_path):
    """
    Trim WRF coordinates and save as NetCDF for CorrDiff visualization.
    
    Parameters
    ----------
    wrf_coord_path : str
        Path to WRF coordinate file
    trim_pixels : tuple
        (top, bottom, left, right) pixels to remove
    output_path : str
        Path to save trimmed coordinates
    """
    top, bottom, left, right = trim_pixels
    
    # Load WRF coordinates
    wrf_coord = xr.open_dataset(wrf_coord_path)
    
    # Extract lat/lon arrays
    xlat = wrf_coord["XLAT"]
    xlon = wrf_coord["XLONG"]
    
    # Remove time dimension if present
    if "Time" in xlat.dims:
        xlat = xlat.isel(Time=0)
        xlon = xlon.isel(Time=0)
    elif "time" in xlat.dims:
        xlat = xlat.isel(time=0)
        xlon = xlon.isel(time=0)
    
    # Apply trimming: [top:-bottom, left:-right]
    lat_trimmed = xlat.values[top:-bottom, left:-right]
    lon_trimmed = xlon.values[top:-bottom, left:-right]
    
    # Save trimmed coordinates
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        ny, nx = lat_trimmed.shape
        
        f.createDimension('y', ny)
        f.createDimension('x', nx)
        
        lat_var = f.createVariable('lat', 'f4', ('y', 'x'))
        lon_var = f.createVariable('lon', 'f4', ('y', 'x'))
        
        lat_var[:] = lat_trimmed
        lon_var[:] = lon_trimmed
        
        # Add attributes
        lat_var.long_name = 'latitude'
        lat_var.units = 'degrees_north'
        lon_var.long_name = 'longitude'
        lon_var.units = 'degrees_east'
        
        f.title = 'Trimmed WRF coordinates for CorrDiff'
        f.description = f'Coordinates trimmed by {trim_pixels} pixels from original WRF grid'


def load_trimmed_coordinates(coord_path):
    """
    Load trimmed coordinates for visualization.
    
    Parameters
    ----------
    coord_path : str
        Path to trimmed coordinate file
        
    Returns
    -------
    tuple
        (lat_2d, lon_2d) coordinate arrays
    """
    with xr.open_dataset(coord_path) as ds:
        return ds.lat.values, ds.lon.values


if __name__ == "__main__":
    # Example usage
    wrf_path = "/home/younes.abid/git/physicsnemo/data/georefrenced/Space42_CorrDiff/wrf_coord.nc"
    trim_pixels = (7, 8, 7, 8)  # (top, bottom, left, right)
    output_path = "trimmed_coordinates_432x432.nc"
    
    trim_coordinates(wrf_path, trim_pixels, output_path)
    print(f"Trimmed coordinates saved to {output_path}")