## 4.1 Downloading and Preparing the HRRR Data

In this section, we will guide you through downloading HRRR (High-Resolution Rapid Refresh) data and preparing the `hrrrr_mini.nc` file. This file is a small, sample NetCDF dataset that can be used for testing and prototyping with the CorrDiff model.

---

### Step 1: Download HRRR Data

HRRR data is available from several public sources. Below are the most accessible options:

#### Option 1: NOAA HRRR Data on AWS
- **Description**: NOAA provides a rolling archive of HRRR data in GRIB2 and Zarr formats, freely accessible via Amazon S3.
- **How to Access**:
  1. Use the AWS S3 bucket:
     - GRIB2: `s3://noaa-hrrr-pds`
     - Zarr: `s3://hrrrzarr/`
  2. For browser access and documentation, visit the [NOAA Open Data Registry page](https://registry.opendata.aws/noaa-hrrr-pds/).

#### Option 2: University of Utah HRRR Download Page
- **Description**: A user-friendly web interface for downloading HRRR GRIB2 files.
- **How to Access**:
  1. Visit the [HRRR Download Page](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/hrrr.html).
  2. Select the model type, variable, and date.
  3. Click the blue button to download the GRIB2 file directly.

#### Option 3: Python Tools for Automated Download
- **hrrrb (PyPI)**:
  - Install the package:
    ```bash
    pip install hrrrb
    ```
  - Example usage:
    ```python
    from hrrrb import download_hrrr
    download_hrrr(dates=["2025-07-01"], model="hrrr", field="sfc", save_dir="./")
    ```

#### Notes:
- HRRR data is typically large (hundreds of MB per file). For a "mini" dataset, download only a small region, a few variables, and a short time range.
- Most HRRR data is in GRIB2 or Zarr format, which needs to be converted to NetCDF.

---

### Step 2: Convert and Subset the Data

Once you have downloaded the HRRR data, you need to convert it to NetCDF format and create a smaller subset.

#### Tools Required:
- **xarray**: For handling NetCDF files.
- **cfgrib**: For reading GRIB2 files in Python.

Install the required libraries:
```bash
pip install xarray cfgrib
```

#### Example Python Workflow:
```python
import xarray as xr

# Open HRRR GRIB2 file (requires cfgrib)
ds = xr.open_dataset("your_hrrr_file.grib2", engine="cfgrib")

# Subset: select variables, region, and time
mini_ds = ds.isel(x=slice(0, 64), y=slice(0, 64), time=slice(0, 1))  # Adjust as needed

# Save as NetCDF
mini_ds.to_netcdf("hrrrr_mini.nc")
```

#### Explanation:
- `isel`: Selects a subset of the data. Adjust the slices (`x`, `y`, `time`) to match the desired region and time range.
- `to_netcdf`: Saves the subset as a NetCDF file.

---

### Step 3: Verify the `hrrrr_mini.nc` File

After creating the `hrrrr_mini.nc` file, verify its contents to ensure it is correct.

1. **Check File Size**:
   ```bash
   ls -lh hrrrr_mini.nc
   ```
   The file should be small (a few MB).

2. **Inspect Metadata**:
   Use the `ncdump` command to inspect the file:
   ```bash
   ncdump -h hrrrr_mini.nc
   ```

3. **Load and Visualize**:
   Use Python to load and visualize the data:
   ```python
   import xarray as xr
   import matplotlib.pyplot as plt

   # Load the NetCDF file
   data = xr.open_dataset("hrrrr_mini.nc")

   # Print dataset structure
   print(data)

   # Visualize a variable (e.g., temperature)
   temperature = data["temperature"]
   temperature.isel(time=0).plot()
   plt.title("Temperature at Time Step 0")
   plt.show()
   ```

---

### Summary

In this section, we learned how to download HRRR data, convert it to NetCDF format, and create a smaller subset (`hrrrr_mini.nc`) for testing. This file will serve as the input data for the CorrDiff model. In the next section, we will run the pre-trained CorrDiff model on this data and analyze the results.