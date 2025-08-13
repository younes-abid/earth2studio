## 4. Exploring the Sample Data

In this section, we will explore the sample data provided for the CorrDiff model, specifically the `hrrrr_mini.nc` file. This includes understanding what the data represents, how to load it, and how to visualize it. We will also cover how to locate the file on your machine and check its size.

---

### Overview of `hrrrr_mini.nc`

The `hrrrr_mini.nc` file is a NetCDF (Network Common Data Form) file, which is a standard format for storing multidimensional scientific data. This file contains weather data from the **High-Resolution Rapid Refresh (HRRR)** model, which is widely used for short-term weather forecasting.

Key features of the `hrrrr_mini.nc` file:
- **Variables**: It contains weather variables such as temperature, wind speed, and precipitation.
- **Dimensions**: The data is organized along dimensions like time, latitude, and longitude.
- **Compact Size**: This is a smaller version of the full HRRR dataset, designed for quick experimentation and learning.

---

### Locating and Checking the Data File

1. **Downloading the Data**:
   If the `hrrrr_mini.nc` file is not already available in your repository, you can download it from the provided source or ask your colleague for the file. Once downloaded, place it in a directory where it can be accessed by your scripts or notebooks.

2. **Finding the File**:
   To locate the file on your machine, use the `find` command in the terminal:
   ```bash
   find /home/younes.abid -name "hrrrr_mini.nc"
   ```
   This will search for the file starting from your home directory.

3. **Checking the File Size**:
   To check the size of the file, use the `ls` command:
   ```bash
   ls -lh /path/to/hrrrr_mini.nc
   ```
   This will display the file size in a human-readable format.

4. **Verifying the File**:
   To ensure the file is not corrupted, you can use the `ncdump` command (part of the NetCDF tools):
   ```bash
   ncdump -h /path/to/hrrrr_mini.nc
   ```
   This will display the metadata of the file, including its dimensions and variables.

---

### Loading and Visualizing the Data

We will use the `xarray` library, which is specifically designed for working with multidimensional data like NetCDF files.

1. **Installing `xarray`**:
   If `xarray` is not already installed, you can install it using pip:
   ```bash
   pip install xarray
   ```

2. **Loading the Data**:
   Use the following Python code to load the `hrrrr_mini.nc` file:
   ```python
   import xarray as xr

   # Load the NetCDF file
   data = xr.open_dataset("/path/to/hrrrr_mini.nc")

   # Display the dataset structure
   print(data)
   ```

3. **Exploring the Variables**:
   After loading the data, you can explore the available variables:
   ```python
   print(data.variables)
   ```

4. **Visualizing the Data**:
   To visualize a specific variable (e.g., temperature), use the following code:
   ```python
   import matplotlib.pyplot as plt

   # Select a variable (e.g., temperature)
   temperature = data["temperature"]

   # Plot the data for the first time step
   temperature.isel(time=0).plot()
   plt.title("Temperature at Time Step 0")
   plt.show()
   ```

---

### Understanding the Input and Output Formats

1. **Input Format**:
   - The input to the CorrDiff model is typically a low-resolution weather dataset.
   - Dimensions: Time, latitude, longitude, and sometimes additional features like pressure levels.
   - Variables: Temperature, wind speed, precipitation, etc.

2. **Output Format**:
   - The output is a high-resolution version of the input data.
   - The dimensions and variables remain the same, but the spatial resolution is significantly improved.

---

### Summary

In this section, we learned how to locate, load, and visualize the `hrrrr_mini.nc` file. This file serves as the input data for the CorrDiff model. Understanding the structure and content of this data is crucial for working effectively with the model. In the next section, we will run the pre-trained CorrDiff model on this data and analyze the results.