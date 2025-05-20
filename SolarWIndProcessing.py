import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')# tkagg fixes rendering errors for me
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.stats import weibull_min, beta
import os # Importing fun things!!
top_n = 5


def load_and_preprocess_data(file_path, column_name, data_type='unknown'):
    # load data from csv
    data = pd.read_csv(file_path)
    # convert date -> datetime format. Date must be in Excel short type
    # we wanna handle case insensitive data because for some reason it varies on the met eireann data itself
    date_col = next((col for col in data.columns if col.lower() == 'date'), None)
    if date_col is None:
        raise ValueError("No 'Date' or 'date' column found in the input CSV.")
    data['Date'] = pd.to_datetime(data[date_col], format='%d/%m/%Y')
    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
    data = data.dropna(subset=[column_name])# drop rows with missing/invalid values in column
    data = data[data[column_name] != 0] # also ignore 0s
    # extract day of year (ignoring leap years to simplify)
    data['Day_of_Year'] = data['Date'].dt.strftime('%m-%d')  # extract MM-DD for grouping
    # average values for each day of the year across all years
    daily_avg = data.groupby('Day_of_Year')[column_name].mean()
    # find standard deviation for each day of the year
    daily_std = data.groupby('Day_of_Year')[column_name].std()

    if data_type == 'wind': # Fit appropriate distributions to each day's data
        # For wind use Weibull distribution (thanks prof farhang)
        daily_dist_params = data.groupby('Day_of_Year')[column_name].apply(
            lambda x: weibull_min.fit(x, floc=0) if len(x) > 1 else (np.nan, np.nan, np.nan)
        )
        dist_type = 'weibull'

    elif data_type == 'solar':
        # For solar use Beta distribution
        daily_dist_params = data.groupby('Day_of_Year')[column_name].apply(
            lambda x: fit_beta_safely(x) if len(x) > 1 and x.std() > 0 else (np.nan, np.nan, np.nan, np.nan)
        )
        dist_type ='beta'
    else:
        daily_dist_params = None
        dist_type = 'unknown'

    # convert to numpy arrays
    averaged_values = daily_avg.values
    std_values = daily_std.values

    return averaged_values, std_values, daily_avg.index, daily_dist_params, dist_type


# A safer beta fitting function to handle edge cases
def fit_beta_safely(data):
    try:
        # Normalize data to (0, 1) range, excluding exact 0 and 1 values
        min_val = data.min()
        max_val = data.max()

        # If all vals are the same, return default params
        if min_val == max_val:
            return (1.0, 1.0, 0.0, 1.0)#Default parameters

        # Scale data to (0.001, 0.999) to avoid boundary issues
        epsilon = 0.001
        normalized_data = (data - min_val) / (max_val - min_val)
        normalized_data = normalized_data * (1 - 2 * epsilon) + epsilon

        # Fit the Beta distribution w constraints
        with np.errstate(all='ignore'):  # Suppress numpy warnings because I waws getting spammed w these before
            params = beta.fit(normalized_data, floc=0, fscale=1)

        # If fit fails /gives invalid params, return default params
        if np.isnan(params).any() or params[0] <= 0 or params[1] <= 0:
            return (1.0, 1.0, 0.0, 1.0) # Default parameters

        return params
    except Exception:
        # Fallback for any other errors
        return (1.0, 1.0, 0.0, 1.0)


def perform_fourier_transform(signal):# find fourier transform
    fourier_coefficients = fft(signal) # compute FFT
    return fourier_coefficients

def reconstruct_signal_with_top_components(fourier_coefficients, num_days, top_n):# reconstruct signal with top n components
    # compute amplitudes of the fourier coefficients
    amplitudes = np.abs(fourier_coefficients) / num_days

    significant_indices = np.argsort(amplitudes)[-top_n:][::-1]# get the indices of the top n amplitudes (ignoring DC component @ index 0)
    truncated_coefficients = np.zeros_like(fourier_coefficients)# create a new array for truncated fourier coefficients
    truncated_coefficients[significant_indices] = fourier_coefficients[significant_indices]
    reconstructed_signal = ifft(truncated_coefficients).real# reconstruct signal using the truncated coefficients

    return reconstructed_signal, significant_indices

def get_fourier_equation(fourier_coefficients, num_days, significant_indices): # print fourier transform eqn for top n components
    frequencies = np.fft.fftfreq(num_days) # frequency vals
    equation_terms = []

    # loop through the significant indices
    for k in significant_indices:
        amplitude = np.abs(fourier_coefficients[k]) / num_days
        phase = np.angle(fourier_coefficients[k])
        frequency = frequencies[k]
        term = f"{amplitude:.6f} * cos(2π * {frequency:.6f} * t + {phase:.6f})"
        equation_terms.append(term)

    # combine terms into an equation
    equation = " + ".join(equation_terms)
    return f"f(t) = {equation}"


# get fourier transform into cos sum format
def get_cosine_form(fourier_coefficients, num_days, significant_indices):
    frequencies = np.fft.fftfreq(num_days) #frequency vals
    cosine_terms = []

    # loop through the significant indices
    for k in significant_indices:
        amplitude = np.abs(fourier_coefficients[k]) / num_days
        phase =np.angle(fourier_coefficients[k])
        frequency = frequencies[k]
        term = f"{amplitude:.6f} * np.cos(2 * np.pi * {frequency:.6f} * days + {phase:.6f})"
        cosine_terms.append(term)

    # combine terms into the requested format
    return " + ".join(cosine_terms)


# plot results
def plot_results(original_signal, reconstructed_signal, fourier_coefficients, significant_indices, day_labels,
                 column_name):
    num_days = len(original_signal)

    plt.figure(figsize=(12, 6)) # Plot original vs reconstructed signal
    plt.plot(range(1, num_days + 1), original_signal, label=f"Original Signal ({column_name})", marker="o")
    plt.plot(range(1, num_days + 1), reconstructed_signal, label="Reconstructed Signal (Top Components)",
             linestyle="--")
    plt.xlabel("Day of the Year")
    plt.ylabel(column_name)
    plt.title(f"Original vs Reconstructed Signal ({column_name})")
    plt.legend()
    plt.grid()
    plt.show()


def entry_exists(df, station_id, column_name): # Does entry exist already..?
    if df.empty:
        return False

    # Check if there's already an entry with the same station ID and column name
    mask = (df['Station_ID'] == station_id) & (df['Column'] == column_name)
    return mask.any()


# function to write Fourier results to CSV
def write_fourier_to_csv(csv_filename, file_path, column_name, avg_equation, std_equation, avg_cosine_form, std_cosine_form, dist_type='unknown', daily_dist_params=None):
    station_id = os.path.splitext(os.path.basename(file_path))[0]
    # Extract station ID directly from the inputf ilename
    # Create a unique filename for the distribution parameters inc. station ID
    dist_params_filename = f"{station_id}_{column_name}_dist_params.csv"
    try: # Read existing CSV or create a new one with proper dtypes
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        # Create a new DataFrame with object dtype to avoid.. more warnings
        df = pd.DataFrame(
            columns=["Station_ID", "File", "Column", "Fourier_Avg", "Fourier_Std", "Fourier_Avg_Cosine_Form",
                     "Fourier_Std_Cosine_Form", "Distribution_Type", "Dist_Params_Filename"])

    # already entry for this station/type?
    if entry_exists(df, station_id, column_name):
        print(f"Entry for station {station_id}, column {column_name} already exists in {csv_filename}. Skipping.")
        return
    for col in df.columns:
        df[col] = df[col].astype(object)

    new_row = {  # Prepare new row as a dictionary with proper types
        "Station_ID": station_id,
        "File": file_path,
        "Column": column_name,
        "Fourier_Avg": avg_equation,
        "Fourier_Std": std_equation,
        "Fourier_Avg_Cosine_Form": avg_cosine_form,
        "Fourier_Std_Cosine_Form": std_cosine_form,
        "Distribution_Type": dist_type,
        "Dist_Params_Filename": ""
    }

    if daily_dist_params is not None: # Save the daily distribution parameters to a separate CSV if they exist
        dist_params_df = pd.DataFrame(daily_dist_params) # Convert the Series to df and add a station_id column
        dist_params_df['Station_ID'] = station_id

        # Save w/day and station info
        dist_params_df.to_csv(dist_params_filename, index=True)
        new_row["Dist_Params_Filename"] = dist_params_filename
        print(f"Distribution parameters saved to {dist_params_filename} for station {station_id}")

    # Append row to df +save
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_filename, index=False)
    print(f"Fourier series saved to {csv_filename} for station {station_id}, column {column_name}")


# main fourier analysis function
def analyze_fourier(file_path, column_name, data_type='unknown', csv_filename="FourierResults.csv"):
    # analyse specified column of dataset and compute fourier transform for avg + stdev
    # load and preprocess data
    averaged_values, std_values, day_labels, daily_dist_params, dist_type = load_and_preprocess_data(file_path, column_name, data_type)
    # perform fourier transform for averages
    avg_fourier_coefficients = perform_fourier_transform(averaged_values)
    # perform fourier transform for standard deviations
    std_fourier_coefficients = perform_fourier_transform(std_values)
    num_days = len(averaged_values)
    # reconstruct average signal using top components
    avg_reconstructed_signal, avg_significant_indices = reconstruct_signal_with_top_components(
        avg_fourier_coefficients, num_days, top_n
    )
    # reconstruct stdev using top components
    std_reconstructed_signal, std_significant_indices = reconstruct_signal_with_top_components(
        std_fourier_coefficients, num_days, top_n
    )

    # get fourier transform equations
    avg_equation = get_fourier_equation(avg_fourier_coefficients, num_days, avg_significant_indices)
    std_equation = get_fourier_equation(std_fourier_coefficients, num_days, std_significant_indices)

    # get cos sum forms
    avg_cosine_form = get_cosine_form(avg_fourier_coefficients, num_days, avg_significant_indices)
    std_cosine_form = get_cosine_form(std_fourier_coefficients, num_days, std_significant_indices)

    # save fourier series to CSV
    write_fourier_to_csv(csv_filename, file_path, column_name, avg_equation, std_equation, avg_cosine_form,
                         std_cosine_form, dist_type, daily_dist_params)

    plot_results(averaged_values, avg_reconstructed_signal, avg_fourier_coefficients, avg_significant_indices,
                 day_labels, column_name + " (Avg)")
    plot_results(std_values, std_reconstructed_signal, std_fourier_coefficients, std_significant_indices, day_labels,
                 column_name + " (Std Dev)")


# run fourier analysis when this gets executed
if __name__ == "__main__":
    file_path = "WeatherAthenryProcessed.csv"  # replace with the actual file path
    csv_filename = "FourierResults.csv"  # file to store results

    # Wind data - use power_wind column
    column_name_wind = "wdsp^3"  # wind speed cubed
    analyze_fourier(file_path, column_name_wind, 'wind', csv_filename)

    # Solar data - use glorad column
    column_name_solar = "glorad"  # solar radiation
    analyze_fourier(file_path, column_name_solar, 'solar', csv_filename)