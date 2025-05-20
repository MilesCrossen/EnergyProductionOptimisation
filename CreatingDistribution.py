import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import weibull_min, beta
import ast
# Number of simulations to run
NUM_SIMULATIONS = 5
# Dictionary to store station weightings (init -> 1.0), will add sim annealing later tho
station_weightings = {}

def parse_dist_params(param_str):
    """
    Parse distribution parameters from string format in CSV.
    Handles both tuple format and np.float64 format.
    """
    # Clean up the parameter string
    param_str = param_str.strip()
    # Replace np.float64 with just the value
    param_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', param_str)
    # Try to safely evaluate the string as a tuple
    try:
        params = ast.literal_eval(param_str)
        return params
    except:
        print(f"Error parsing parameters: {param_str}")
        # Fallback to default parameters if failure
        return (1.0, 1.0, 0.0, 1.0)



def load_dist_params_file(filename, station_id):
    """
    Load distribution parameters from a CSV file.
    Returns a dictionary mapping day of year to distribution parameters.
    """
    print(f"Loading distribution parameters for {station_id} from {filename}")

    try:
        df = pd.read_csv(filename)

        # Initialize dictionary to store parameters for each day
        day_params = {}

        for _, row in df.iterrows():
            # Check if this row belongs to our station
            if 'Station_ID' in df.columns and row['Station_ID'] != station_id:
                continue


            # Get day of year (format: MM-DD)
            day = row['Day_of_Year']
            # Get parameter column (use column name instead of position)
            if 'glorad' in df.columns:
                params = parse_dist_params(row['glorad'])
            elif 'wdsp^3' in df.columns:
                params = parse_dist_params(row['wdsp^3'])
            else:
                # Last resort - try second column but with iloc (avoids FutureWarning)
                params = parse_dist_params(row.iloc[1])
            # Store in dict
            day_params[day] = params

        return day_params
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return {}


def load_fourier_results():
    """
    Load the FourierResults.csv file to get information about
    which stations, columns, and distribution types are available.
    """
    print("Loading FourierResults.csv to identify available stations and parameters...")

    try:
        df = pd.read_csv("FourierResults.csv")

        # Create a list to store station information
        stations = []

        for _, row in df.iterrows():
            station_info = {
                'station_id': row['Station_ID'],
                'file': row['File'],
                'column': row['Column'],
                'dist_type': row['Distribution_Type'],
                'params_file': row['Dist_Params_Filename']
            }

            # Add to station weightings dictionary with default weight of 1.0
            key = f"{row['Station_ID']}_{row['Column']}"
            station_weightings[key] = 1.0

            stations.append(station_info)

        return stations
    except Exception as e:
        print(f"Error loading FourierResults.csv: {e}")
        return []


def generate_random_sample(day, params, dist_type):
    """
    Generate a random sample based on distribution type and parameters.
    """
    if dist_type == 'weibull':
        #Weibull params (shape, loc, scale)
        shape, loc, scale = params
        if shape <= 0:
            shape = 1.0 # Default to 1.0 if shape is invalid
        if scale <= 0:
            scale = 1.0 # Default to 1.0 if scale is invalid

        return weibull_min.rvs(shape, loc=loc, scale=scale)

    elif dist_type == 'beta':
        # Beta parameters (a, b, loc, scale)
        a, b, loc, scale = params
        if a <= 0:
            a = 1.0 # Default to 1.0 if a is invalid
        if b <= 0:
            b = 1.0 # Default to 1.0 if b is invalid

        return beta.rvs(a, b, loc=loc, scale=scale)

    else:
        print(f"Unknown distribution type: {dist_type} for day {day}")
        return np.nan


def run_simulation(stations, simulation_num):
    """
    Run a single simulation for all days of the year for all stations.
    Returns a DataFrame with the simulation results.
    """
    print(f"Running simulation {simulation_num}...")
    # Create a DataFrame with days of the year
    days = [f"{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 32)
            if (month != 2 or day <= 29) and (month not in [4, 6, 9, 11] or day <= 30)]

    results = pd.DataFrame(index=days)

    # Load distribution parameters for each station
    for station in stations:
        station_id = station['station_id']
        column = station['column']
        dist_type = station['dist_type']
        params_file = station['params_file']

        key = f"{station_id}_{column}"
        weighting = station_weightings.get(key, 1.0)
        # Skp if no params file specified
        if not params_file or pd.isna(params_file) or params_file == "":
            print(f"No parameter file specified for {station_id}, {column}. Skipping.")
            continue

        day_params= load_dist_params_file(params_file, station_id)# Load parameters

        # Generate random samples for each day
        samples = []
        for day in days:
            if day in day_params:
                # Generate weighted sample
                sample = generate_random_sample(day, day_params[day], dist_type) * weighting
            else:
                # Default parameters if day not found
                if dist_type == 'weibull':
                    sample = weibull_min.rvs(1.0, loc=0, scale=1.0) * weighting
                else: # beta
                    sample= beta.rvs(1.0, 1.0, loc=0, scale=1.0) * weighting

            samples.append(sample)

        # Add to results DataFrame with simulation number in column name
        col_name = f"{key}_sim{simulation_num}"
        results[col_name] = samples

    return results


def plot_all_simulations(all_results):
    """
    Plot all simulations on the same graph, overlaying the lines.
    """
    # Create a directory for plots if it doesn't exist
    if not os.path.exists('simulation_plots'):
        os.makedirs('simulation_plots')

    # Combine all DataFrames
    combined = pd.concat(all_results, axis=1)

    # Get all unique station_column combinations (without sim number)
    pattern = r'(.+)_sim\d+$'
    base_columns = set()
    for col in combined.columns:
        match = re.match(pattern, col)
        if match:
            base_columns.add(match.group(1))

    # Group by station and column type
    wind_stations = [col for col in base_columns if 'wdsp' in col]
    solar_stations = [col for col in base_columns if 'glorad' in col]

    # Create a color map for the simulations
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    # Plot wind stations
    plt.figure(figsize=(15, 10))
    for i, station in enumerate(wind_stations):
        plt.subplot(len(wind_stations), 1, i + 1)

        # Get all columns for this station across all simulations
        sim_cols = [col for col in combined.columns if col.startswith(station + '_sim')]

        # Plot each simulation
        for j, col in enumerate(sim_cols):
            sim_num = int(col.split('sim')[1])
            color = colors[sim_num % len(colors)]
            plt.plot(combined.index, combined[col], label=f"Simulation {sim_num}", color=color, alpha=0.7)

        plt.title(f"Wind Power - {station}")
        plt.ylabel('Wind Power (wdsp^3)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('simulation_plots', 'wind_simulations_overlay.png'))

    # Plot solar stations
    plt.figure(figsize=(15, 10))
    for i, station in enumerate(solar_stations):
        plt.subplot(len(solar_stations), 1, i + 1)

        # Get all columns for this station across all simulations
        sim_cols = [col for col in combined.columns if col.startswith(station + '_sim')]

        # Plot each simulation
        for j, col in enumerate(sim_cols):
            sim_num = int(col.split('sim')[1])
            color = colors[sim_num % len(colors)]
            plt.plot(combined.index, combined[col], label=f"Simulation {sim_num}", color=color, alpha=0.7)

        plt.title(f"Solar Radiation - {station}")
        plt.ylabel('Solar Radiation (glorad)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('simulation_plots', 'solar_simulations_overlay.png'))

    print("Created overlay plots for all simulations in simulation_plots directory.")


def main():
    """
    Main function to run the simulations and generate plots.
    """
    print("Starting distribution simulation process...")

    # Load station information from FourierResults.csv
    stations = load_fourier_results()

    if not stations:
        print("No stations found in FourierResults.csv. Exiting.")
        return

    print(f"Found {len(stations)} station-parameter combinations.")

    # Print station information
    for i, station in enumerate(stations):
        print(f"{i + 1}. Station: {station['station_id']}, Column: {station['column']}, "
              f"Type: {station['dist_type']}, Params file: {station['params_file']}")

    # Create directory for simulation results if it doesn't exist
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')

    # Print weightings
    print("\nStation weightings (all set to 1.0 by default):")
    for key, value in station_weightings.items():
        print(f"{key}: {value}")

    # Run simulations
    all_results = []
    for i in range(1, NUM_SIMULATIONS + 1):
        results = run_simulation(stations, i)
        all_results.append(results)
        # Save results to CSV
        csv_file = os.path.join('simulation_results', f'simulation_{i}.csv')
        results.to_csv(csv_file)
        print(f"Saved simulation {i} results to {csv_file}")


    # Create overlay plots of all simulations
    plot_all_simulations(all_results)
    print("All simulations completed successfully.")
    # Calculate and display overall statistics
    print("\nOverall Statistics:")
    # Create a directory for statistics if it doesn't exist
    if not os.path.exists('simulation_stats'):
        os.makedirs('simulation_stats')

    # Combine all simulation results for statistics
    # First, rename columns to remove simulation number for grouping
    all_renamed = []
    for i, df in enumerate(all_results):
        renamed = df.copy()
        renamed.columns = [col.split('_sim')[0] for col in df.columns]
        all_renamed.append(renamed)

    combined = pd.concat(all_renamed)
    grouped = combined.groupby(level=0)

    # Calculate statistics
    mean_values = grouped.mean()
    max_values = grouped.max()
    min_values = grouped.min()
    std_values = grouped.std()

    # Save statistics to CSV
    mean_values.to_csv(os.path.join('simulation_stats', 'mean_values.csv'))
    max_values.to_csv(os.path.join('simulation_stats', 'max_values.csv'))
    min_values.to_csv(os.path.join('simulation_stats', 'min_values.csv'))
    std_values.to_csv(os.path.join('simulation_stats', 'std_values.csv'))

    print("Statistics saved to simulation_stats directory.")


if __name__ == "__main__":
    main()