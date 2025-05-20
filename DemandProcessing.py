import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import fft, ifft
import os

# tkagg works on my laptop so I use this
matplotlib.use("TkAgg")
# Number of top Fourier components to keep
top_n = 5
file_path = "22-23 Electricity Demand.csv"


def load_demand_data(file_path):
    """
    Load the electricity demand data from the CSV file and perform basic validation.
    Returns the loaded DataFrame or none if there was an error.
    """
    print(f"Loading demand data from {file_path}...")
    try:
        # Load the data with appropriate encoding settings
        data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

        # Check that required columns exist
        if "DateTime" not in data.columns or "AI Demand" not in data.columns:
            print(f"Error: Required columns ('DateTime', 'AI Demand') not found in {file_path}.")
            return None

        print(f"Successfully loaded demand data with {len(data)} rows and {len(data.columns)} columns.")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def preprocess_demand_data(data):
    """
    Preprocess the demand data:
    - Convert DateTime to datetime format
    - Filter for data within a specific year
    - Ensure demand values are numeric
    - Compute daily averages and standard deviations

    Returns two DataFrames: daily average and daily standard deviation
    """
    print("Preprocessing demand data...")

    try:
        # Convert DateTime column to datetime format
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M')

        # Filter for data within the year 2022
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31, 23, 59)  # End of 2022
        data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]
        # Ensure AI Demand is numeric and dro invalid rows
        data['AI Demand'] = pd.to_numeric(data['AI Demand'], errors='coerce')
        data=data.dropna(subset=['AI Demand'])
        # Extract date from DateTime and group by date
        data['Date'] = data['DateTime'].dt.date

        # Calculate daily average demand
        daily_avg_demand = data.groupby('Date')['AI Demand'].mean()

        # Calculate daily standard deviation
        daily_std_demand = data.groupby('Date')['AI Demand'].std()

        # For days with only one measurement, std will be NaN, replace with 0
        daily_std_demand = daily_std_demand.fillna(0)

        print(f"Preprocessed demand data to {len(daily_avg_demand)} daily entries.")
        return daily_avg_demand, daily_std_demand
    except Exception as e:
        print(f"Error preprocessing demand data: {e}")
        return None, None



def save_daily_data(daily_avg_demand, daily_std_demand):
    """
    Save the daily average and standard deviation data to CSV files.
    """
    # Does output directory exist?
    os.makedirs("demand_data", exist_ok=True)

    # Save daily average demand
    avg_output_file = os.path.join("demand_data", "daily_avg_ai_demand.csv")
    daily_avg_demand.to_csv(avg_output_file, index=True, header=["Average AI Demand (MW)"])
    print(f"Daily average AI demand saved to '{avg_output_file}'.")
    # Also save to root
    daily_avg_demand.to_csv("daily_avg_ai_demand.csv", index=True, header=["Average AI Demand (MW)"])
    print(f"Daily average AI demand also saved to root directory for compatibility.")
    # Save daily standard deviation
    std_output_file = os.path.join("demand_data", "daily_std_ai_demand.csv")
    daily_std_demand.to_csv(std_output_file, index=True, header=["Std Dev AI Demand (MW)"])
    print(f"Daily standard deviation of AI demand saved to '{std_output_file}'.")

    # combined data...
    combined_df = pd.DataFrame({
        "Average AI Demand (MW)": daily_avg_demand,
        "Std Dev AI Demand (MW)": daily_std_demand
    })
    combined_output_file = os.path.join("demand_data", "daily_demand_stats.csv")
    combined_df.to_csv(combined_output_file, index=True)
    print(f"Combined daily demand statistics saved to '{combined_output_file}'.")


def plot_daily_demand(daily_avg_demand, daily_std_demand):
    """
    Plot the daily average demand and standard deviation.
    """
    try:
        # Create output directory for plots if it doesn't exist
        os.makedirs("demand_plots", exist_ok=True)
        # Convert index to datetime for better plotting
        dates= pd.to_datetime(daily_avg_demand.index)
        # Plot daily average demand
        plt.figure(figsize=(12, 6))
        plt.plot(dates, daily_avg_demand.values, marker='', color='blue', linewidth=1)
        plt.title("Average Daily AI Demand (2022)")
        plt.xlabel("Date")
        plt.ylabel("AI Demand (MW)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("demand_plots", "daily_avg_demand.png"))
        plt.close()




        plt.figure(figsize=(12, 6)) # Plot daily standard deviation
        plt.plot(dates, daily_std_demand.values, marker='', color='red', linewidth=1)
        plt.title("Daily Standard Deviation of AI Demand (2022)")
        plt.xlabel("Date")
        plt.ylabel("Standard Deviation (MW)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("demand_plots", "daily_std_demand.png"))
        plt.close()

        # Plot combined (avg ± std)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, daily_avg_demand.values, color='blue', linewidth=1, label="Average")

        # Plot the confidence interval (avg ± std)
        plt.fill_between(
            dates,
            daily_avg_demand.values - daily_std_demand.values,
            daily_avg_demand.values + daily_std_demand.values,
            color='blue', alpha=0.2, label="±1 Std Dev"
        )

        plt.title("Daily AI Demand with Variability (2022)")
        plt.xlabel("Date")
        plt.ylabel("AI Demand (MW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("demand_plots", "daily_demand_with_variability.png"))
        plt.close()

        print("Created demand plots in 'demand_plots' directory.")
    except Exception as e:
        print(f"Error creating demand plots: {e}")


def perform_fourier_transform(signal, name):
    """
    Perform Fourier Transform on a signal and extract the top n components.
    Returns the Fourier coefficients, significant indices, and reconstructed signal.
    """
    print(f"Performing Fourier Transform on {name}...")

    try:
        # Ensure the signal is a numpy array
        signal_values = np.array(signal)
        num_days = len(signal_values)
        # Compute Fourier coefficients
        fourier_coefficients = fft(signal_values)
        # Compute amplitudes and find significant components
        amplitudes = np.abs(fourier_coefficients) / num_days
        significant_indices = np.argsort(amplitudes)[-top_n:][::-1]
        truncated_coefficients = np.zeros_like(fourier_coefficients)# Create truncated coefficients for reconstruction
        truncated_coefficients[significant_indices] = fourier_coefficients[significant_indices]



        # Reconstruct signal
        reconstructed_signal = ifft(truncated_coefficients).real

        return fourier_coefficients, significant_indices, reconstructed_signal
    except Exception as e:
        print(f"Error performing Fourier Transform on {name}: {e}")
        return None, None, None


def get_fourier_equation(fourier_coefficients, significant_indices, num_days):
    """
    Generate the Fourier equation for the top components.
    Returns the equation as a string and a cosine form for computational use.
    """
    frequencies = np.fft.fftfreq(num_days)
    # Generate eqn terms
    equation_terms = []
    cosine_terms = []

    for k in significant_indices:
        amplitude = np.abs(fourier_coefficients[k]) / num_days
        phase = np.angle(fourier_coefficients[k])
        frequency = frequencies[k]

        #Human-readable equation
        term = f"{amplitude:.6f} * cos(2π * {frequency:.6f} * t + {phase:.6f})"
        equation_terms.append(term)

        # Computational form
        comp_term = f"{amplitude:.6f} * np.cos(2 * np.pi * {frequency:.6f} * days + {phase:.6f})"
        cosine_terms.append(comp_term)

    equation = "f(t) = " + " + ".join(equation_terms)
    cosine_form = " + ".join(cosine_terms)

    return equation, cosine_form


def plot_fourier_results(original_signal, reconstructed_signal, name):
    """
    Plot the original and reconstructed signals for visual comparison.
    """
    try:
        os.makedirs("demand_plots", exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(original_signal) + 1), original_signal,
                 label="Original Signal", linewidth=1, alpha=0.7)
        plt.plot(range(1, len(reconstructed_signal) + 1), reconstructed_signal,
                 label="Reconstructed Signal", linestyle="--", linewidth=1.5)
        plt.xlabel("Day of the Year")
        plt.ylabel(name)
        plt.title(f"Original vs Reconstructed Signal ({name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("demand_plots", f"fourier_{name.lower().replace(' ', '_')}.png"))
        plt.close()

        print(f"Created Fourier comparison plot for {name}.")
    except Exception as e:
        print(f"Error creating Fourier plot for {name}: {e}")


def save_fourier_results(name, equation, cosine_form):
    """
    Save the Fourier equation and cosine form to a text file.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs("demand_data", exist_ok=True)

        output_file = os.path.join("demand_data", f"fourier_{name.lower().replace(' ', '_')}.txt")

        with open(output_file, 'w') as f:
            f.write(f"Fourier Transform Equation for {name}:\n")
            f.write(f"{equation}\n\n")
            f.write(f"Cosine Form for Computational Use:\n")
            f.write(f"{cosine_form}\n")

        print(f"Saved Fourier equations for {name} to {output_file}")
    except Exception as e:
        print(f"Error saving Fourier results for {name}: {e}")


def save_to_fourier_demand_csv(demand_avg_equation, demand_std_equation,
                               demand_avg_cosine, demand_std_cosine):
    """
    Save demand Fourier results to a separate FourierDemand.csv file instead of
    modifying the existing FourierResults.csv file.
    """
    try:
        fourier_csv = "FourierDemand.csv"
        # Create a new dataframe for demand data
        df = pd.DataFrame(columns=["Station_ID", "File", "Column", "Fourier_Avg", "Fourier_Std", "Fourier_Avg_Cosine_Form", "Fourier_Std_Cosine_Form","Distribution_Type", "Dist_Params_Filename"])
        # Check if file already exists
        if os.path.exists(fourier_csv):
            try:
                df = pd.read_csv(fourier_csv)
                # Do all required columns exist
                for col in df.columns:
                    if col not in df.columns:
                        df[col] = ""
            except Exception as e:
                print(f"Error reading existing {fourier_csv}, creating new file: {e}")
                # Use the empty dataframe above

        # Check if entry already exists
        mask = (df['Station_ID'] == 'AI_Demand') & (df['Column'] == 'Demand')

        # If exists, update the row
        if mask.any():
            idx = mask.idxmax()
            df.loc[idx, 'Fourier_Avg'] = demand_avg_equation
            df.loc[idx, 'Fourier_Std'] = demand_std_equation
            df.loc[idx, 'Fourier_Avg_Cosine_Form'] = demand_avg_cosine
            df.loc[idx, 'Fourier_Std_Cosine_Form'] = demand_std_cosine
            print(f"Updated existing demand entry in {fourier_csv}")
        else:
            # Add new row
            new_row = {
                'Station_ID': 'AI_Demand',
                'File': file_path,
                'Column': 'Demand',
                'Fourier_Avg': demand_avg_equation,
                'Fourier_Std': demand_std_equation,
                'Fourier_Avg_Cosine_Form': demand_avg_cosine,
                'Fourier_Std_Cosine_Form': demand_std_cosine,
                'Distribution_Type': 'normal',  # Assume normal distribution for demand
                'Dist_Params_Filename': 'daily_demand_stats.csv'
            }

            # Append new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added demand entry to {fourier_csv}")

        # Save the updated dataframe
        df.to_csv(fourier_csv, index=False)
        print(f"Saved demand Fourier results to {fourier_csv}")
    except Exception as e:
        print(f"Error saving to {fourier_csv}: {e}")


def process_demand_and_fourier():
    """
    Main function to process the demand data and perform Fourier analysis.
    """
    print("Starting demand data processing and Fourier analysis...")

    # Load the demand data
    data = load_demand_data(file_path)
    if data is None:
        return
    # Preprocess the data
    daily_avg_demand, daily_std_demand = preprocess_demand_data(data)
    if daily_avg_demand is None or daily_std_demand is None:
        return
    # Save the processed data
    save_daily_data(daily_avg_demand, daily_std_demand)


    # Plot the daily demand
    plot_daily_demand(daily_avg_demand, daily_std_demand)
    # Perform Fourier transform on average demand
    avg_coeffs, avg_indices, avg_reconstructed = perform_fourier_transform(
        daily_avg_demand.values, "Average AI Demand")
    # Initialize variables for Fourier equations
    avg_equation = ""
    avg_cosine = ""
    std_equation = ""
    std_cosine = ""



    if avg_coeffs is not None:
        # Get the Fourier equation for average demand
        avg_equation, avg_cosine = get_fourier_equation(
            avg_coeffs, avg_indices, len(daily_avg_demand))

        # Print the equation
        print("\nDerived Fourier Transform Equation for Average Demand (Top Components):")
        print(avg_equation)
        # Save the Fourier results
        save_fourier_results("Average AI Demand", avg_equation, avg_cosine)
        # Plot the Fourier results
        plot_fourier_results(daily_avg_demand.values, avg_reconstructed, "Average AI Demand")

    # Perform Fourier transform on demand standard deviation
    std_coeffs, std_indices, std_reconstructed = perform_fourier_transform(
        daily_std_demand.values, "AI Demand Standard Deviation")

    if std_coeffs is not None:
        # Get the Fourier equation for standard deviation
        std_equation, std_cosine = get_fourier_equation(
            std_coeffs, std_indices, len(daily_std_demand))

        # Print the equation
        print("\nDerived Fourier Transform Equation for Demand Std Dev (Top Components):")
        print(std_equation)

        # Save the Fourier results
        save_fourier_results("AI Demand Standard Deviation", std_equation, std_cosine)

        # Plot the Fourier results
        plot_fourier_results(daily_std_demand.values, std_reconstructed, "AI Demand Standard Deviation")

    # Save results to FourierDemand.csv (not FourierResults.csv)
    save_to_fourier_demand_csv(avg_equation, std_equation, avg_cosine, std_cosine)

    print("Demand data processing and Fourier analysis completed successfully.")


if __name__ == "__main__":
    process_demand_and_fourier()