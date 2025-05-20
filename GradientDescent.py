import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import weibull_min, beta
import ast
import re


class WeightOptimizer:
    def __init__(self, fourier_results_path="FourierResults.csv",
                 fourier_demand_path="FourierDemand.csv",
                 learning_rate=0.001, max_iterations=1000, convergence_threshold=1e-5):
        """
        Initialize the weight optimizer with parameters for gradient descent.

        Args:
            fourier_results_path: Path to the FourierResults.csv file
            fourier_demand_path; Path to the FourierDemand.csv file
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations for optimization
            convergence_threshold: Convergence threshold for early stopping
        """
        self.fourier_results_path = fourier_results_path
        self.fourier_demand_path = fourier_demand_path
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Initialize dictionaries to store data
        self.station_data = {}
        self.demand_data = None
        self.weightings = {}

        # Days of the year (ignore leap years, shoul probably add that later though...)
        self.days = [f"{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 32)
                     if (month != 2 or day <= 29) and (month not in [4, 6, 9, 11] or day <= 30)]

        # Create directories for outputs
        os.makedirs('optimization_results', exist_ok=True)
        os.makedirs('optimization_plots', exist_ok=True)

    def parse_dist_params(self, param_str):
        """Parse distribution parameters from string format in CSV."""
        param_str = param_str.strip()
        param_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', param_str)

        try:
            params = ast.literal_eval(param_str)
            return params
        except:
            print(f"Error parsing parameters: {param_str}")
            # Fallback to default parameters
            return (1.0, 1.0, 0.0, 1.0)

    def load_dist_params_file(self, filename, station_id):
        """Load distribution parameters from a CSV file."""
        print(f"Loading distribution parameters for {station_id} from {filename}")

        try:
            df = pd.read_csv(filename)
            day_params = {}

            for _, row in df.iterrows():
                # Check if this row belongs to our station
                if 'Station_ID' in df.columns and row['Station_ID'] != station_id:
                    continue

                # Get day of year
                day = row['Day_of_Year']

                if 'glorad' in df.columns:# Get parameter column
                    params= self.parse_dist_params(str(row['glorad']))
                elif 'wdsp^3' in df.columns:
                    params= self.parse_dist_params(str(row['wdsp^3']))
                else:
                    # Last resort - try second column
                    params =  self.parse_dist_params(str(row.iloc[1]))
                day_params[day] = params

            return day_params
        except Exception as e:
            print(f"Errorl oading file {filename}: {e}")
            return {}

    def load_station_data(self):
        """Load station data from FourierResults.csv."""
        print("Loading station data from FourierResults.csv...")

        try:
            df = pd.read_csv(self.fourier_results_path)

            for _, row in df.iterrows():
                station_id = row['Station_ID']
                column = row['Column']
                dist_type = row['Distribution_Type']
                params_file = row['Dist_Params_Filename']

                key = f"{station_id}_{column}"

                # Initialize weighting to 1.0
                self.weightings[key] = 1.0

                # Skip if no params file specified
                if not params_file or pd.isna(params_file) or params_file == "":
                    print(f"No parameter file specified for {station_id}, {column}. Skipping.")
                    continue

                # Load distribution parameters
                day_params = self.load_dist_params_file(params_file, station_id)

                # Store station data
                self.station_data[key] = {
                    'dist_type': dist_type,
                    'day_params': day_params
                }

            print(f"Loaded data for {len(self.station_data)} stations.")

        except Exception as e:
            print(f"Error loading station data: {e}")

    def load_demand_data(self):
        """Load demand data from FourierDemand.csv."""
        print("Loading demand data from FourierDemand.csv...")

        try:
            df = pd.read_csv(self.fourier_demand_path)
            demand_row = df.iloc[0]  # Assuming there's only one row for demand

            # Extract demand cosine form for average
            demand_avg_cosine = demand_row['Fourier_Avg_Cosine_Form']

            def evaluate_cosine_form(cosine_form, day_index):  # Function to evaluate cosine form for each day
                # Replace 'days' with day_index in the cosine form
                expr = cosine_form.replace('days', str(day_index))

                # Define the numpy sine and cosine functions
                cos = np.cos
                sin = np.sin
                pi = np.pi

                # Evaluate the expression
                return eval(expr)

            # Gen demand data for each day
            self.demand_data = {}
            for i, day in enumerate(self.days):
                self.demand_data[day] = evaluate_cosine_form(demand_avg_cosine, i)

            print(f"Loaded demand data for {len(self.demand_data)} days.")

        except Exception as e:
            print(f"Error loading demand data: {e}")

    def generate_random_sample(self, day, params, dist_type, seed=None):
        """Generate a random sample based on distribution type and parameters."""
        if seed is not None:
            np.random.seed(seed)

        if dist_type == 'weibull':
            # Weibull parameters (shape, loc, scale)
            shape, loc, scale = params
            if shape <= 0:
                shape = 1.0
            if scale <= 0:
                scale = 1.0

            return weibull_min.rvs(shape, loc=loc, scale=scale)

        elif dist_type == 'beta':
            # Beta params (a, b, loc, scale)
            a, b, loc, scale = params
            if a <= 0:
                a = 1.0
            if b <= 0:
                b = 1.0

            return beta.rvs(a, b, loc=loc, scale=scale)

        else:
            print(f"Unknown distribution type: {dist_type} for day {day}")
            return np.nan

    def calculate_total_energy(self, seed=None):
        """
        Calc the total energy production using current weightings.
        Returns a dictionary mapping days to total energy production.
        """
        total_energy = {day: 0 for day in self.days}

        for key, data in self.station_data.items():
            dist_type = data['dist_type']
            day_params = data['day_params']
            weighting = self.weightings[key]

            for day in self.days:
                if day in day_params:
                    # Generate sample and apply weighting
                    sample = self.generate_random_sample(day, day_params[day], dist_type, seed) * weighting
                    total_energy[day] += sample
                else:
                    # Use default parameters if day not found
                    if dist_type == 'weibull':
                        sample = weibull_min.rvs(1.0, loc=0, scale=1.0, random_state=seed) * weighting
                    else:  # beta
                        sample = beta.rvs(1.0, 1.0, loc=0, scale=1.0, random_state=seed) * weighting
                    total_energy[day] += sample

        return total_energy

    def calculate_loss(self, seed=None):
        """
        Calculate the loss (mean absolute error) between energy production and demand.
        """
        total_energy = self.calculate_total_energy(seed)

        # Calculate mean absolute error
        mae = 0
        for day in self.days:
            mae += abs(total_energy[day] - self.demand_data[day])

        return mae / len(self.days)

    def calculate_gradient(self, seed=None):
        """
        Calculate the gradient of the loss with respect to each weighting parameter.
        """
        epsilon = 1e-6 # Small value for numerical gradient calculation
        gradient = {}

        # Calculate loss with current weightings
        current_loss = self.calculate_loss(seed)

        # Calculate gradient for each weighting parameter
        for key in self.weightings:
            # Temporarily increase the weighting
            self.weightings[key] += epsilon

            # Calculate loss with increased weighting
            new_loss = self.calculate_loss(seed)

            # Calculate gradient
            gradient[key] = (new_loss - current_loss) / epsilon

            # Restore original weighting
            self.weightings[key] -= epsilon

        return gradient

    def optimize(self):
        """
        Optimize weightings using gradient descent.
        """
        print("Starting optimization...")

        # Load station data and demand data
        self.load_station_data()
        self.load_demand_data()

        # Check if data loaded successfully
        if not self.station_data or not self.demand_data:
            print("Error: Failed to load data. Optimization aborted.")
            return

        # Lists to store losses and weightings for plotting
        losses = []
        weighting_history = {key: [] for key in self.weightings}

        # Same seed for all iterations, might change this idk
        seed = 42

        # Gradient descent iterations
        for iteration in range(self.max_iterations):
            # Calculate current loss
            current_loss = self.calculate_loss(seed)
            losses.append(current_loss)

            # Store current weightings
            for key, weight in self.weightings.items():
                weighting_history[key].append(weight)
            # Print progress every 100 iterations
            if iteration %100 == 0:
                print(f"Iteration {iteration}, Loss: {current_loss}")
                print(f"Current weightings: {self.weightings}")

            # Calculate gradient
            gradient = self.calculate_gradient(seed)

            # Update weightings
            for key in self.weightings:
                # Apply gradient descent update
                self.weightings[key] -= self.learning_rate * gradient[key]

                # Ensure weightings stay positive
                self.weightings[key] = max(0.0, self.weightings[key])

            # Check for convergence
            if iteration > 0 and abs(losses[-1] - losses[-2]) < self.convergence_threshold:
                print(f"Converged after {iteration} iterations.")
                break

        # Print final results
        print("\nOptimization completed.")
        print(f"Final loss: {losses[-1]}")
        print("Final weightings:")
        for key, weight in self.weightings.items():
            print(f"  {key}: {weight}")

        # Save results
        self.save_results(losses, weighting_history)
        # Create plots
        self.create_plots(losses, weighting_history)

        return self.weightings



    def save_results(self, losses, weighting_history):
        """Save optimization results to CSV files."""
        # Save final weightings
        weightings_df = pd.DataFrame({
            'Station_Parameter': list(self.weightings.keys()),
            'Optimal_Weight': list(self.weightings.values())
        })
        weightings_df.to_csv('optimization_results/optimal_weightings.csv', index=False)

        # Save loss history
        loss_df = pd.DataFrame({
            'Iteration': list(range(len(losses))),
            'Loss': losses
        })
        loss_df.to_csv('optimization_results/loss_history.csv', index=False)

        # Save weighting history
        weighting_df = pd.DataFrame({
            'Iteration': list(range(len(losses)))
        })
        for key, history in weighting_history.items():
            weighting_df[key] = history

        weighting_df.to_csv('optimization_results/weighting_history.csv', index=False)

        print("Results saved to optimization_results directory.")

    def create_plots(self, losses, weighting_history):
        """Create plots for visualization."""
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.savefig('optimization_plots/loss_history.png')

        # Plot weighting history
        plt.figure(figsize=(12, 8))
        for key, history in weighting_history.items():
            plt.plot(history, label=key)

        plt.title('Weighting History')
        plt.xlabel('Iteration')
        plt.ylabel('Weight Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('optimization_plots/weighting_history.png')

        # Plot final energy vs demand
        plt.figure(figsize=(12, 6))

        # Calculate total energy with optimized weightings
        total_energy = self.calculate_total_energy()

        # Convert to lists for plotting
        days_list = list(range(len(self.days)))
        energy_values = [total_energy[day] for day in self.days]
        demand_values = [self.demand_data[day] for day in self.days]

        plt.plot(days_list, energy_values, label='Optimized Energy Production')
        plt.plot(days_list, demand_values, label='Demand')

        plt.title('Optimized Energy Production vs Demand')
        plt.xlabel('Day of Year')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('optimization_plots/energy_vs_demand.png')

        print("Plots saved to optimization_plots directory.")


# Example usage
if __name__ == "__main__":
    # Create weight optimizer
    optimizer = WeightOptimizer(
        fourier_results_path="FourierResults.csv",
        fourier_demand_path="FourierDemand.csv",
        learning_rate=0.01,
        max_iterations=500,
        convergence_threshold=1e-5
    )

    # Run optimization
    optimal_weightings = optimizer.optimize()