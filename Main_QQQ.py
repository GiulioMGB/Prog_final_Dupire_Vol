import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import warnings


# Suppress warnings
def suppress_warnings():
    warnings.filterwarnings('ignore')


# Call the function to suppress warnings
suppress_warnings()


# Black-Scholes Model for call and put option pricing
def black_scholes_call(s, k, t, r, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def black_scholes_put(s, k, t, r, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


# Function to calculate implied volatility
def implied_volatility(option_price, s, k, t, r, option_type):
    if pd.isna(option_price):
        return np.nan
    else:
        def option_price_model(sigma):
            if option_type == 'call':
                return black_scholes_call(s, k, t, r, sigma) - option_price
            else:
                return black_scholes_put(s, k, t, r, sigma) - option_price
        try:
            iv = brentq(option_price_model, 1e-6, 10, maxiter=1000)
            return iv
        except (ValueError, RuntimeError):
            # Fallback or default IV value
            return np.nan  # Consider adjusting this to use a fallback method


file_path = "/Users/giuliogranati/PycharmProjects/Prog final Dupire Vol/QQQ_option_chain.xlsx"


option_chain = pd.read_excel(file_path)

option_chain['Expire Date'] = pd.to_datetime(option_chain['Expire Date'])
today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
option_chain['TimeToMaturity'] = (option_chain['Expire Date'] - today).dt.days / 365.25
option_chain['Last Price'] = pd.to_numeric(option_chain['Last Price'], errors='coerce')
option_chain['Strike Price'] = pd.to_numeric(option_chain['Strike Price'], errors='coerce')
option_chain['Is Put'] = option_chain['Is Put'].astype(str)

missing_prices_before = option_chain[option_chain['Last Price'].isna()]['Strike Price'].unique()
print(f"Missing Prices Before: {len(missing_prices_before)} Strikes")

def fill_missing_prices(row):
    if pd.isna(row['Last Price']):
        # Calculate the difference between the current strike and all other strikes
        diff = np.abs(option_chain['Strike Price'] - row['Strike Price'])
        diff[row.name] = np.inf  # Ignore the current row by setting its difference to infinity

        # Find the closest index with a non-NaN 'Last Price'
        non_nan_prices = option_chain[~option_chain['Last Price'].isna()]
        closest_idx = (diff.loc[non_nan_prices.index]).idxmin()

        # Print details about the operation


        # Return the Last Price from the closest strike with a non-NaN price
        return non_nan_prices.loc[closest_idx, 'Last Price']
    else:
        return row['Last Price']

option_chain['Last Price'] = option_chain.apply(fill_missing_prices, axis=1)

missing_prices_after = option_chain[option_chain['Last Price'].isna()]['Strike Price'].unique()
print(f"Missing Prices After: {len(missing_prices_after)} Strikes")

strikes_still_missing = set(missing_prices_before).intersection(set(missing_prices_after))
strikes_filled = set(missing_prices_before).difference(set(missing_prices_after))

print(f"Strikes still missing prices after fill: {len(strikes_still_missing)}")
print(f"Strikes filled: {len(strikes_filled)}")

Filled_Option_Chain = strikes_filled

# Define the file path where you want to save the Excel file
# output_file_path = 'C:\\Users\\fract\\Downloads\\Filled_Option_Chain.xlsx'

# Save the DataFrame to an Excel file
# option_chain.to_excel(output_file_path, index=False)

# print(f"Option chain saved to {output_file_path}")

S_0 = 434

r = 0.055

# Calculate implied volatility for all options
option_chain['Implied Volatility'] = option_chain.apply(
    lambda row: implied_volatility(
        option_price=row['Last Price'], s=S_0, k=row['Strike Price'],
        t=row['TimeToMaturity'], r=r, option_type='put' if row['Is Put'] == 'TRUE' else 'call'
    ), axis=1
)


def report_missing_ivs(option_chain, iv_col='Implied Volatility'):
    """Reports the number of options with missing implied volatility."""
    missing_iv_count = option_chain[option_chain[iv_col].isna()].shape[0]
    print(f"Options with missing {iv_col}: {missing_iv_count}")
    return missing_iv_count


# Report missing IVs before filling
print("Before filling missing IVs:")
missing_iv_before = report_missing_ivs(option_chain)


def fill_missing_ivs(row, iv_col='Implied Volatility'):
    if pd.isna(row[iv_col]):
        # Find the strike closest to the current one that has a non-missing IV
        closest_strikes = option_chain[np.isfinite(option_chain[iv_col])]['Strike Price']
        closest_strike = closest_strikes.iloc[(closest_strikes - row['Strike Price']).abs().argsort()[:1]]
        return option_chain.loc[closest_strike.index, iv_col].values[0]
    else:
        return row[iv_col]


option_chain['Implied Volatility'] = option_chain.apply(fill_missing_ivs, axis=1)

# Report missing IVs after filling
print("After filling missing IVs:")
missing_iv_after = report_missing_ivs(option_chain)

# Optionally, save the updated option chain with filled IVs
output_path_filled_iv = 'C:\\Users\\fract\\Downloads\\Option_Chain_Filled_IV.xlsx'
option_chain.to_excel(output_path_filled_iv, index=False)
print(f"Option chain with filled implied volatilities saved to {output_path_filled_iv}")

option_chain.sort_values(by=['TimeToMaturity', 'Strike Price'], inplace=True)

# Load the filled IV data
file_path_filled_iv = 'C:\\Users\\fract\\Downloads\\Option_Chain_Filled_IV.xlsx'
option_chain_filled_iv = pd.read_excel(file_path_filled_iv)

# Ensure proper data types and calculations for required columns
option_chain_filled_iv['Expire Date'] = pd.to_datetime(option_chain_filled_iv['Expire Date'])
today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
option_chain_filled_iv['TimeToMaturity'] = (option_chain_filled_iv['Expire Date'] - today).dt.days / 365.25
option_chain_filled_iv['Last Price'] = pd.to_numeric(option_chain_filled_iv['Last Price'], errors='coerce')
option_chain_filled_iv['Strike Price'] = pd.to_numeric(option_chain_filled_iv['Strike Price'], errors='coerce')
option_chain_filled_iv['Implied Volatility'] = pd.to_numeric(option_chain_filled_iv['Implied Volatility'],
                                                             errors='coerce')
option_chain_filled_iv['Is Put'] = option_chain_filled_iv['Is Put'].astype(str)

print(option_chain_filled_iv.pivot_table)

# Assuming option_chain_filled_iv is loaded from 'Option_Chain_Filled_IV.xlsx'


def count_strikes_per_maturity_before_iv_grid(df):
    """
    Counts the number of unique strikes for each maturity in the option chain DataFrame.

    Parameters:
    - df: DataFrame loaded from 'Option_Chain_Filled_IV.xlsx'.

    Returns:
    - A DataFrame with counts of strikes per maturity.
    """
    maturity_counts = df.groupby('TimeToMaturity')['Strike Price'].nunique().reset_index()
    maturity_counts.rename(columns={'Strike Price': 'Count of Strikes'}, inplace=True)
    return maturity_counts


before_iv_grid_counts = count_strikes_per_maturity_before_iv_grid(option_chain_filled_iv)
print("Before IV_grid:")
print(before_iv_grid_counts)

# Recreate the IV grid with the updated option_chain DataFrame
IV_grid_updated = option_chain_filled_iv.pivot_table(index='Strike Price',
                                                     columns='TimeToMaturity',
                                                     values='Implied Volatility',
                                                     aggfunc='mean'
                                                     )

# Check again for NaN values
print(IV_grid_updated)

# Define the file path where you want to save the IV_grid_updated Excel file
iv_grid_updated_file_path = 'C:\\Users\\fract\\Downloads\\IV_Grid_Updated.xlsx'

# Save the IV_grid_updated DataFrame to an Excel file
IV_grid_updated.to_excel(iv_grid_updated_file_path, index=True)

print(f"IV grid updated saved to {iv_grid_updated_file_path}")

# Interpolate missing values in IV_grid to handle non-uniform strike numbers
IV_grid_interpolated = IV_grid_updated.interpolate(method='linear',
                                                   axis=0).fillna(method='bfill').fillna(method='ffill'
                                                                                         )

# Save the interpolated IV grid to a new Excel file
# iv_grid_interpolated_file_path = 'C:\\Users\\fract\\Downloads\\IV_Grid_Interpolated.xlsx'
# IV_grid_interpolated.to_excel(iv_grid_interpolated_file_path, index=True)
# print(f"Interpolated IV grid saved to {iv_grid_interpolated_file_path}")


def count_strikes_per_maturity(IV_grid_interpolated):
    """
    Counts the number of strikes for each maturity in the IV grid.

    Parameters:
    - IV_grid: The implied volatility grid, a pivot table with maturities as columns and strikes as rows.

    Returns:
    - A DataFrame with the maturity (TimeToMaturity) and the corresponding count of strikes.
    """
    # Count non-NaN values across rows for each maturity
    maturity_counts = IV_grid_interpolated.notna().sum().reset_index()
    maturity_counts.columns = ['TimeToMaturity', 'Count of Strikes']
    return maturity_counts


after_iv_grid_counts = count_strikes_per_maturity(IV_grid_updated)
print("After IV_grid:")
print(after_iv_grid_counts)


# Adjust pandas display settings
pd.set_option('display.max_rows', None)  # Adjusts to display all rows
pd.set_option('display.max_columns', None)  # Adjusts to display all columns
pd.set_option('display.width', 1000)  # Adjusts the width of the display to prevent wrapping
pd.set_option('display.float_format', '{:.4f}'.format)  # Optional: Format the float values for better readability

# Assuming 'strikes' is already defined and contains unique strike prices from the option chain
strikes = np.sort(option_chain['Strike Price'].unique())

IV_grid_np = IV_grid_interpolated.to_numpy()

# Check if there are any NaN values in IV_grid_np
if np.isnan(IV_grid_np).any():
    print("There are still NaN values in the IV grid.")
else:
    print("No NaN values in the IV grid. Proceeding with local volatility calculations.")


# Now, perform gradient calculations on the numpy array
# Ensure that 'strikes' is also in a compatible shape for the operation
strikes_reshaped = np.sort(option_chain['Strike Price'].unique()).reshape(-1, 1)

# After filtering and sorting your DataFrame
if len(option_chain) > 1:
    strikes = option_chain['Strike Price'].unique()
    time_to_maturities = option_chain['TimeToMaturity'].unique()

    unique_strikes_in_chain = option_chain['Strike Price'].nunique()
    unique_strikes_in_grid = IV_grid_updated.index.nunique()

    print(f"Unique strikes in option chain: {unique_strikes_in_chain}")
    print(f"Unique strikes in IV_grid: {unique_strikes_in_grid}")

    missing_strikes = set(option_chain['Strike Price'].unique()) - set(IV_grid_updated.index)
    print(f"Missing Strikes: {sorted(missing_strikes)}")

    print("Unique Strikes:", len(strikes))
    print("Unique Time to Maturities:", len(time_to_maturities))

# Assuming time_to_maturity_np represents unique time to maturity values in a numpy array format
# Assuming time_to_maturity_np is reshaped correctly
time_to_maturities_1D = np.sort(option_chain['TimeToMaturity'].unique())
dIV_dT = np.gradient(IV_grid_np, axis=1, edge_order=2) / np.gradient(time_to_maturities_1D)


# Calculate the first derivative of IV with respect to strike price (dIV/dK)
dIV_dK = np.gradient(IV_grid_np, axis=0, edge_order=2) / np.gradient(strikes_reshaped, axis=0)

# Calculate the second derivative of IV with respect to strike price (d2IV/dK2)
d2IV_dK2 = np.gradient(dIV_dK, axis=0, edge_order=2) / np.gradient(strikes_reshaped, axis=0)

# Calculate local volatility using the simplified Dupire's formula (ignoring dividends)
local_volatility = np.sqrt((2 * dIV_dT + r * strikes_reshaped * dIV_dK) / (strikes_reshaped**2 * d2IV_dK2))

# Handle potential issues like division by zero or negative square roots
local_volatility = np.where(np.isnan(local_volatility) | np.isinf(local_volatility) | (local_volatility < 0), np.nan,
                            local_volatility
                            )

strikes_flat = strikes_reshaped.flatten()
# Taking the mean across all maturities for simplification
local_vol_flat = np.nanmean(np.sqrt(local_volatility), axis=1)


# Constructing the DataFrame
df = pd.DataFrame({
    'Strike': strikes_flat,
    'Local Volatility': local_vol_flat
})

# Merge with the original option_chain to include the Last Price
# This assumes the option_chain is sorted and indexed to align with your calculated local volatility
df_merged = pd.merge(option_chain[['Strike Price', 'Last Price']].drop_duplicates(), df, left_on='Strike Price',
                     right_on='Strike',
                     how='inner'
                     )

# Display the DataFrame
print(df_merged[['Strike', 'Last Price', 'Local Volatility']])

# After calculating the local volatility and creating df_merged DataFrame
df_merged['Moneyness'] = df_merged['Strike'] / S_0

# Filter df_merged to only include options between 80% to 120% moneyness
df_filtered = df_merged[(df_merged['Moneyness'] >= 0.8) & (df_merged['Moneyness'] <= 1.2)]

# Display the DataFrame with Moneyness
print(df_filtered[['Strike', 'Last Price', 'Local Volatility', 'Moneyness']])


def display_statistics(df):
    """
    Displays the maximum, minimum, and mean for each column in the DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    """
    print("Maximum Values:\n", df.max())
    print("\nMinimum Values:\n", df.min())
    print("\nMean Values:\n", df.mean())


def save_dataframe_to_excel(df, file_path):
    """
    Saves the DataFrame to an Excel file.

    Parameters:
    - df: DataFrame to save.
    - file_path: File path where the Excel file will be saved.
    """
    df.to_excel(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


# Assuming df_merged is your DataFrame containing strike prices, last prices, and local volatilities
# Display max, min, mean values
display_statistics(df_merged)

# Define the file path for saving df_merged
merged_file_path = 'C:\\Users\\fract\\Downloads\\Merged_Local_Volatility.xlsx'

# Save df_merged to an Excel file
save_dataframe_to_excel(df_merged, merged_file_path)


def remove_outliers(df, column_name, threshold=2.0):
    """
    Identifies and removes rows from a DataFrame where the value in the specified column exceeds the given threshold.

    Parameters:
    - df: DataFrame from which to remove high values.
    - column_name: Name of the column to check for high values.
    - threshold: Value threshold above which rows are considered outliers and excluded.

    Returns:
    - DataFrame without outliers.
    """

    # Filter the DataFrame to exclude rows where the column value exceeds the threshold
    df_filtered = df[df[column_name] <= threshold]
    return df_filtered


# Example usage:
# df_no_outliers = remove_outliers(df_merged, 'Local Volatility', 2.0)


# Assuming 'df_merged' is your DataFrame and 'Local Volatility' is the column to check
df_no_outliers = remove_outliers(df_filtered, 'Local Volatility')


def display_adjusted_iv_statistics(df, column_name='Local Volatility', threshold=2.0):
    """
    Filters out extreme outliers based on a threshold before displaying statistics.

    Parameters:
    - df: DataFrame to analyze.
    - column_name: The column for implied volatility.
    - threshold: Volatility values above this threshold will be considered outliers and excluded.
    """
    # Exclude extreme outliers
    filtered_df = df[df[column_name] <= threshold]

    max_iv = filtered_df[column_name].max()
    min_iv = filtered_df[column_name].min()
    mean_iv = filtered_df[column_name].mean()

    print(f"Adjusted Implied Volatility Statistics (excluding values > {threshold}):")
    print(f"Maximum {column_name}: {max_iv:.4f}")
    print(f"Minimum {column_name}: {min_iv:.4f}")
    print(f"Mean {column_name}: {mean_iv:.4f}")


# Example usage with your DataFrame:
display_adjusted_iv_statistics(df_filtered, 'Local Volatility', 2.0)

# Since df_no_outliers might not have 'TimeToMaturity', merge it to include this data
df_no_outliers_with_ttm = pd.merge(
    df_no_outliers,
    option_chain_filled_iv[['Strike Price', 'TimeToMaturity']],
    on='Strike Price',
    how='left'
)

# Now df_no_outliers_with_ttm includes TimeToMaturity and has outliers removed, which can be used for plotting

# Sample data for demonstration purposes after excluding outliers
strikes = np.array(df_no_outliers_with_ttm['Strike'])
ttm = np.array(df_no_outliers_with_ttm['TimeToMaturity'])
vol = np.array(df_no_outliers_with_ttm['Local Volatility'])

# Creating a 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_trisurf(strikes, ttm, vol, cmap='viridis', edgecolor='none')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Local Volatility')
ax.set_title('Volatility Surface')

plt.show()
