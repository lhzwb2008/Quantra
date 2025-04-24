import pandas as pd
from datetime import time

# Read the CSV file
print("Reading data from momentum/tqqq.csv...")
df = pd.read_csv('momentum/tqqq.csv')

# Convert DateTime to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract date and time components
df['Date'] = df['DateTime'].dt.date
df['Time'] = df['DateTime'].dt.time

# Display the first few rows to understand the data structure
print("Original data sample:")
print(df.head())
print(f"Total rows in original data: {len(df)}")

# Filter data to include only regular market hours (9:30 AM - 4:00 PM)
market_open = time(9, 30)
market_close = time(16, 0)

# Create a mask for regular market hours
market_hours_mask = (df['Time'] >= market_open) & (df['Time'] <= market_close)
df_market_hours = df[market_hours_mask].copy()

print(f"Rows after filtering for market hours: {len(df_market_hours)}")

# Group by date to find the first (9:30 AM) and last (4:00 PM) data points for each day
# For each day, get the first row (9:30 AM opening price)
opening_prices = df_market_hours.groupby('Date').first().reset_index()
opening_prices = opening_prices[['Date', 'Open']].rename(columns={'Open': 'DayOpen'})

# For each day, get the last row (4:00 PM closing price)
closing_prices = df_market_hours.groupby('Date').last().reset_index()
closing_prices = closing_prices[['Date', 'Close']].rename(columns={'Close': 'DayClose'})

# Merge the opening and closing prices back to the main dataframe
df_market_hours = pd.merge(df_market_hours, opening_prices, on='Date', how='left')
df_market_hours = pd.merge(df_market_hours, closing_prices, on='Date', how='left')

# Display the first few rows with the new columns
print("Data with day open and close prices:")
print(df_market_hours.head())

# Check if 'Year' column exists, if not, extract it from DateTime
if 'Year' not in df_market_hours.columns:
    df_market_hours['Year'] = df_market_hours['DateTime'].dt.year

# Create a new dataframe with the filtered data
filtered_df = df_market_hours[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'DayOpen', 'DayClose']].copy()

# Save the filtered data to a new CSV file
filtered_df.to_csv('momentum/tqqq_market_hours.csv', index=False)

print("Filtered data saved to 'momentum/tqqq_market_hours.csv'")

# Display some statistics about the filtered data
print(f"Total number of trading days: {filtered_df['DateTime'].dt.date.nunique()}")
print(f"Date range: {filtered_df['DateTime'].min().date()} to {filtered_df['DateTime'].max().date()}")
print(f"Average number of data points per day: {len(filtered_df) / filtered_df['DateTime'].dt.date.nunique():.2f}")

# Check if there are any days with missing 9:30 AM or 4:00 PM data points
market_open_time = time(9, 30)
market_close_time = time(16, 0)

# Group by date and check if each day has data at 9:30 AM and 4:00 PM
day_times = df_market_hours.groupby('Date')['Time'].apply(list).reset_index()
days_missing_open = []
days_missing_close = []

for _, row in day_times.iterrows():
    if market_open_time not in row['Time']:
        days_missing_open.append(row['Date'])
    if market_close_time not in row['Time']:
        days_missing_close.append(row['Date'])

print(f"Number of days missing 9:30 AM data: {len(days_missing_open)}")
print(f"Number of days missing 4:00 PM data: {len(days_missing_close)}")

if days_missing_open:
    print("Sample of days missing 9:30 AM data:")
    print(days_missing_open[:5])
    
if days_missing_close:
    print("Sample of days missing 4:00 PM data:")
    print(days_missing_close[:5])
