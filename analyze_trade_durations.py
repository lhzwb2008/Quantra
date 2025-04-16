import pandas as pd

# Load the trades data
df = pd.read_csv('trades_full_period.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['duration'] = df['exit_time'] - df['entry_time']

# Print overall average duration
print(f'Overall average duration: {df["duration"].mean()}')

# Print average duration by month
print('\nAverage trade duration by month:')
for month in range(1, 13):
    month_df = df[df['Date'].dt.month == month]
    if len(month_df) > 0:
        print(f'Month {month}: {month_df["duration"].mean()}')

# Print average duration by side (long vs short)
print('\nAverage duration by side:')
for side in ['Long', 'Short']:
    side_df = df[df['side'] == side]
    if len(side_df) > 0:
        print(f'{side}: {side_df["duration"].mean()}')

# Print average duration for winning vs losing trades
print('\nAverage duration for winning vs losing trades:')
win_df = df[df['is_win'] == True]
loss_df = df[df['is_win'] == False]
if len(win_df) > 0:
    print(f'Winning trades: {win_df["duration"].mean()}')
if len(loss_df) > 0:
    print(f'Losing trades: {loss_df["duration"].mean()}')
