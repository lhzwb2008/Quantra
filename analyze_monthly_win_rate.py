import pandas as pd

# Load the trades data
df = pd.read_csv('trades_full_period.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Print overall win rate
win_rate = df['is_win'].mean() * 100
print(f'Overall win rate: {win_rate:.1f}% ({df["is_win"].sum()}/{len(df)})')

# Print win rate by month
print('\nWin rate by month in 2022:')
for month in range(1, 13):
    month_df = df[df['Date'].dt.month == month]
    if len(month_df) > 0:
        month_wins = month_df['is_win'].sum()
        month_win_rate = month_df['is_win'].mean() * 100
        print(f'Month {month}: {month_win_rate:.1f}% ({month_wins}/{len(month_df)})')

# Print win rate by side (long vs short)
print('\nWin rate by side:')
for side in ['Long', 'Short']:
    side_df = df[df['side'] == side]
    if len(side_df) > 0:
        side_wins = side_df['is_win'].sum()
        side_win_rate = side_df['is_win'].mean() * 100
        print(f'{side}: {side_win_rate:.1f}% ({side_wins}/{len(side_df)})')

# Print win rate by exit reason
print('\nWin rate by exit reason:')
for reason in df['exit_reason'].unique():
    reason_df = df[df['exit_reason'] == reason]
    if len(reason_df) > 0:
        reason_wins = reason_df['is_win'].sum()
        reason_win_rate = reason_df['is_win'].mean() * 100
        print(f'{reason}: {reason_win_rate:.1f}% ({reason_wins}/{len(reason_df)})')
