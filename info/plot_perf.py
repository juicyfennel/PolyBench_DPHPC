import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the CSV file
df = pd.read_csv("perf_stats.csv")

# Select only numeric columns
# Select only the desired columns
selected_columns = ["page-faults", "branch-misses", "cache-misses", "time-elapsed"]

# Filter the dataframe to only include the selected columns
df_selected = df[selected_columns]


# Use Seaborn to create a pairplot of every numeric column against each other
sns.pairplot(df_selected, y_vars=["time-elapsed"])

# Show the plots
plt.show()
