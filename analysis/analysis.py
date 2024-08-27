# SCRIPT for running the analysis and displaying the results 
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import ast
import re
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# some functions
def tensor_to_int(tensor):
    return int(tensor.item())

def string_to_tensor(tensor_str):
    value = ast.literal_eval(tensor_str.replace("tensor", "").strip())
    return torch.tensor(value)

adder = 0
def cum_mean(df):
    global adder
    l = []
    for i in range(len(df)):
        if (i != 0):
            if (i%200 == 0):
                adder = i
        l.append(df["output"][adder:i].mean())
    df["cumulative_rates"] = l
    df['cumulative_rates'].fillna(0, inplace=True)
    return df

# load the data
file_path = "C:/Users/Privat/OneDrive/Dokumente/SEMESTER 8/4.Modelling of synaptic plasticity/SNNs Project/SNNs Project [Implementation]_v1.0/NeuroNetV1-main/NeuroNetV1-main/output_data/"
run_23082024_orientation0_no1_kernel1 = pd.read_csv(file_path + "run_23082024_orientation0_no1_kernel1.csv")
run_23082024_orientation45_no1_kernel1 = pd.read_csv(file_path + "run_23082024_orientation45_no1_kernel1.csv")
run_23082024_orientation90_no1_kernel1 = pd.read_csv(file_path + "run_23082024_orientation90_no1_kernel1.csv")
run_23082024_orientation135_no1_kernel1 = pd.read_csv(file_path + "run_23082024_orientation135_no1_kernel1.csv")

run_23082024_orientation0_no1_kernel2 = pd.read_csv(file_path + "run_23082024_orientation0_no1_kernel2.csv")
run_23082024_orientation45_no1_kernel2 = pd.read_csv(file_path + "run_23082024_orientation45_no1_kernel2.csv")
run_23082024_orientation90_no1_kernel2 = pd.read_csv(file_path + "run_23082024_orientation90_no1_kernel2.csv")
run_23082024_orientation135_no1_kernel2 = pd.read_csv(file_path + "run_23082024_orientation135_no1_kernel2.csv")

run_23082024_orientation0_no1_kernel3 = pd.read_csv(file_path + "run_23082024_orientation0_no1_kernel3.csv")
run_23082024_orientation45_no1_kernel3 = pd.read_csv(file_path + "run_23082024_orientation45_no1_kernel3.csv")
run_23082024_orientation90_no1_kernel3 = pd.read_csv(file_path + "run_23082024_orientation90_no1_kernel3.csv")
run_23082024_orientation135_no1_kernel3 = pd.read_csv(file_path + "run_23082024_orientation135_no1_kernel3.csv")

run_23082024_orientation0_no1_kernel4 = pd.read_csv(file_path + "run_23082024_orientation0_no1_kernel4.csv")
run_23082024_orientation45_no1_kernel4 = pd.read_csv(file_path + "run_23082024_orientation45_no1_kernel4.csv")
run_23082024_orientation90_no1_kernel4 = pd.read_csv(file_path + "run_23082024_orientation90_no1_kernel4.csv")
run_23082024_orientation135_no1_kernel4 = pd.read_csv(file_path + "run_23082024_orientation135_no1_kernel4.csv")

# prepare the data for ANOVA 
inter_data = pd.concat([run_23082024_orientation0_no1_kernel1, run_23082024_orientation45_no1_kernel1, run_23082024_orientation90_no1_kernel1, run_23082024_orientation135_no1_kernel1], axis=0, ignore_index = True)
inter_data = inter_data.dropna()
inter_data['metadata'] = inter_data['metadata'].apply(ast.literal_eval)
inter_data['output'] = inter_data['output'].apply(string_to_tensor)
inter_data['output'] = inter_data['output'].apply(tensor_to_int)
group_size = 200
mean_output = inter_data['output'].groupby(inter_data.index // group_size).mean()
mean_metadata = inter_data['metadata'].groupby(inter_data.index // group_size).first()
inter_data = pd.DataFrame({
    'output': mean_output.values,
    'metadata': mean_metadata.values
})
inter_data = cum_mean(inter_data)
inter_data = inter_data[['output', 'metadata']]
inter_data[['orientation', 'position', 'sample']] = pd.DataFrame(inter_data['metadata'].tolist(), index=inter_data.index)
inter_data = inter_data.drop(columns=['metadata'])
df = inter_data

# run ANOVA per data loaded (that is per kernel)
model = ols('output ~ C(orientation) + C(position) + C(orientation):C(position)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=3)

# display boxplots in analogy to the ANOVA
print(anova_table)
plt.figure(figsize=(12, 6))
sns.boxplot(x='orientation', y='output', hue='position', data=df)
plt.title('Boxplot of Output by Orientation and Position')
plt.xlabel('Rotation')
plt.ylabel('Output')
plt.legend(title='Position')
plt.show()

# prepare data for vizualization
inter_data = pd.concat([run_23082024_orientation0_no1_kernel3, run_23082024_orientation45_no1_kernel3, run_23082024_orientation90_no1_kernel3, run_23082024_orientation135_no1_kernel3], axis=0, ignore_index = True)
inter_data = inter_data.dropna()
inter_data['metadata'] = inter_data['metadata'].apply(ast.literal_eval)
inter_data['output'] = inter_data['output'].apply(string_to_tensor)
inter_data['output'] = inter_data['output'].apply(tensor_to_int)
inter_data = cum_mean(inter_data)
inter_data = inter_data[['output', 'metadata', 'cumulative_rates', 'time_point']]
inter_data[['orientation', 'position', 'sample']] = pd.DataFrame(inter_data['metadata'].tolist(), index=inter_data.index)
inter_data = inter_data.drop(columns=['metadata'])
df = inter_data

# vizialize the data per data loaded (that is per kernel)
g = sns.FacetGrid(df, row='orientation', hue='time_point', palette='viridis', height=5, aspect=2)
g.map(sns.stripplot, 'position', 'cumulative_rates', dodge=True, alpha=0.6, size=4, marker='o')
g.set_axis_labels('Position', 'Cumulative Rates')
g.set_titles(row_template='Orientation: {row_name}')
norm = mpl.colors.Normalize(vmin=0, vmax=len(df['time_point'].unique()) - 1)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar_ax = g.fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.ax.set_ylabel('Time Point')
cbar.set_ticks(np.arange(len(df['time_point'].unique())) + 0.5) 
tick_labels = df['time_point'].unique()
cbar.set_ticklabels(tick_labels)  
cbar.set_ticks(np.arange(0, len(tick_labels), step=4))  
cbar.set_ticklabels(tick_labels[::4])  
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Scatterplot of Kernel 3') # give the scatterplot a suitable title
plt.show()

