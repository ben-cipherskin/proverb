import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('macosx')

file_path1 = '/Users/cipherskin/PycharmProjects/proverb/data/demographic_data/explorer_download.csv'
file_path2 = '/Users/cipherskin/PycharmProjects/proverb/data/demographic_data/explorer_download2.csv'
data1 = pd.read_csv(file_path1, header=3)
data2 = pd.read_csv(file_path2, header=3)

for column in data1.columns:
    # Combine the text in the first and second rows
    combined_text = column + '; ' + data1.loc[0, column]

    # Rename the 'OldColumn' to 'NewColumn'
    data1 = data1.rename(columns={f'{column}': f'{combined_text}'})
for column in data2.columns:
    # Combine the text in the first and second rows
    combined_text = column + '; ' + data2.loc[0, column]

    # Rename the 'OldColumn' to 'NewColumn'
    data2 = data2.rename(columns={f'{column}': f'{combined_text}'})
# Drop the first row
data1 = data1.drop(0)
data2 = data2.drop(0)
age = data1.iloc[:19, 0].tolist()

fig, axes = plt.subplots(nrows=2, ncols=2)
box_data1, positions1, labels1 = [], [], []
box_data2, positions2, labels2 = [], [], []

for i, column in enumerate(data1.columns):
    if 'Rate per 100,000' in column:
        pass
    elif i == 0:
        continue
    else:
        continue
    col_name = data1.columns[i].split(';')[0]
    rate_per_1000 = data1.iloc[:19, i]
    rate_per_1000 = ['0.0' if cell == '^' else cell for cell in rate_per_1000]
    rate_per_1000 = [float(cell) for cell in rate_per_1000]
    axes[0, 0].plot(age, rate_per_1000, label=f'{col_name}')
    axes[0, 0].set_xticklabels(age, rotation=45)
    box_data1.append(rate_per_1000)
    positions1.append(i)
    labels1.append(f'{col_name}')
    if 'All' in col_name:
        all_races = rate_per_1000
for i, column in enumerate(data2.columns):
    if 'Rate per 100,000' in column:
        pass
    elif i == 0:
        continue
    else:
        continue
    col_name = data2.columns[i].split(';')[0]
    rate_per_1000 = data2.iloc[:19, i]
    rate_per_1000 = ['0.0' if cell == '^' else cell for cell in rate_per_1000]
    rate_per_1000 = [float(cell) for cell in rate_per_1000]
    axes[0, 1].plot(age, rate_per_1000, label=f'{col_name}')
    axes[0, 1].set_xticklabels(age, rotation=45)
    box_data2.append(rate_per_1000)
    positions2.append(i)
    labels2.append(f'{col_name}')
    if 'All' in col_name:
        all_races2 = rate_per_1000

axes[0, 0].set_title('Breast Cancer in Women: Age vs Rate per 100,000 by race/ethnicity')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Rate per 100,000')

axes[0, 1].set_title('Breast Cancer in Women: Age vs Rate per 100,000 by cancer type')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Age Group')
axes[0, 1].set_ylabel('Rate per 100,000')

axes[1, 0].set_title('Box and Whisker of Above')
axes[1, 0].boxplot(box_data1, positions=positions1)
axes[1, 0].set_xticks(positions1)
axes[1, 0].set_xticklabels(labels1, rotation=10)
axes[1, 0].set_ylabel('Rate per 100,000')

axes[1, 1].set_title('Box and Whisker of Above')
axes[1, 1].boxplot(box_data2, positions=positions2)
axes[1, 1].set_xticks(positions2)
axes[1, 1].set_xticklabels(labels2, rotation=10)
axes[1, 1].set_ylabel('Rate per 100,000')


avg_num = 0
for i, num in enumerate(age):
    if i == 0:
        temp_num = 0
    elif num == '85+':
        temp_num = 85
    else:
        temp_num = sum([float(val) for val in num.split('-')])/2
    avg_num += temp_num*float(all_races[i])
avg_age = avg_num / sum(all_races)
print(f'Average age for women with Breast Cancer: {round(avg_age)} years')
plt.show()
