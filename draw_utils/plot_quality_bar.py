import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Define the file paths
real_data_paths = [
    'synthetic/default/real.csv',
    'synthetic/adult/real.csv',
    'synthetic/shoppers/real.csv',
    'synthetic/cardio_train/real.csv'
]

synthetic_data_paths = [
    # default
    ['sample_end_csv/tabsyn_default_ori.csv', 'sample_end_csv/tabsyn_default_-1_90w.csv',
     'sample_end_csv/tabddpm_default_ori.csv', 'sample_end_csv/tabddpm_default_new.csv'],
    # adult
    ['sample_end_csv/tabsyn_adult_ori.csv', 'sample_end_csv/tabsyn_adult_new.csv', 
     'sample_end_csv/tabddpm_adult_ori.csv', 'sample_end_csv/tabddpm_adult_new.csv'],
    # shoppers
    ['sample_end_csv/tabsyn_shoppers_ori.csv', 'sample_end_csv/tabsyn_shoppers_900000.csv',
     'sample_end_csv/tabddpm_shoppers_ori.csv', 'sample_end_csv/tabddpm_shoppers_new.csv'],
    # cardio_train
    ['sample_end_csv/tabsyn_cardio_ori.csv', 'sample_end_csv/tabsyn_cardio_-5_90w.csv',
     'sample_end_csv/tabddpm_cardio_ori.csv', 'sample_end_csv/tabddpm_cardio_-5_90w.csv']
]

colors = ['#7CCD7C', '#EEA2AD', '#20B2AA']

def plot_comparison(datasets):
    sns.set_context("notebook", font_scale=1)  # Adjust font scale to increase text size

    # Set all font sizes to 18
    plt.rc('font', size=40)  # controls default text sizes
    plt.rc('axes', titlesize=40)  # fontsize of the axes title
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the tick labels
    plt.rc('legend', fontsize=40)  # legend fontsize
    plt.rc('figure', titlesize=40)  # fontsize of the figure title

    fig, axes = plt.subplots(2, len(datasets), figsize=(43, 15))

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])  # Adjust the rect to reduce margins
    plt.subplots_adjust(wspace=0.28, hspace=0.28)

    def format_k(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}K'
        return f'{int(x)}'

    for i, dataname in enumerate(datasets):
        print(f"\n--- Processing dataset: {dataname} ---\n")

        # File paths
        real_data_path = real_data_paths[i]
        generated_data_path_tabsyn = synthetic_data_paths[i][0]
        generated_data_path_tabcutmix = synthetic_data_paths[i][1]

        # Load the data
        real_data = pd.read_csv(real_data_path)[:50]
        tabsyn_data = pd.read_csv(generated_data_path_tabsyn)[:50]
        tabcutmix_data = pd.read_csv(generated_data_path_tabcutmix)[:50]

        # Select a numerical feature and a categorical feature for each dataset
        if dataname == 'adult':
            num_feature = 'fnlwgt'
            cat_feature = 'relationship'
        elif dataname == 'default':
            num_feature = 'BILL_AMT4'
            cat_feature = 'PAY_0'
        elif dataname == 'shoppers':
            num_feature = 'ExitRates'
            cat_feature = 'VisitorType'
        elif dataname == 'magic':
            num_feature = 'Asym'  # Example feature, adjust as needed
            cat_feature = 'class'  # Example feature, adjust as needed
        elif dataname == 'cardio_train':
            num_feature = 'height'
            cat_feature = 'cholesterol'

        # Plot numerical feature (Density Plot)
        ax = axes[0, i]
        ax.grid()
        sns.kdeplot(real_data[num_feature], ax=ax, label='Real', color=colors[0], fill=True)
        sns.kdeplot(tabsyn_data[num_feature], ax=ax, label='TabSyn', color=colors[1], fill=True)
        sns.kdeplot(tabcutmix_data[num_feature], ax=ax, label='TabSyn+Tame', color=colors[2], fill=True)
        ax.set_title(f'{dataname.capitalize()}')
        if dataname == 'cardio_train':
            ax.set_title('Cardio')
            
        if i == 0:  # Only show ylabel for the first plot in the row
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('')
        if i == 0:  # Only show legend for the first plot in the row
            ax.legend(loc='upper right', framealpha=0.3, fontsize=26)
        else:
            ax.legend().remove()

        # Plot categorical feature (Bar Plot)
        ax = axes[1, i]
        ax.grid()
        real_counts = real_data[cat_feature].value_counts(normalize=True)
        tabsyn_counts = tabsyn_data[cat_feature].value_counts(normalize=True)
        tabcutmix_counts = tabcutmix_data[cat_feature].value_counts(normalize=True)

        df_bar = pd.DataFrame({
            'Category': real_counts.index,
            'Real': real_counts.values,
            'TabSyn': tabsyn_counts.reindex(real_counts.index, fill_value=0).values,
            'Tabsyn+Tame': tabcutmix_counts.reindex(real_counts.index, fill_value=0).values
        })

        df_bar_melted = df_bar.melt(id_vars='Category', var_name='Model', value_name='Proportion')
        # sns.barplot(x='Category', y='Proportion', hue='Model', data=df_bar_melted, ax=ax)
        sns.barplot(x='Category', y='Proportion', hue='Model', data=df_bar_melted, ax=ax, palette=colors)
        
        titles = ax.get_xticklabels()
        new_titles = []
        
        for title in titles:
            text = title.get_text()
            print(text)
            if text == ' Husband':
                new_titles.append('Hus.')
            elif text == ' Not-in-family':
                new_titles.append('Notin.')
            elif text == ' Own-child':
                new_titles.append('Kid')
            elif text == ' Unmarried':
                new_titles.append('Unma.')
            elif text == "Returning_Visitor":
                new_titles.append("Re.")
            elif text == "New_Visitor":
                new_titles.append("New")
            elif text == " Wife":
                new_titles.append("Wife")
            else:
                new_titles.append(text)
        
        print("处理完后")
        print(new_titles)
        
        ax.set_xticklabels(new_titles)
        
        ax.set_xlabel(cat_feature.capitalize())  # Set x-axis label to feature name
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        if i == 0:  # Only show ylabel for the first plot in the row
            ax.set_ylabel('Proportion')
        else:
            ax.set_ylabel('')
        if i == 0:  # Only show legend for the first plot in the row
            ax.legend(loc='upper right', framealpha=0.3, fontsize=26)
        else:
            ax.legend().remove()

        # Apply the formatter to the x-axis of the numerical feature plot
        if dataname == 'adult' or dataname == 'default':
            ax = axes[0, i]
            ax.xaxis.set_major_formatter(FuncFormatter(format_k))

    plt.savefig('quality_bar.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    datasets = ['default', 'adult', 'shoppers', 'cardio_train']
    plot_comparison(datasets)