import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm # For OLS

def conduct_channel_analysis(df, region, channel, output_file):
    """
    Performs statistical analysis for a given channel and region,
    writing the output to a specified file.
    Includes descriptive stats, normality, skewness, kurtosis,
    correlation, Levene's, Kruskal-Wallis, AND
    descriptive stats grouped by Image_Group and P_Group.
    """
    df_channel = df[df['Channel'] == channel].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Ensure categorical columns are indeed categorical with appropriate types
    # This is crucial for proper handling of categorical variables
    if 'Image_Group' in df_channel.columns:
        df_channel['Image_Group'] = df_channel['Image_Group'].astype('category')
    if 'P_Group' in df_channel.columns:
        df_channel['P_Group'] = df_channel['P_Group'].astype('category')

    with open(output_file, 'a') as f: # Use 'a' for append mode
        f.write(f"\n--- Statistical Analysis for {region} (Channel {channel}) ---\n")
        f.write("-" * 60 + "\n") # Separator for readability

        # --- Overall Descriptive Statistics for the Channel & Region ---
        f.write("\n## Overall Descriptive Statistics:\n")
        desc = df_channel[f'Normalized_Fluorescence_Intensity_{region}'].describe()
        f.write(desc.to_string() + "\n") # to_string() for proper formatting

        # --- Normality Test (Shapiro-Wilk) ---
        f.write("\n## Normality Test (Shapiro-Wilk):\n")
        
        # Filter out NaN values before normality test
        y_for_normality = df_channel[f'Normalized_Fluorescence_Intensity_{region}'].dropna()
        if len(y_for_normality) > 3: # Shapiro-Wilk requires at least 3 data points
            stat_shapiro, p_shapiro = stats.shapiro(y_for_normality)
            f.write(f"Statistic = {stat_shapiro:.4f}, p-value = {p_shapiro:.4f}\n")
            if p_shapiro > 0.05:
                f.write("- The data appears to be normally distributed (p > 0.05).\n")
            else:
                f.write("- The data does NOT appear to be normally distributed (p <= 0.05).\n")
        else:
            f.write("Not enough data points (less than 4) to perform Shapiro-Wilk Test for Normality.\n")


        # --- Skewness and Kurtosis ---
        f.write("\n## Skewness and Kurtosis:\n")
        skewness = stats.skew(df_channel[f'Normalized_Fluorescence_Intensity_{region}'].dropna())
        kurtosis = stats.kurtosis(df_channel[f'Normalized_Fluorescence_Intensity_{region}'].dropna())
        f.write(f"Skewness: {skewness:.4f}\n")
        f.write(f"Kurtosis: {kurtosis:.4f}\n")

        # --- Correlation Analysis (Spearman) ---
        f.write("\n## Correlation Analysis (Spearman):\n")
        
        # Filter out NaNs for correlation
        temp_df_corr = df_channel.dropna(subset=[f'Normalized_Fluorescence_Intensity_{region}'])
        
        if 'Image_Group' in temp_df_corr.columns:
            if len(temp_df_corr) > 0 and temp_df_corr['Image_Group'].nunique() > 1: # Ensure enough data and categories
                corr_image_group, p_image_group = stats.spearmanr(
                    temp_df_corr[f'Normalized_Fluorescence_Intensity_{region}'],
                    temp_df_corr['Image_Group'].cat.codes
                )
                f.write(f"Spearman Correlation with Image Group: r = {corr_image_group:.4f}, p-value = {p_image_group:.4f}\n")
            else:
                f.write("Skipping Spearman Correlation for Image Group: Not enough data or categories after dropping NaNs.\n")
        
        if 'P_Group' in temp_df_corr.columns:
            if len(temp_df_corr) > 0 and temp_df_corr['P_Group'].nunique() > 1: # Ensure enough data and categories
                corr_p_group, p_p_group = stats.spearmanr(
                    temp_df_corr[f'Normalized_Fluorescence_Intensity_{region}'],
                    temp_df_corr['P_Group'].cat.codes
                )
                f.write(f"Spearman Correlation with P Group: r = {corr_p_group:.4f}, p-value = {p_p_group:.4f}\n")
            else:
                f.write("Skipping Spearman Correlation for P Group: Not enough data or categories after dropping NaNs.\n")

        # --- Levene's Test for Equal Variances ---
        f.write("\n## Levene's Test for Equal Variances:\n")
        if 'Image_Group' in df_channel.columns and len(df_channel['Image_Group'].unique()) > 1:
            levene_groups_image = [df_channel[df_channel['Image_Group'] == group][f'Normalized_Fluorescence_Intensity_{region}'].dropna() for group in df_channel['Image_Group'].unique()]
            # Levene's test requires at least 2 data points per group, and at least 2 groups with data
            if len(levene_groups_image) >= 2 and all(len(g) >= 2 for g in levene_groups_image if len(g) > 0): 
                stat_levene_image_group, p_levene_image_group = stats.levene(*[g for g in levene_groups_image if len(g) > 0])
                f.write(f"Levene’s Test for Image Group: Statistic = {stat_levene_image_group:.4f}, p-value = {p_levene_image_group:.4f}\n")
            else:
                f.write("Skipping Levene's Test for Image Group: Not enough data in all groups (need at least 2 non-empty groups with 2+ data points each).\n")
        
        if 'P_Group' in df_channel.columns and len(df_channel['P_Group'].unique()) > 1:
            levene_groups_p = [df_channel[df_channel['P_Group'] == group][f'Normalized_Fluorescence_Intensity_{region}'].dropna() for group in df_channel['P_Group'].unique()]
            if len(levene_groups_p) >= 2 and all(len(g) >= 2 for g in levene_groups_p if len(g) > 0):
                stat_levene_p_group, p_levene_p_group = stats.levene(*[g for g in levene_groups_p if len(g) > 0])
                f.write(f"Levene’s Test for P Group: Statistic = {stat_levene_p_group:.4f}, p-value = {p_levene_p_group:.4f}\n")
            else:
                f.write("Skipping Levene's Test for P Group: Not enough data in all groups (need at least 2 non-empty groups with 2+ data points each).\n")


        # --- Kruskal-Wallis Test (Non-parametric Group Comparison) ---
        f.write("\n## Kruskal-Wallis Test:\n")
        if 'Image_Group' in df_channel.columns and len(df_channel['Image_Group'].unique()) > 1:
            kruskal_groups_image = [df_channel[df_channel['Image_Group'] == group][f'Normalized_Fluorescence_Intensity_{region}'].dropna() for group in df_channel['Image_Group'].unique()]
            # Kruskal-Wallis requires at least 2 non-empty groups
            if len(kruskal_groups_image) >= 2 and all(len(g) > 0 for g in kruskal_groups_image):
                kruskal_image_group_stat, kruskal_image_group_p = stats.kruskal(*[g for g in kruskal_groups_image if len(g) > 0])
                f.write(f"Kruskal-Wallis Test for Image Group: Statistic = {kruskal_image_group_stat:.4f}, p-value = {kruskal_image_group_p:.4f}\n")
            else:
                f.write("Skipping Kruskal-Wallis Test for Image Group: Not enough data in all groups (need at least 2 non-empty groups).\n")
        
        if 'P_Group' in df_channel.columns and len(df_channel['P_Group'].unique()) > 1:
            kruskal_groups_p = [df_channel[df_channel['P_Group'] == group][f'Normalized_Fluorescence_Intensity_{region}'].dropna() for group in df_channel['P_Group'].unique()]
            if len(kruskal_groups_p) >= 2 and all(len(g) > 0 for g in kruskal_groups_p):
                kruskal_p_group_stat, kruskal_p_group_p = stats.kruskal(*[g for g in kruskal_groups_p if len(g) > 0])
                f.write(f"Kruskal-Wallis Test for P Group: Statistic = {kruskal_p_group_stat:.4f}, p-value = {kruskal_p_group_p:.4f}\n")
            else:
                f.write("Skipping Kruskal-Wallis Test for P Group: Not enough data in all groups (need at least 2 non-empty groups).\n")


        # --- Descriptive Statistics Grouped by Image Group and P Group ---
        f.write(f"\n## Descriptive Statistics by Image Group and P Group (for {region}, Channel {channel}):\n")
        
        # Ensure we have the necessary columns for grouping
        required_cols = [f'Normalized_Fluorescence_Intensity_{region}', 'Image_Group', 'P_Group']
        if all(col in df_channel.columns for col in required_cols):
            # Drop rows where the key columns are NaN before grouping
            grouped_df = df_channel.dropna(subset=required_cols)
            
            if not grouped_df.empty:
                # Group by Image_Group and P_Group and describe the fluorescence intensity
                grouped_desc = grouped_df.groupby(['Image_Group', 'P_Group'])[f'Normalized_Fluorescence_Intensity_{region}'].describe()
                f.write(grouped_desc.to_string() + "\n")
            else:
                f.write("No data available for grouping after dropping NaNs.\n")
        else:
            f.write("Skipping grouped descriptive statistics: Missing 'Image_Group' or 'P_Group' columns, or the target fluorescence column.\n")
        
        f.write("\n" + "=" * 80 + "\n") # End of channel analysis separator

# Helper functions for cleaning and categorizing
def categorize_p_value(name):
    match = re.search(r'p(\d+)', name.lower())
    if match:
        return f'p{match.group(1)}'
    return 'flat'

def categorize_image_group(letter):
    if letter in 't':
        return 'Thin'
    elif letter in 'm':
        return 'Mushroom'
    elif letter in 's':
        return 'Stubby'
    elif letter == 'f':  # Assuming 'flat' is an acceptable letter
        return 'Flat'
    return None

def heatmap_generated(results_df):
    grouped = results_df.groupby(['Image_Group', 'P_Group', 'Channel']).agg({
        'Normalized_Fluorescence_Intensity_COMPOUND': 'mean',
        'Normalized_Fluorescence_Intensity_SOMA': 'mean',
        'Normalized_Fluorescence_Intensity_NEURITE': 'mean'
    }).unstack('Channel')


    # Define color palettes
    paxillin_palette = sns.color_palette("Reds")
    integrin_palette = sns.color_palette("Blues")
    neurite_palette = sns.color_palette("Greens")
    # Define font sizes for easier adjustment
    TITLE_FONT_SIZE = 8
    LABEL_FONT_SIZE = 4
    TICK_LABEL_FONT_SIZE = 4
    COLORBAR_LABEL_FONT_SIZE = 4
    COLORBAR_TICK_FONT_SIZE = 4
    # Loop through each channel and create heatmaps for compound, soma, and neurites
    for channel in [0, 1, 2]:  # Assuming three channels
        # Select the appropriate color palette for each channel
        palette = paxillin_palette if channel == 0 else integrin_palette if channel == 1 else neurite_palette
        
        # Plot heatmap for Compound fluorescence
        plt.figure(figsize=(4, 2))
        heatmap_data = grouped['Normalized_Fluorescence_Intensity_COMPOUND', channel].unstack().fillna(0)
        ax = sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f", linewidths=1)
        # Get the colorbar object and set its tick label font size
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        plt.title(f'Mean Fluorescence {"Paxillin" if channel == 0 else "Integrin" if channel == 1 else "Other"} Across Different Substrate Conditions', fontsize=TITLE_FONT_SIZE)
        plt.ylabel('Fluorescence Intensity', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('Pitch', fontsize=LABEL_FONT_SIZE)
        
        # Adjust tick label font sizes
        plt.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
        plt.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
        plt.savefig(f'figure_compound_{channel}.pdf')
        # plt.show()

        # Plot heatmap for Soma fluorescence
        plt.figure(figsize=(4, 2))
        heatmap_data = grouped['Normalized_Fluorescence_Intensity_SOMA', channel].unstack().fillna(0)
        ax = sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f", linewidths=1)
        # Get the colorbar object and set its tick label font size
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        plt.title(f'Mean Fluorescence Soma {"Paxillin" if channel == 0 else "Integrin" if channel == 1 else "Other"} Across Different Substrate Conditions', fontsize=TITLE_FONT_SIZE)
        plt.ylabel('Fluorescence Intensity', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('Pitch', fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
        plt.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
        plt.savefig(f'figure_soma_{channel}.pdf')
        # plt.show()

        # Plot heatmap for Neurite fluorescence
        plt.figure(figsize=(4, 2))
        heatmap_data = grouped['Normalized_Fluorescence_Intensity_NEURITE', channel].unstack().fillna(0)
        ax = sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f", linewidths=1)
        # Get the colorbar object and set its tick label font size
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        plt.title(f'Mean Fluorescence Neurites {"Paxillin" if channel == 0 else "Integrin" if channel == 1 else "Other"} Across Different Substrate Conditions', fontsize=TITLE_FONT_SIZE)
        plt.ylabel('Fluorescence Intensity', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('Pitch', fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
        plt.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)

        plt.savefig(f'figure_neurite_{channel}.pdf')
        # plt.show()

        print(f"Heatmaps saved for channel {channel} (Compound, Soma, Neurites)")

# Apply normalization by pillar density (skip flat by using 1.0 as dummy divisor)
def normalize_by_density(row, col_name):
    density = pillar_densities.get(row['P_Group'], 1.0)
    return row[col_name] / density


# Load the data
results_df = pd.read_csv('soma_neurite_separate_data_first_clean_changed.csv')

results_df.rename(columns={
    'Normalized Fluorescence Intensity COMPOUND': 'Normalized_Fluorescence_Intensity_COMPOUND',
    'Normalized Fluorescence Intensity SOMA': 'Normalized_Fluorescence_Intensity_SOMA',
    'Normalized Fluorescence Intensity NEURITE': 'Normalized_Fluorescence_Intensity_NEURITE'
}, inplace=True)

# Correct the column creation for 'P Group' and 'Image Group'
# Make sure to create them with underscores from the start
results_df['P_Group'] = results_df['Key Image Name'].apply(categorize_p_value) # Changed to P_Group
results_df['Image_Group'] = results_df['Key Image Name'].apply(lambda name: name[0]) # Changed to Image_Group
results_df['Image_Group'] = results_df['Image_Group'].apply(categorize_image_group) # Changed to Image_Group

# Filter out rows where Image Group is None
results_df = results_df[results_df['Image_Group'].notna()] # Changed to Image_Group

# Define category order for P Group and set as categorical
p_group_order = ['p4', 'p10', 'p30', 'flat']
results_df['P_Group'] = pd.Categorical(results_df['P_Group'], categories=p_group_order, ordered=True) # Changed to P_Group

# Define pillar densities
pillar_densities = {
    'p4': 0.28,
    'p10': 0.11,
    'p30': 0.045,
    'flat': 1.0  # Use 1.0 so division doesn’t change flat values
}

# Extract flat data and replicate it for each image group
# When you replicate flat data, ensure 'Image_Group' is used
flat_data = results_df[results_df['P_Group'] == 'flat'].copy() # Changed to P_Group
for group in ['Thin', 'Mushroom', 'Stubby']:
    replicated_flat = flat_data.copy()
    replicated_flat['Image_Group'] = group # Changed to Image_Group
    results_df = pd.concat([results_df, replicated_flat])

# Group and calculate mean fluorescence intensity for each region (Compound, Soma, Neurites) and channel
# Normalize fluorescence intensity columns
for region in ['Normalized_Fluorescence_Intensity_COMPOUND',
               'Normalized_Fluorescence_Intensity_SOMA',
               'Normalized_Fluorescence_Intensity_NEURITE']:
    results_df[region] = results_df.apply(lambda row: normalize_by_density(row, region), axis=1)

# method for generating heatmaps    
heatmap_generated(results_df)

# Analyze for each region and channel
regions = ['COMPOUND', 'SOMA', 'NEURITE']
channels = [0, 1, 2]  # Assuming these represent some channels

output_filename = 'statistical_analysis_report_3_og.txt'

# Clear the file before starting (optional, depends on desired behavior)
with open(output_filename, 'w') as f:
    f.write("Statistical Analysis Report\n")
    f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.write("=" * 80 + "\n")

for region in regions:
    for channel in channels:
        conduct_channel_analysis(results_df, region, channel, output_filename)

print(f"Statistical analysis report generated successfully: {output_filename}")


# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import re 
# import os
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from scipy import stats
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from statsmodels.stats.diagnostic import het_breuschpagan # For homoscedasticity test

# # --- Helper functions (remain mostly the same) ---
# def categorize_p_value(name):
#     match = re.search(r'p(\d+)', name.lower())
#     if match:
#         return f'p{match.group(1)}'
#     return 'flat'

# def categorize_image_group(name): # Changed 'letter' to 'name' as we'll check the full string
#     name_lower = name.lower()
#     if 'thin' in name_lower:
#         return 'Thin'
#     elif 'mushroom' in name_lower:
#         return 'Mushroom'
#     elif 'stubby' in name_lower:
#         return 'Stubby'
#     elif 'flat' in name_lower: # Look for 'flat' anywhere in the name
#         return 'Flat'
#     return None # Return None if no category is found

# def normalize_by_density(row, col_name, pillar_densities):
#     density = pillar_densities.get(row['P Group'], 1.0)
#     return row[col_name] / density

# # --- New/Refined Function for Advanced Statistical Analysis ---
# def perform_advanced_statistical_analysis(df_input, region, channel, channel_name, plot_dir="plots"):
#     """
#     Performs statistical analysis including model fitting, assumption checks,
#     and post-hoc tests for a given region and channel.
#     Generates diagnostic plots for residuals.
#     """
#     os.makedirs(plot_dir, exist_ok=True)
    
#     df_channel = df_input[df_input['Channel'] == channel].copy() # Ensure working on a copy

#     print(f"\n--- Statistical Analysis for {region} ({channel_name}, Channel {channel}) ---")

#     response_var = f'Normalized_Fluorescence_Intensity_{region}'

#     # --- 1. Initial Data Distribution Checks ---
#     print("\n--- Initial Data Distribution Checks ---")
    
#     # Check if there's enough data for analysis
#     if df_channel.empty or len(df_channel[response_var].dropna()) < 3:
#         print(f"Not enough data for {region} ({channel_name}). Skipping analysis.")
#         return

#     # Descriptive statistics
#     print("\nDescriptive Statistics:")
#     print(df_channel[response_var].describe())

#     # Shapiro-Wilk Test for Normality (on original data)
#     stat, p = stats.shapiro(df_channel[response_var].dropna())
#     print("\nShapiro-Wilk Test for Normality (on raw data):")
#     print(f"Statistic = {stat:.4f}, p-value = {p:.4f}")
#     if p > 0.05:
#         print("The raw data appears to be normally distributed.")
#     else:
#         print("The raw data does NOT appear to be normally distributed. Consider transformations or non-parametric tests.")

#     # Visual distribution plots
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 3, 1)
#     sns.histplot(df_channel[response_var], kde=True)
#     plt.title(f'Histogram of {region} {channel_name}')
    
#     plt.subplot(1, 3, 2)
#     sns.boxplot(x='Image_Group', y=response_var, data=df_channel)
#     plt.title(f'{region} {channel_name} by Image Group')
#     plt.xticks(rotation=45, ha='right')

#     plt.subplot(1, 3, 3)
#     sns.boxplot(x='P_Group', y=response_var, data=df_channel)
#     plt.title(f'{region} {channel_name} by P Group')
#     plt.xticks(rotation=45, ha='right')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, f'distribution_plots_{region}_{channel_name}.pdf'))
#     plt.close()

#     # Levene's test for equal variance across groups (on original data)
#     try:
#         levene_groups_image = [df_channel[df_channel['Image_Group'] == group][response_var].dropna() 
#                                for group in df_channel['Image_Group'].unique()]
#         levene_groups_image = [arr for arr in levene_groups_image if len(arr) > 1] # Filter out groups with <2 data points
#         if len(levene_groups_image) > 1:
#             stat_levene_image, p_levene_image = stats.levene(*levene_groups_image)
#             print(f"\nLevene’s Test for Image Group (equal variances): Statistic = {stat_levene_image:.4f}, p-value = {p_levene_image:.4f}")
#             if p_levene_image < 0.05:
#                 print("Variances are NOT equal across Image Groups. This violates an ANOVA assumption.")
#         else:
#             print("\nNot enough groups with sufficient data for Levene's Test for Image Group.")

#         levene_groups_p = [df_channel[df_channel['P_Group'] == group][response_var].dropna() 
#                            for group in df_channel['P_Group'].unique()]
#         levene_groups_p = [arr for arr in levene_groups_p if len(arr) > 1] # Filter out groups with <2 data points
#         if len(levene_groups_p) > 1:
#             stat_levene_p, p_levene_p = stats.levene(*levene_groups_p)
#             print(f"Levene’s Test for P Group (equal variances): Statistic = {stat_levene_p:.4f}, p-value = {p_levene_p:.4f}")
#             if p_levene_p < 0.05:
#                 print("Variances are NOT equal across P Groups. This violates an ANOVA assumption.")
#         else:
#             print("Not enough groups with sufficient data for Levene's Test for P Group.")

#     except ValueError as e:
#         print(f"\nCould not perform Levene's Test due to data issues: {e}. Ensure groups have at least 2 data points.")


#     # --- 2. OLS Model Fitting ---
#     print("\n--- OLS Model Fitting and Residual Diagnostics ---")
#     formula = f'{response_var} ~ C(Image_Group, Treatment(reference="Flat")) + C(P_Group, Treatment(reference="flat"))'
#     try:
#         model = ols(formula, data=df_channel).fit()
#         print("\nOLS Model Summary:")
#         print(model.summary())

#         # --- 3. Residual Diagnostics ---
#         print("\n--- Residual Checks ---")
        
#         # Plot Residuals vs. Fitted values
#         plt.figure(figsize=(10, 5))
#         sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
#                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#         plt.xlabel('Fitted Values')
#         plt.ylabel('Residuals')
#         plt.title(f'Residuals vs. Fitted Values for {region} {channel_name}')
#         plt.axhline(0, color='gray', linestyle='--', lw=0.5)
#         plt.savefig(os.path.join(plot_dir, f'residuals_vs_fitted_{region}_{channel_name}.pdf'))
#         plt.close()

#         # Normal Q-Q plot of residuals
#         plt.figure(figsize=(6, 6))
#         sm.qqplot(model.resid, line='s')
#         plt.title(f'Normal Q-Q Plot of Residuals for {region} {channel_name}')
#         plt.savefig(os.path.join(plot_dir, f'qq_plot_{region}_{channel_name}.pdf'))
#         plt.close()

#         # Shapiro-Wilk Test for Normality (on residuals)
#         stat_resid, p_resid = stats.shapiro(model.resid)
#         print("\nShapiro-Wilk Test for Normality (on residuals):")
#         print(f"Statistic = {stat_resid:.4f}, p-value = {p_resid:.4f}")
#         if p_resid > 0.05:
#             print("Residuals appear to be normally distributed.")
#         else:
#             print("Residuals do NOT appear to be normally distributed. This violates an ANOVA assumption.")

#         # Breusch-Pagan Test for Homoscedasticity
#         # Need to ensure there's enough variation in exogenous variables for test
#         if model.model.exog.shape[1] > 1: # Check if there's more than just an intercept
#             try:
#                 lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
#                 print("\nBreusch-Pagan Test for Homoscedasticity:")
#                 print(f"LM Statistic = {lm:.4f}, p-value = {lm_pvalue:.4f}")
#                 if lm_pvalue < 0.05:
#                     print("Residuals exhibit heteroscedasticity (non-constant variance). This violates an ANOVA assumption.")
#                 else:
#                     print("Residuals appear homoscedastic (constant variance).")
#             except Exception as e:
#                 print(f"Could not perform Breusch-Pagan Test: {e}. Check data or model specification.")
#         else:
#             print("Breusch-Pagan Test skipped: Only an intercept in the model's exogenous variables.")

#         # --- 4. Post-hoc Analysis (Tukey's HSD) ---
#         print("\n--- Post-hoc Tukey's HSD ---")
        
#         # Tukey HSD for Image Group
#         if len(df_channel['Image_Group'].unique()) > 1:
#             try:
#                 tukey_image_group = pairwise_tukeyhsd(endog=df_channel[response_var], 
#                                                       groups=df_channel['Image_Group'], 
#                                                       alpha=0.05)
#                 print("\nTukey HSD for Image Group:")
#                 print(tukey_image_group)
#             except ValueError as e:
#                 print(f"Could not perform Tukey HSD for Image Group: {e}. Check group sizes or data.")
#         else:
#             print("Tukey HSD for Image Group skipped: Only one unique Image Group found.")
            
#         # Tukey HSD for P Group
#         if len(df_channel['P_Group'].unique()) > 1:
#             try:
#                 tukey_p_group = pairwise_tukeyhsd(endog=df_channel[response_var], 
#                                                   groups=df_channel['P_Group'], 
#                                                   alpha=0.05)
#                 print("\nTukey HSD for P Group:")
#                 print(tukey_p_group)
#             except ValueError as e:
#                 print(f"Could not perform Tukey HSD for P Group: {e}. Check group sizes or data.")
#         else:
#             print("Tukey HSD for P Group skipped: Only one unique P Group found.")


#     except ValueError as e:
#         print(f"Error fitting OLS model or performing subsequent tests: {e}. Check data for NaNs or insufficient variation.")
#     except Exception as e:
#         print(f"An unexpected error occurred during OLS analysis: {e}")

#     # --- 5. Non-parametric alternative (Kruskal-Wallis) ---
#     print("\n--- Non-parametric Test (Kruskal-Wallis) ---")
#     try:
#         kruskal_image_group_stat, kruskal_image_group_p = stats.kruskal(
#             *[df_channel[df_channel['Image_Group'] == group][response_var].dropna() 
#               for group in df_channel['Image_Group'].unique() if not df_channel[df_channel['Image_Group'] == group][response_var].dropna().empty]
#         )
#         print(f"Kruskal-Wallis Test for Image Group: Statistic = {kruskal_image_group_stat:.4f}, p-value = {kruskal_image_group_p:.4f}")
        
#         kruskal_p_group_stat, kruskal_p_group_p = stats.kruskal(
#             *[df_channel[df_channel['P_Group'] == group][response_var].dropna() 
#               for group in df_channel['P_Group'].unique() if not df_channel[df_channel['P_Group'] == group][response_var].dropna().empty]
#         )
#         print(f"Kruskal-Wallis Test for P Group: Statistic = {kruskal_p_group_stat:.4f}, p-value = {kruskal_p_group_p:.4f}")
#     except ValueError as e:
#         print(f"Could not perform Kruskal-Wallis Test: {e}. Ensure groups have at least 2 data points.")


# # --- Heatmap Generation (same as before, good for initial visualization) ---
# def heatmap_generation(results_df_processed, output_dir):
#     grouped = results_df_processed.groupby(['Image_Group', 'P_Group', 'Channel']).agg({
#         'Normalized_Fluorescence_Intensity_COMPOUND': 'mean',
#         'Normalized_Fluorescence_Intensity_SOMA': 'mean',
#         'Normalized_Fluorescence_Intensity_NEURITE': 'mean'
#     }).unstack('Channel')

#     paxillin_palette = sns.color_palette("Reds")
#     integrin_palette = sns.color_palette("Blues")
#     other_palette = sns.color_palette("Greens") # Assuming channel 2 is 'Other'
    
#     TITLE_FONT_SIZE = 8
#     LABEL_FONT_SIZE = 4
#     TICK_LABEL_FONT_SIZE = 4
#     COLORBAR_TICK_FONT_SIZE = 4

#     # Loop through channels and create heatmaps
#     channel_map = {0: 'Paxillin', 1: 'Integrin', 2: 'Other'} 
#     palette_map = {0: paxillin_palette, 1: integrin_palette, 2: other_palette}

#     for channel_idx in [0, 1, 2]: 
#         channel_name = channel_map[channel_idx]
#         palette = palette_map[channel_idx]
        
#         for intensity_type, title_prefix in [
#             ('Normalized_Fluorescence_Intensity_COMPOUND', 'Mean Fluorescence'),
#             ('Normalized_Fluorescence_Intensity_SOMA', 'Mean Fluorescence Soma'),
#             ('Normalized_Fluorescence_Intensity_NEURITE', 'Mean Fluorescence Neurites')
#         ]:
#             plt.figure(figsize=(4, 2))
#             heatmap_data = grouped[intensity_type, channel_idx].unstack().fillna(0)
#             ax = sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f", linewidths=1)
#             cb = ax.collections[0].colorbar
#             cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
            
#             plt.title(f'{title_prefix} {channel_name} Across Different Substrate Conditions', fontsize=TITLE_FONT_SIZE)
#             plt.ylabel('Fluorescence Intensity', fontsize=LABEL_FONT_SIZE)
#             plt.xlabel('Pitch', fontsize=LABEL_FONT_SIZE)
#             plt.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
#             plt.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
#             plt.tight_layout()
#             plt.savefig(f'{output_dir}/heatmap_{intensity_type.split("_")[-1].lower()}_{channel_name.lower()}.pdf')
#             plt.close()
#             print(f"Heatmap saved: {intensity_type.split('_')[-1].lower()}_{channel_name.lower()}.pdf")


# # --- Main script execution (MODIFIED) ---
# if __name__ == "__main__":

#     output_dir = 'plots_third_clean'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Load the data
#     results_df = pd.read_csv('soma_neurite_separate_data_third_clean.csv')
    
#     results_df.rename(columns={
#         'Normalized Fluorescence Intensity COMPOUND': 'Normalized_Fluorescence_Intensity_COMPOUND',
#         'Normalized Fluorescence Intensity SOMA': 'Normalized_Fluorescence_Intensity_SOMA',
#         'Normalized Fluorescence Intensity NEURITE': 'Normalized_Fluorescence_Intensity_NEURITE'
#     }, inplace=True)
    
#     # Apply functions
#     results_df['P Group'] = results_df['Key Image Name'].apply(categorize_p_value)
#     results_df['Image Group'] = results_df['Key Image Name'].apply(categorize_image_group)

#     # Filter out rows where Image Group is None (invalid categorization)
#     results_df = results_df[results_df['Image Group'].notna()]
    
#     # Define category order for P Group and set as categorical
#     p_group_order = ['p4', 'p10', 'p30', 'flat']
#     results_df['P Group'] = pd.Categorical(results_df['P Group'], categories=p_group_order, ordered=True)

#     # Define pillar densities
#     pillar_densities = {
#         'p4': 0.28,
#         'p10': 0.11,
#         'p30': 0.045,
#         'flat': 1.0
#     }

#     results_df_processed = results_df.copy() # Start with the correctly categorized dataframe

#     # Normalize fluorescence intensity columns now that P Group is set
#     for region_col_name in ['Normalized_Fluorescence_Intensity_COMPOUND',
#                             'Normalized_Fluorescence_Intensity_SOMA',
#                             'Normalized_Fluorescence_Intensity_NEURITE']:
#         results_df_processed[region_col_name] = results_df_processed.apply(
#             lambda row: normalize_by_density(row, region_col_name, pillar_densities), axis=1
#         )
        
#     # Ensure categorical types are correctly set on the final processed DataFrame
#     # Note the column names Image_Group and P_Group (with underscore) that the OLS formula expects.
#     image_group_order = ['Flat', 'Thin', 'Mushroom', 'Stubby'] 
#     results_df_processed['Image_Group'] = pd.Categorical(
#         results_df_processed['Image Group'], # Use the original 'Image Group' which has 'Flat'
#         categories=image_group_order, ordered=True
#     )
#     # P Group is already categorical and ordered, just rename for consistency
#     results_df_processed['P_Group'] = results_df_processed['P Group'] 

#     # Drop original 'Image Group' and 'P Group' as they are now mapped to _ versions
#     results_df_processed.drop(columns=['Image Group', 'P Group'], inplace=True)

#     print("results_df_processed (Image_Group and P_Group categories):")
#     print(results_df_processed[['Key Image Name', 'Image_Group', 'P_Group']].head()) # Debug print
#     print("Unique Image_Group categories:", results_df_processed['Image_Group'].cat.categories)
#     print("Unique P_Group categories:", results_df_processed['P_Group'].cat.categories)
#     print("Count of 'Flat' in Image_Group:", (results_df_processed['Image_Group'] == 'Flat').sum())
#     print("Count of 'flat' in P_Group:", (results_df_processed['P_Group'] == 'flat').sum())

#     # --- Heatmap Generation (same as before, good for initial visualization) ---
#     paxillin_palette = sns.color_palette("Reds")
#     integrin_palette = sns.color_palette("Blues")
#     other_palette = sns.color_palette("Greens") 
    
#     TITLE_FONT_SIZE = 8
#     LABEL_FONT_SIZE = 4
#     TICK_LABEL_FONT_SIZE = 4
#     COLORBAR_TICK_FONT_SIZE = 4

#     channel_map = {0: 'Paxillin', 1: 'Integrin', 2: 'Other'} 
#     palette_map = {0: paxillin_palette, 1: integrin_palette, 2: other_palette}

#     for channel_idx in [0, 1, 2]: 
#         channel_name = channel_map[channel_idx]
#         palette = palette_map[channel_idx]
        
#         for intensity_type, title_prefix in [
#             ('Normalized_Fluorescence_Intensity_COMPOUND', 'Mean Fluorescence'),
#             ('Normalized_Fluorescence_Intensity_SOMA', 'Mean Fluorescence Soma'),
#             ('Normalized_Fluorescence_Intensity_NEURITE', 'Mean Fluorescence Neurites')
#         ]:
#             plt.figure(figsize=(4, 2))
            
#             # Filter for the specific channel
#             df_for_heatmap = results_df_processed[results_df_processed['Channel'] == channel_idx]
            
#             # Check if there's any data for the current channel, intensity type, and groups
#             if df_for_heatmap.empty or \
#                df_for_heatmap['Image_Group'].nunique() < 2 or \
#                df_for_heatmap['P_Group'].nunique() < 2:
#                 print(f"Not enough diverse data for heatmap for {intensity_type}, Channel {channel_idx}. Skipping heatmap.")
#                 plt.close()
#                 continue # Skip to next iteration

#             heatmap_data = df_for_heatmap.pivot_table(
#                 index='Image_Group',
#                 columns='P_Group',
#                 values=intensity_type,
#                 aggfunc='mean'
#             ).loc[results_df_processed['Image_Group'].cat.categories, results_df_processed['P_Group'].cat.categories].fillna(0)
            
#             ax = sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f", linewidths=1)
#             cb = ax.collections[0].colorbar
#             cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
            
#             plt.title(f'{title_prefix} {channel_name} Across Different Substrate Conditions', fontsize=TITLE_FONT_SIZE)
#             plt.ylabel('Fluorescence Intensity', fontsize=LABEL_FONT_SIZE)
#             plt.xlabel('Pitch', fontsize=LABEL_FONT_SIZE)
#             plt.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
#             plt.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
#             plt.tight_layout()
#             plt.savefig(f'{output_dir}/heatmap_{intensity_type.split("_")[-1].lower()}_{channel_name.lower()}.pdf')
#             plt.close()
#             print(f'Heatmap saved: {intensity_type.split("_")[-1].lower()}_{channel_name.lower()}.pdf')

#     # --- Advanced Statistical Analysis for Integrin and Paxillin ---
#     regions_to_analyze = ['COMPOUND', 'SOMA', 'NEURITE']
#     channels_to_analyze = {0: 'Paxillin', 1: 'Integrin', 2: 'Other'} # Include all channels now

#     for region in regions_to_analyze:
#         for channel, channel_name in channels_to_analyze.items():
#             perform_advanced_statistical_analysis(results_df_processed, region, channel, channel_name, output_dir)