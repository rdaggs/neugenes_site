import pandas as pd
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.io.export import export_png
import os
from model.config import *
def plot_expression_levels_scaled(filepath1, structures,names,title,output_filepath):
    print("inside here")
    # Load the datasets
    data1 = pd.read_csv(filepath1)

    # Sum over the specified structures
    region_sums1 = data1[structures].sum()

    # Scale intensities to a range of 0-100
    region_sums1_scaled = (region_sums1 / region_sums1.max()) * 100

    # Create DataFrame for compatibility and sorting
    df1 = pd.DataFrame({'region': region_sums1.index, 'intensity': region_sums1_scaled.values})

    # Sort by intensity in descending order for the first dataset
    df1 = df1.sort_values(by='intensity', ascending=False)

    # Create ColumnDataSource
    source1 = ColumnDataSource(df1)

    # Configure the figure
    p = figure(
        width=900,
        height=500,
        x_range=df1['region'].tolist(),  # Set x-axis to sorted regions
        title=title,
        toolbar_location="above",
        
    )

    # Add bars for Dataset 1
    p.vbar(
        x='region', 
        top='intensity', 
        width=0.4, 
        source=source1, 
        color=Spectral4[0], 
        legend_label= names[0]
    )

    # Configure legend and style
    p.legend.location = "top_right"
    p.legend.click_policy = "mute"  # Click on legend to mute datasets
    p.xaxis.major_label_orientation = "vertical"
    p.xaxis.axis_label = f"Brain Region (Sorted by {names[0]})"
    p.yaxis.axis_label = "Expression Intensity (Scaled to 0-100)"
    print("before export")
    export_png(p, filename=output_filepath)

# plot_expression_levels_scaled(
#     filepath1=os.path.join(f'result_norm.csv'),
#     structures=['RH'],
#     names = ['control_1','stress_5'],
#     title = 'title',
#     output_filepath = 
# )
