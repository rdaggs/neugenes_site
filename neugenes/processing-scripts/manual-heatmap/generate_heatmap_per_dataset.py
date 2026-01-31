import matplotlib.pyplot as plt
import CustomHeatMap as chm
import ResultProcessor as rp
import argparse
import os


def plot_group_heatmaps(data_dict, min_val, max_val, title_prefix, save_path, cmap):
    fig, axs = plt.subplots(4, 4, figsize=(18, 12))
    positions = range(3000, 9000, 500)
    scenes = []


    for distance in positions:
        scene = chm.CustomHeatMap(
            data_dict,
            position=distance,
            orientation="frontal",
            thickness=10,
            format="2D",
            check_latest = False,
            cmap=cmap,
            vmin=min_val,
            vmax=max_val,
            label_regions=False,
            annotate_regions=False,
        )
        scenes.append(scene)


    for scene, ax, pos in zip(scenes, axs.flatten(), positions, strict=False):
        scene.plot_subplot(fig=fig, ax=ax, show_cbar=True, hide_axes=False)
        print(f"{title_prefix}: finished processing slice at {pos} µm")


    # Add title to the figure
    # fig.suptitle(f"{title_prefix} Heatmap", fontsize=16, y=0.95)
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate heatmap from CSV data')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', type=str, help='Output path for the heatmap image (default: based on CSV filename)')
    parser.add_argument('--title', '-t', type=str, help='Title prefix for the heatmap (default: based on CSV filename)')
   
    args = parser.parse_args()
   
    # Validate input file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: Input file '{args.csv_path}' does not exist")
        return
   
    # Set default output path and title if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        args.output = f"{base_name}.png"
   
    if args.title is None:
        args.title = os.path.splitext(os.path.basename(args.csv_path))[0]
   
    # Create colormap
    cmap = chm.create_transparent_colormap("PuRd")


    # Read CSV file
    structure_data, (min_val, max_val) = rp.read_single_csv_and_generate_dict(args.csv_path)
    print("Structure data loaded successfully")


    data_dict = chm.create_mapped_nan_dict(structure_data)


    # Plot heatmap
    plot_group_heatmaps(data_dict, min_val, max_val, args.title, args.output, cmap)
    print(f"Heatmap saved to: {args.output}")


if __name__ == "__main__":
    main()