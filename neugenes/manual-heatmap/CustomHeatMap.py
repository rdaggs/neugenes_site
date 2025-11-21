import brainglobe_heatmap as bgh
import numpy as np
from brainglobe_heatmap.heatmaps import find_annotation_position_inside_polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create a custom colormap with proper NaN handling
def create_transparent_colormap(base_cmap="RdBu"):
    base = plt.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, base.N))
    
    transparent_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'transparent_{base_cmap}', color_list)
    
    # Set NaN values to be transparent
    transparent_cmap.set_bad(alpha=0)
    
    return transparent_cmap

def create_mapped_nan_dict(raw_dict):

    # Initialize a dummy Heatmap to access the atlas
    temp_hm = bgh.Heatmap(
        {"AM": 1.0},  # Use non-zero value for initialization
        position=0,
        orientation="frontal",
        thickness=10,
        format="2D",
        check_latest=False
    )
    
    # Get all brain region acronyms
    all_structures = temp_hm.scene.atlas.lookup_df.acronym.values

    # Create dictionary with all regions as NaN
    data_dict = {structure: np.nan for structure in all_structures}

    # Convert 0 values to NaN for purpose of color mapping
    processed_dict = {k: (np.nan if v == 0 else v) for k, v in raw_dict.items()}

    data_dict.update(processed_dict)
    return data_dict

def create_mapped_zero_dict(raw_dict):

    # Initialize a dummy Heatmap to access the atlas
    temp_hm = bgh.Heatmap(
        {"AM": 1.0},  # Use non-zero value for initialization
        position=0,
        orientation="frontal",
        thickness=10,
        format="2D",
        check_latest=False
    )
    # Get all brain region acronyms
    all_structures = temp_hm.scene.atlas.lookup_df.acronym.values

    # Create dictionary with all regions as NaN
    data_dict = {structure: 0.0 for structure in all_structures}


    data_dict.update(raw_dict)
    return data_dict

class CustomHeatMap(bgh.Heatmap):
    def plot_subplot(self, fig, ax, show_legend=False, xlabel="µm", ylabel="µm", 
                  hide_axes=False, cbar_label=None, show_cbar=True, **kwargs):
        """
        Enhanced version of plot_subplot that properly handles NaN values and 
        draws outlines for all brain regions.
        """
        # Get the projected coordinates for all structures
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )

        segments = []
        for r, coords in projected.items():
            name, segment_nr = r.split("_segment_")
            x = coords[:, 0]
            y = coords[:, 1]
            # calculate area of polygon with Shoelace formula
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

            segments.append({
                "name": name,
                "segment_nr": segment_nr,
                "coords": coords,
                "area": area,
            })

        # Sort region segments by area (largest first)
        segments.sort(key=lambda s: s["area"], reverse=True)

        # Draw filled regions first
        for segment in segments:
            name = segment["name"]
            segment_nr = segment["segment_nr"]
            coords = segment["coords"]
            
            # Determine color based on whether the region has a non-NaN value
            if name in self.values:
                if np.isnan(self.values[name]):
                    # Treat NaN as 0.0 for color mapping
                    value = 0.0
                    is_nan = True
                else:
                    value = self.values[name]
                    is_nan = False

                # Use colormap to compute color
                color = self.cmap((value - self.vmin) / (self.vmax - self.vmin))

                # Adjust alpha and zorder based on region
                if name == "root":
                    alpha = 0.3
                    zorder = -1
                elif is_nan:
                    alpha = 1.0
                    zorder = -2  # Push NaN-as-zero below others
                else:
                    alpha = 1.0
                    zorder = None
            else:
                # Region not in values dict at all - make transparent
                color = [0, 0, 0, 0]
                alpha = 0
                zorder = -2
            
            # Fill the region
            ax.fill(
                coords[:, 0],
                coords[:, 1],
                color=color,
                label=name if segment_nr == "0" and name != "root" else None,
                lw=0,  # No border here, we'll add them separately
                ec=None,
                zorder=zorder,
                alpha=alpha,
            )

        # Now draw all outlines separately
        for segment in segments:
            name = segment["name"]
            coords = segment["coords"]
            
            # Draw outline for all regions
            linewidth = 0.5
            if name == "root":
                linecolor = 'gray'
                alpha = 0.3
            elif name in self.values:
                if np.isnan(self.values[name]):
                    # NaN values get light gray outline
                    linecolor = 'gray'
                    alpha = 0.5
                else:
                    # Valid values get dark outline
                    linecolor = 'black'
                    alpha = 1.0
                    linewidth = 1.0
            else:
                # Regions not in values dict get light gray outline
                linecolor = 'gray'
                alpha = 0.5
            
            ax.plot(
                np.append(coords[:, 0], coords[0, 0]),  # Close the polygon
                np.append(coords[:, 1], coords[0, 1]),
                color=linecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=10  # Make sure outlines are on top
            )

            display_text = self.get_region_annotation_text(name)
            if display_text is not None:
                annotation_pos = find_annotation_position_inside_polygon(coords)
                if annotation_pos is not None:
                    ax.annotate(
                        display_text,
                        xy=annotation_pos,
                        ha="center",
                        va="center",
                        **(self.annotate_text_options_2d if self.annotate_text_options_2d is not None else {})
                    )

        if show_cbar:
            # make colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib as mpl
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            if self.label_regions is True:
                # Filter out NaN values for the colorbar
                valid_values = {k: v for k, v in self.values.items() 
                               if not np.isnan(v)}
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.cm.get_cmap(self.cmap, len(valid_values)),
                    ),
                    cax=cax,
                )
                cbar.ax.set_yticklabels([r.strip() for r in valid_values.keys()])
            else:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
                )

            if cbar_label is not None:
                cbar.set_label(cbar_label)

        # style axes
        ax.invert_yaxis()
        ax.axis("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # ax.set(title=self.title)

        if isinstance(self.orientation, str) or np.sum(self.orientation) == 1:
            # orthogonal projection
            ax.set(xlabel=xlabel, ylabel=ylabel)

        if hide_axes:
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(xlabel="", ylabel="")

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            # Filter out regions with NaN values from the legend
            if handles:
                valid_indices = [i for i, label in enumerate(labels) 
                                if label in self.values and not np.isnan(self.values[label])]
                ax.legend([handles[i] for i in valid_indices], 
                          [labels[i] for i in valid_indices])

        return fig, ax