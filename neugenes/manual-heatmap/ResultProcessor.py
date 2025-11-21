import pandas as pd
import os
def read_csv_and_generate_dict(file_path):
    """
    Reads a CSV file and generates a dictionary with region names as keys
    and their corresponding values.


    Parameters
    ----------
    file_path : str
        Path to the CSV file.


    Returns
    -------
    dict
        Dictionary with region names as keys and their corresponding values.
    """
    data_dict = {}
    df = pd.read_csv(file_path)
    structures = df.columns[1:].to_list()


    # Sum over the specified structures
    region_sums1 = df[structures].sum()


    # Scale intensities to a range of 0-100
    region_sums1_scaled = (region_sums1 / region_sums1.max()) * 100


    data_dict = {structure: region_sums1_scaled[structure] for structure in structures if structure != "AHA"}
    return data_dict


# from brainrender.atlas import Atlas
# atlas = Atlas("allen_mouse_25um")
# print(atlas.lookup_df[["acronym", "name"]].to_string())


def read_all_csv_and_generate_dict(file_path_list):
    """
    Reads multiple CSV files and organizes structure data into stress and control groups.
    Computes raw sums per structure per mouse, group averages, and global min/max values.


    Parameters
    ----------
    file_path_list : list of str
        List of paths to the CSV files.


    Returns
    -------
    dict
        Dictionary structured as:
        {
            structure_name: {
                'control': [...],
                'stress': [...],
                'control_avg': float,
                'stress_avg': float
            },
            ...
            'min_value': float,
            'max_value': float
        }
    """
    structure_data = {}


    all_values = []


    for file_path in file_path_list:
        df = pd.read_csv(file_path)
        structures = df.columns[1:]  # Skip first column (e.g., region name)

        group = None
        filename = os.path.basename(file_path).lower()


        if filename.endswith("_control1.csv") or filename.endswith("_control2.csv"):
            group = "control"
        elif any(filename.endswith(f"_stress{i}.csv") for i in range(1, 6)):
            group = "stress"
        else:
            print("Bad file name")
            continue  # Skip unrecognized files


        for structure in structures:
            if structure == "AHA":
                continue


            structure_sum = df[structure].sum()


            if structure not in structure_data:
                structure_data[structure] = {"control": [], "stress": []}


            structure_data[structure][group].append(structure_sum)
            all_values.append(structure_sum)


    # Compute averages and add them to the dict
    for structure, groups in structure_data.items():
        control_vals = groups["control"]
        stress_vals = groups["stress"]
        groups["control_avg"] = sum(control_vals) / len(control_vals) if control_vals else 0
        groups["stress_avg"] = sum(stress_vals) / len(stress_vals) if stress_vals else 0


    # Compute min and max over all raw values
    structure_data["min_value"] = min(all_values)
    structure_data["max_value"] = max(all_values)


    return structure_data



def read_single_csv_and_generate_dict(file_path):
    """
    Reads a single CSV file and returns a tuple containing a dictionary of structure sums
    and a tuple of min/max values.


    Parameters
    ----------
    file_path : str
        Path to the CSV file.


    Returns
    -------
    tuple
        (structure_sums_dict, (min_value, max_value))
        where structure_sums_dict has structure names as keys and their corresponding sums as values
    """
    structure_data = {}
    all_values = []


    df = pd.read_csv(file_path)
    structures = df.columns[1:]  # Skip first column (e.g., region name)


    for structure in structures:
        if structure == "AHA":
            continue


        structure_sum = df[structure].sum()
        structure_data[structure] = structure_sum
        all_values.append(structure_sum)


    min_value = min(all_values)
    max_value = max(all_values)


    return structure_data, (min_value, max_value)

