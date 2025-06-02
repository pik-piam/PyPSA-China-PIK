import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chardet
import logging
import sys
import yaml
import re
import ultraplot as uplt
from typing import Dict, List, Tuple
import pypsa
from _plot_utilities import validate_hex_colors
from _helpers import mock_snakemake
# --------------------------------------------------
# Constants
# --------------------------------------------------

# Currency exchange rates (as of 2024)
EXCHANGE_RATES: Dict[Tuple[str, str], float] = {
    ("eur", "eur"): 1.0,
    ("eur", "cny"): 7.8,
    ("cny", "eur"): 1/7.8,
    ("usd", "cny"): 7.2,
    ("cny", "usd"): 1/7.2,
    ("usd", "eur"): 7.2/7.8,
    ("eur", "usd"): 7.8/7.2,
    ("cny", "cny"): 1.0,
    ("usd", "usd"): 1.0,
}

# Currency aliases for standardization
CURRENCY_ALIASES: Dict[str, str] = {
    "eur": "eur",
    "€": "eur",
    "usd": "usd",
    "$": "usd",
    "cny": "cny",
    "rmb": "cny"
}

# --------------------------------------------------
# 1) Helper Functions
# --------------------------------------------------


def detect_file_encoding(file_path: str) -> str:
    """Detect the encoding of a file.

    Args:
        file_path: Path to the file whose encoding needs to be detected.

    Returns:
        str: The detected encoding of the file.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    return encoding


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load and clean data from a CSV file.

    load a CSV file, detects its encoding, replaces '-' with NaN,
    and drops the 'link' column if it exists.

    Args:
        file_path: Path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized data.
    """
    encoding = detect_file_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        df.replace("-", np.nan, inplace=True)
        if "link" in df.columns:
            df.drop(columns=["link"], inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


# --------------------------------------------------
# 2) Filtering Logic
# --------------------------------------------------


def filter_investment_parameter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to keep only rows where parameter is 'investment'.

    Args:
        df (pd.DataFrame): Input DataFrame containing cost data with a 'parameter' column.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only investment parameter rows.
            If 'parameter' column does not exist, returns the original DataFrame unchanged.
    """
    if "parameter" not in df.columns:
        logging.warning("Warning: 'parameter' column not found.")
        return df
    # Filter for rows that have parameter == "investment"
    mask = df["parameter"].str.lower() == "investment"
    df_investment = df[mask].copy()
    return df_investment


def filter_technologies_by_config(df: pd.DataFrame, 
                                techs_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Filter and categorize technologies based on the provided technology dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame containing technology data
        techs_dict (Dict[str, List[str]]): Dictionary containing technology lists with keys:
            - 'vre_techs': List of variable renewable energy technologies
            - 'conv_techs': List of conventional technologies
            - 'store_techs': List of storage technologies

    Returns:
        pd.DataFrame: Filtered DataFrame with mapped technologies and categories
            - 'mapped_technology': Standardized technology names
            - 'category': Technology category (VRE, Conventional, Storage, Solar Thermal)
    """
    try:
        # Validate input DataFrame
        if df.empty:
            logging.warning("Warning: Input DataFrame is empty")
            return pd.DataFrame()
            
        if "technology" not in df.columns:
            logging.error("Error: 'technology' column not found in DataFrame")
            return pd.DataFrame()

        # Example aliases
        alias_map: Dict[str, List[str]] = {
            "solar thermal": ["central solar thermal", "decentral solar thermal"],
            "hydroelectricity": ["hydro"],
            "heat pump": ["central air heat pump", "decentral air heat pump"],
            "resistive heater": ["central resistive heater", "decentral resistive heater"],
            "Sabatier": ["methanation"],
            "H2 CHP": ["central hydrogen CHP"],
            "OCGT gas": ["OCGT"],
            "CHP gas": ["central gas CHP", "decentral CHP"],
            "gas boiler": ["central gas boiler", "decentral gas boiler"],
            "coal boiler": ["central coal boiler", "decentral coal boiler"],
            "coal power plant": ["coal"],
            "CHP coal": ["central coal CHP"],
            "H2": ["H2 pipeline", "hydrogen storage tank type 1"],
            "battery": ["battery storage"],
            "water tanks": ["central water tank storage", "decentral water tank storage"]
        }

        # Reverse alias map for easier lookup
        tech_aliases: Dict[str, str] = {}
        for main_tech, aliases in alias_map.items():
            for alias in aliases:
                tech_aliases[alias.lower()] = main_tech

        # get all techs
        vre_techs = techs_dict.get("vre_techs", [])
        conv_techs = techs_dict.get("conv_techs", [])
        store_techs = techs_dict.get("store_techs", [])
        solar_thermal = ["solar thermal"]  # special case for solar thermal
        
        # combine all techs
        all_techs = vre_techs + conv_techs + store_techs + solar_thermal

        # create tech to category mapping
        tech_categories: Dict[str, str] = {}
        for tech in vre_techs:
            if tech != "solar thermal":  # exclude solar thermal
                tech_categories[tech] = "VRE Technologies"
        for tech in conv_techs:
            tech_categories[tech] = "Conventional Technologies"
        for tech in store_techs:
            tech_categories[tech] = "Storage Technologies"
        tech_categories["solar thermal"] = "Solar Thermal"

        # Create a copy of the DataFrame
        df_filtered = df.copy()
        
        # Direct mapping
        df_filtered["mapped_technology"] = df_filtered["technology"].where(
            df_filtered["technology"].isin(all_techs)
        )
        
        # Alias mapping
        df_filtered["mapped_technology"].fillna(
            df_filtered["technology"].str.lower().map(tech_aliases),
            inplace=True
        )
        
        # Log unmapped technologies
        unmapped_techs = df_filtered[df_filtered["mapped_technology"].isna()]["technology"].unique()
        if len(unmapped_techs) > 0:
            logging.warning(
                f"Warning: The following technologies could not be mapped: {unmapped_techs}"
            )
        
        # Drop rows where no match was found
        df_filtered = df_filtered.dropna(subset=["mapped_technology"])
        
        if df_filtered.empty:
            logging.warning("Warning: No matching technologies found!")
            return pd.DataFrame()

        # Add category based on mapped technology
        df_filtered["category"] = df_filtered["mapped_technology"].map(tech_categories)
        
        return df_filtered
        
    except KeyError as e:
        logging.error(f"Error: Missing key in technology dictionary - {str(e)}")
        return pd.DataFrame()
    except AttributeError as e:
        logging.error(f"Error: Invalid DataFrame structure - {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in technology filtering: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------
# 3) Unit Conversion
# --------------------------------------------------


def parse_unit_string(unit_str: str) -> Tuple[str | None, str | None]:
    """Parse a unit string into currency and capacity parts.

    Args:
        unit_str (str): Unit string in format "currency/capacity"

    Returns:
        Tuple[str | None, str | None]: 
            - First element: currency part (e.g., "eur", "usd", "cny")
            - Second element: capacity part (e.g., "kw", "kwh")
            Returns (None, None) if invalid format
    """
    if pd.isna(unit_str) or "/" not in unit_str:
        return None, None
    currency_part, capacity_part = unit_str.split("/", 1)
    return currency_part.strip().lower(), capacity_part.strip().lower()


def is_storage_unit(capacity_part: str) -> bool:
    """Check if the capacity part indicates a storage unit.

    Args:
        capacity_part (str): Capacity part of the unit string

    Returns:
        bool: True if the unit is for storage (kWh or MWh)
    """
    return "kwh" in capacity_part.lower() or "mwh" in capacity_part.lower()


def get_capacity_factor(orig_cap: str, target_cap: str) -> float:
    """Calculate the capacity conversion factor.

    Args:
        orig_cap (str): Original capacity unit
        target_cap (str): Target capacity unit

    Returns:
        float: Conversion factor between capacity units
    """
    capacity_factors: Dict[Tuple[str, str], float] = {
        ("kw", "kw"): 1.0,
        ("mw", "kw"): 1/1000.0,
        ("kwh", "kwh"): 1.0,
        ("mwh", "kwh"): 1/1000.0,
        ("kw", "mw"): 1000.0,
        ("mw", "mw"): 1.0,
        ("kwh", "mwh"): 1000.0,
        ("mwh", "mwh"): 1.0
    }
    return capacity_factors.get((orig_cap.lower(), target_cap.lower()), 1.0)


def normalize_capacity_unit(cap: str) -> str:
    """Normalize capacity unit string.

    Args:
        cap (str): Capacity unit string

    Returns:
        str: Normalized capacity unit
            - Converts to lowercase
            - Removes spaces
            - Standardizes square notation
    """
    if not cap:
        return ""
    cap = cap.lower()
    cap = cap.replace(" ", "")
    cap = cap.replace("²", "2")
    return cap


def get_numeric_columns(row: pd.Series) -> List[str]:
    """Get list of numeric columns in the row.

    Args:
        row (pd.Series): Input row

    Returns:
        List[str]: List of column names containing numeric values
            - Year columns (e.g., "2020", "2025")
            - Cost columns (e.g., "cost_2020", "cost_2025")
    """
    numeric_cols = []
    
    # Check for year columns
    for col in row.index:
        if isinstance(col, str) and col.isdigit():
            numeric_cols.append(col)
            
    # Check for cost columns
    if not numeric_cols:
        for col in row.index:
            if isinstance(col, str) and "cost_" in col and col.split("_")[-1].isdigit():
                numeric_cols.append(col)
                
    # Check for specific years
    possible_years = ["2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060"]
    for year in possible_years:
        if year in row.index and year not in numeric_cols:
            numeric_cols.append(year)
            
    return numeric_cols


def convert_row_units(row: pd.Series, 
                     target_unit_installation: str,
                     target_unit_storage: str) -> pd.Series:
    """Convert row's numeric columns to target units.

    Args:
        row (pd.Series): Input row containing cost data
        target_unit_installation (str): Target unit for installation costs (e.g. "eur/kW")
        target_unit_storage (str): Target unit for storage costs (e.g. "eur/kWh")

    Returns:
        pd.Series: Row with converted values
    """
    if "unit" not in row or pd.isna(row["unit"]):
        return row

    # Parse unit string
    currency_part, capacity_part = parse_unit_string(row["unit"])
    if not currency_part or not capacity_part:
        return row

    # Determine if storage unit and get target unit
    is_storage = is_storage_unit(capacity_part)
    target_unit = target_unit_storage if is_storage else target_unit_installation
    tgt_currency, tgt_capacity = parse_unit_string(target_unit)

    # Get currency conversion factor
    orig_currency = CURRENCY_ALIASES.get(currency_part.lower(), "unknown")
    currency_factor = EXCHANGE_RATES.get((orig_currency, tgt_currency), 1.0)

    # Get capacity conversion factor
    norm_capacity = normalize_capacity_unit(capacity_part)
    norm_target = normalize_capacity_unit(tgt_capacity)
    
    if norm_capacity in ["kw", "mw", "kwh", "mwh"] and norm_target in ["kw", "mw", "kwh", "mwh"]:
        cap_factor = get_capacity_factor(norm_capacity, norm_target)
        new_capacity_part = norm_target
    else:
        cap_factor = 1.0
        new_capacity_part = capacity_part

    # Calculate total conversion factor
    factor = currency_factor * cap_factor

    # Convert numeric columns
    numeric_cols = get_numeric_columns(row)
    for col in numeric_cols:
        val = row[col]
        if pd.notna(val):
            try:
                row[col] = float(val) * factor
            except ValueError:
                pass

    # Update unit string
    row["unit"] = f"{tgt_currency.upper()}/{new_capacity_part}"

    return row


def apply_conversion(df: pd.DataFrame, 
                    target_unit_installation: str, 
                    target_unit_storage: str) -> pd.DataFrame:
    """Apply unit conversion to each row in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing cost data
        target_unit_installation (str): Target unit for installation costs (e.g. "eur/kW")
        target_unit_storage (str): Target unit for storage costs (e.g. "eur/kWh")

    Returns:
        pd.DataFrame: DataFrame with converted units
            - Preserves original unit in 'original_unit' column
            - Updates 'unit' column with new unit
            - Converts all numeric values to target units
    """
    # Ensure essential columns exist
    for col in ["technology", "reference", "unit"]:
        if col not in df.columns:
            df[col] = f"Default_{col}"

    # Keep the original unit in a separate column if not present
    if "original_unit" not in df.columns:
        df["original_unit"] = df["unit"].copy()

    # Convert row by row
    return df.apply(
        convert_row_units,
        axis=1,
        args=(target_unit_installation, target_unit_storage)
    )


def load_reference_data(file_path: str) -> pd.DataFrame:
    """Load and process reference data for technology cost comparison.

    Args:
        file_path: Path to the reference data CSV file.

    Returns:
        pd.DataFrame: Processed reference data with standardized units.
    """
    try:
        ref_df = load_and_clean_data(file_path)
        ref_df = ref_df[ref_df["reference"] != "PyPSA-China"]
        
        if "parameter" in ref_df.columns:
            ref_df = filter_investment_parameter(ref_df)
        ref_df = apply_conversion(ref_df, "eur/kW", "eur/kWh")
        return ref_df
    except Exception as e:
        logging.error(f"Error loading reference data: {e}")
        return pd.DataFrame()


def plot_technologies_by_category(
    costs_df: pd.DataFrame,
    ref_df: pd.DataFrame | None = None,
    tech_colors: Dict[str, str] | None = None,
    font_size: int = 14,
    plot_reference: bool = True
) -> plt.Figure:
    """Plot technology cost trends with literature comparison.

    Args:
        costs_df (pd.DataFrame): DataFrame containing the main cost data
        ref_df (pd.DataFrame | None): DataFrame containing reference data for comparison
        tech_colors (Dict[str, str] | None): Dictionary mapping technology names to colors
        font_size (int): Font size for plot elements
        plot_reference (bool): Whether to plot reference data

    Returns:
        plt.Figure: The generated plot figure
    """
    if costs_df.empty:
        logging.error("Error: The costs DataFrame is empty; cannot plot.")
        return None

    if tech_colors is None:
        tech_colors = {}

    # Get year columns
    year_cols = [col for col in costs_df.columns if col.isdigit()]
    if not year_cols:
        year_cols = [col.split("_")[-1] for col in costs_df.columns 
                    if "cost_" in col and col.split("_")[-1].isdigit()]
    if not year_cols:
        possible_years = ["2020", "2025", "2030", "2035", "2040", 
                         "2045", "2050", "2055", "2060"]
        year_cols = [year for year in possible_years if year in costs_df.columns]
    if not year_cols:
        logging.error("Error: Unable to identify year columns in costs data.")
        return None
    year_cols = sorted(year_cols)

    ref_year_cols = [col for col in ref_df.columns if col.isdigit()] if ref_df is not None else []
    if plot_reference and ref_df is not None and not ref_year_cols:
        logging.error("Error: Unable to identify year columns in reference data.")
        return None
    ref_year_cols = sorted(ref_year_cols)

    # Create subplots
    technologies = costs_df["technology"].unique()
    num_techs = len(technologies)
    num_cols = 6
    num_rows = (num_techs + num_cols - 1) // num_cols

    fig, axs = uplt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figwidth=6 * num_cols,
        sharex=True,
        sharey=False
    )
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()

    dash_styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]

    # Plot each technology
    for i, tech in enumerate(technologies):
        if i >= len(axs):
            logging.warning(f"Warning: Exceeded subplot limit, skipping {tech}")
            continue

        ax = axs[i]
        tech_df = costs_df[costs_df["technology"] == tech]
        ref_tech_df = ref_df[ref_df["technology"] == tech] if ref_df is not None else None
        color = tech_colors.get(tech, "#999999")

        # Plot main data
        tech_years, tech_values = [], []
        for year in year_cols:
            values = pd.to_numeric(tech_df[year], errors='coerce').dropna().values
            if values.size > 0:
                tech_years.append(int(year))
                tech_values.append(np.median(values))

        legend_handles = []
        legend_labels = []
        if tech_years:
            line, = ax.plot(
                tech_years,
                tech_values,
                linewidth=2.5,
                color=color,
                linestyle='-',
                label=tech
            )
            legend_handles.append(line)
            legend_labels.append(tech)

        # Plot reference data
        if plot_reference and ref_tech_df is not None and ref_tech_df.shape[0] > 0:
            for j, (ref_name, ref_group) in enumerate(ref_tech_df.groupby("reference")):
                ref_years, ref_values = [], []
                for year in ref_year_cols:
                    vals = pd.to_numeric(ref_group[year], errors='coerce').dropna().values
                    if vals.size > 0:
                        ref_years.append(int(year))
                        ref_values.append(np.median(vals))

                if ref_years:
                    dash_style = dash_styles[j % len(dash_styles)]
                    ref_line, = ax.plot(
                        ref_years,
                        ref_values,
                        linewidth=2,
                        color=color,
                        linestyle=dash_style,
                        label=ref_name
                    )
                    legend_handles.append(ref_line)
                    legend_labels.append(ref_name)

        # Set labels and formatting
        if "unit" in tech_df.columns and not tech_df["unit"].isna().all():
            unit = tech_df["unit"].iloc[0]
            unit_parts = unit.split('/')
            if len(unit_parts) == 2:
                currency, capacity = unit_parts
                if 'eur' in currency.lower():
                    ax.set_ylabel(f"EUR/{capacity.upper()}", fontsize=font_size)
                else:
                    ax.set_ylabel(f"{unit}", fontsize=font_size)
            else:
                ax.set_ylabel(f"{unit}", fontsize=font_size)
        else:
            ax.set_ylabel("Cost (EUR)", fontsize=font_size)

        ax2 = ax.twinx()
        if "unit" in tech_df.columns and not tech_df["unit"].isna().all():
            unit = tech_df["unit"].iloc[0]
            unit_parts = unit.split('/')
            if len(unit_parts) == 2:
                _, capacity = unit_parts
                ax2.set_ylabel(f"CNY/{capacity.upper()}", fontsize=font_size)
            else:
                ax2.set_ylabel("Cost (CNY)", fontsize=font_size)
        else:
            ax2.set_ylabel("Cost (CNY)", fontsize=font_size)

        y_min, y_max = ax.get_ylim()
        ax2.set_ylim(y_min * 7.8, y_max * 7.8)

        ax.set_title(f"{tech}", fontsize=font_size)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=font_size)
        ax2.tick_params(axis='y', labelsize=font_size)
        ax.legend(
            legend_handles,
            legend_labels,
            loc='upper right',
            fontsize=font_size-2,
            ncol=1
        )

    # Hide unused subplots
    total_plots = num_rows * num_cols
    for i in range(num_techs, total_plots):
        if i < len(axs):
            axs[i].set_visible(False)

    axs[0].figure.format(
        suptitle="Technology Cost Trends with Literature Comparison",
        abc=True,
        abcloc="ul",
        xlabel="Year",
        fontsize=font_size
    )

    return fig


# --------------------------------------------------
# 5) Main Execution (Snakemake or Standalone)
# --------------------------------------------------

if __name__ == "__main__":
    if 'snakemake' not in globals():
        snakemake = mock_snakemake("plot_input_costs")
    
    target_unit_installation = "eur/kW"
    target_unit_storage = "eur/kWh"
    
    tech_colors = validate_hex_colors(snakemake.config["plotting"]["tech_colors"])
    
    all_data = pd.DataFrame()
    
    for cost_file in snakemake.input.costs:
        year = cost_file.split('_')[-1].split('.')[0]
        df = pd.read_csv(cost_file)
        if not df.empty:
            df[year] = df['value']
            all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        logging.error("Error: No data loaded!")
        sys.exit(1)

    # 1) Filter to keep only investment parameter
    all_data = filter_investment_parameter(all_data)

    # 2) Filter & unify to standard units
    all_data = apply_conversion(all_data, target_unit_installation, target_unit_storage)

    # 3) Filter technologies based on config
    techs_dict = snakemake.config["Techs"]
    filtered_data = filter_technologies_by_config(all_data, techs_dict)
    if filtered_data.empty:
        logging.warning("Warning: No data left after technology filtering!")
        sys.exit(0)

    # 4) Plot
    ref_df = None
    try:
        if snakemake.input.reference_costs is not None:
            ref_df = load_reference_data(snakemake.input.reference_costs)
    except AttributeError:
        pass
    
    fig = plot_technologies_by_category(
        filtered_data, 
        ref_df, 
        tech_colors,
        plot_reference=snakemake.params.get('plot_reference', True)
    )

    output_path = str(snakemake.output.cost_map)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info("Plot successfully generated")
