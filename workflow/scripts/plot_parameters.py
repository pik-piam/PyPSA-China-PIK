import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chardet
import logging
import sys
import yaml
import re
import ultraplot as uplt
from typing import Optional, Dict
import pypsa
# --------------------------------------------------
# 1) Helper Functions
# --------------------------------------------------


def detect_file_encoding(file_path: str) -> str:
    """Detect the encoding of a file.

    Args:
        file_path: Path to the file whose encoding needs to be detected.

    Returns:
        str: The detected encoding of the file.

    Example:
        >>> encoding = detect_file_encoding("data.csv")
        >>> print(encoding)
        'utf-8'
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

    Example:
        >>> df = load_and_clean_data("data.csv")
        >>> print(df.head())
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


def standardize_unit(unit_str):
    """
    Standardize a unit string by trimming whitespace,
    converting to lowercase, and removing internal spaces.
    """
    if pd.isna(unit_str):
        return ""
    return str(unit_str).strip().lower().replace(" ", "")


def load_config(yaml_path):
    """
    Read the 'Techs' section from a YAML file (e.g., default_config.yaml).
    Returns a dictionary with keys like 'vre_techs', 'conv_techs'.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        techs = config["Techs"]
        # If Techs is a string, try to parse it again
        if isinstance(techs, str):
            techs = yaml.safe_load(techs)
        return techs
    except Exception as e:
        logging.error(f"Error loading config file {yaml_path}: {e}")
        return {"vre_techs": [], "conv_techs": [], "store_techs": []}


def load_plot_config(plot_yaml_path):
    """
    Read the 'tech_colors' section under 'plotting' from the given YAML file.
    Ensures color values are valid HEX codes, defaulting to '#999999'.
    """
    try:
        with open(plot_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file {plot_yaml_path} was not found.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return {}

    tech_colors = config.get("plotting", {}).get("tech_colors", {})

    # Validate HEX colors
    hex_color_pattern = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$')
    for tech, color in tech_colors.items():
        if not isinstance(color, str) or not hex_color_pattern.match(color):
            tech_colors[tech] = "#999999"
    return tech_colors

# --------------------------------------------------
# 2) Filtering Logic
# --------------------------------------------------


def filter_investment_parameter(df):
    """
    Keep only rows where parameter == 'investment' (case-insensitive).
    If 'parameter' column does not exist, we return df unchanged.
    """
    if "parameter" not in df.columns:
        logging.warning("Warning: 'parameter' column not found.")
        return df
    # Filter for rows that have parameter == "investment"
    mask = df["parameter"].str.lower() == "investment"
    df_investment = df[mask].copy()
    return df_investment


def filter_technologies_by_config(df, default_config_path):
    """
    Filter and categorize technologies based on the Techs section
    of the default_config.yaml file. An alias map is used for direct
    and fuzzy matching of technology names in the DataFrame.
    Rows whose technology is not found in the config or alias map are dropped.
    """
    try:
        techs_dict = load_config(default_config_path)

        # Example aliases
        alias_map = {
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
        tech_aliases = {}
        for main_tech, aliases in alias_map.items():
            for alias in aliases:
                tech_aliases[alias.lower()] = main_tech

        # Separate out "solar thermal" from VRE
        vre_techs = [t for t in techs_dict.get("vre_techs", []) if t != "solar thermal"]
        conv_techs = techs_dict.get("conv_techs", [])
        store_techs = techs_dict.get("store_techs", [])
        solar_thermal = ["solar thermal"]

        # Combine all known techs
        all_techs = vre_techs + conv_techs + store_techs + solar_thermal

        # Map from tech -> category
        tech_categories = {}
        for t in vre_techs:
            tech_categories[t] = "VRE Technologies"
        for t in conv_techs:
            tech_categories[t] = "Conventional Technologies"
        for t in store_techs:
            tech_categories[t] = "Storage Technologies"
        for t in solar_thermal:
            tech_categories[t] = "Solar Thermal"

        # Build a mapping from DF tech -> config tech
        df_techs = df["technology"].unique()
        tech_mapping = {}

        for df_tech in df_techs:
            df_tech_lower = df_tech.lower()

            # 1) Direct match
            if df_tech_lower in [t.lower() for t in all_techs]:
                for t in all_techs:
                    if t.lower() == df_tech_lower:
                        tech_mapping[df_tech] = t
                        break
                continue

            # 2) Alias match
            if df_tech_lower in tech_aliases:
                tech_mapping[df_tech] = tech_aliases[df_tech_lower]
                continue

        # Filter rows: only keep those that mapped to a known tech
        filtered_rows = []
        for idx, row in df.iterrows():
            tech = row["technology"]
            if tech in tech_mapping:
                mapped_tech = tech_mapping[tech]
                category = tech_categories.get(mapped_tech)
                if category:
                    new_row = row.copy()
                    new_row["mapped_technology"] = mapped_tech
                    new_row["category"] = category
                    filtered_rows.append(new_row)

        if not filtered_rows:
            logging.warning("Warning: No matching technologies found!")
            return pd.DataFrame()

        filtered_df = pd.DataFrame(filtered_rows)
        return filtered_df
    except Exception as e:
        logging.error(f"Error filtering technologies: {e}")
        return df

# --------------------------------------------------
# 3) Unit Conversion
# --------------------------------------------------


def convert_row_units(row, target_unit_installation, target_unit_storage):
    """
    Convert row's numeric columns so that:
      - If the original unit is recognized as storage (kWh or MWh in the capacity part),
        we unify to target_unit_storage (e.g. "eur/kWh").
      - Otherwise, we unify to target_unit_installation (e.g. "eur/kW").
      - If the capacity part is not in [kW, MW, kWh, MWh], we keep it as-is.
      
    Assumes exchange rates:
      1 EUR = 7.8 CNY
      1 USD = 7.2 CNY
    """
    if "unit" not in row or pd.isna(row["unit"]):
        return row  # no unit, do nothing

    original_unit = str(row["unit"]).strip()
    if "/" not in original_unit:
        return row

    # -----------------------------
    # -----------------------------
    currency_part, capacity_part = original_unit.split("/", 1)
    currency_part = currency_part.strip().lower()
    capacity_part = capacity_part.strip().lower()

    is_storage = False
    if "kwh" in capacity_part or "mwh" in capacity_part:
        is_storage = True

    # -----------------------------
    #    e.g. target_unit_installation = "eur/kW"
    #         target_unit_storage      = "eur/kWh"
    # -----------------------------
    def parse_target_unit(unit_str):
        """
        E.g. "eur/kW" -> (target_currency="eur", target_capacity="kw")
        """
        parts = unit_str.lower().split("/")
        if len(parts) == 2:
            return parts[0], parts[1]  # currency, capacity
        else:
            return parts[0], None

    if is_storage:
        tgt_currency, tgt_capacity = parse_target_unit(target_unit_storage)
    else:
        tgt_currency, tgt_capacity = parse_target_unit(target_unit_installation)

    # -----------------------------
    # -----------------------------
    currency_aliases = {
        "eur": "eur",
        "€":   "eur",
        "usd": "usd",
        "$":   "usd",
        "cny": "cny",
        "rmb": "cny"
    }

    orig_currency = None
    for k, v in currency_aliases.items():
        if k in currency_part:
            orig_currency = v
            break
    if orig_currency is None:
        orig_currency = "unknown"

    # Table of exchange rates
    exchange_map = {
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

    currency_factor = exchange_map.get((orig_currency, tgt_currency), 1.0)

    # -----------------------------
    # -----------------------------
    def capacity_factor_func(orig_cap, new_cap):

        if orig_cap == "kw" and new_cap == "kw":
            return 1.0
        if orig_cap == "mw" and new_cap == "kw":
            return 1/1000.0
        if orig_cap == "kwh" and new_cap == "kwh":
            return 1.0
        if orig_cap == "mwh" and new_cap == "kwh":
            return 1/1000.0
        if orig_cap == "kw" and new_cap == "mw":
            return 1000.0
        if orig_cap == "mw" and new_cap == "mw":
            return 1.0
        if orig_cap == "kwh" and new_cap == "mwh":
            return 1000.0
        if orig_cap == "mwh" and new_cap == "mwh":
            return 1.0
        return 1.0 

    def normalize_capacity_unit(cap):
        cap = cap.lower()
        cap = cap.replace(" ", "")
        cap = cap.replace("²", "2")  
        return cap

    norm_capacity = normalize_capacity_unit(capacity_part)
    norm_target = normalize_capacity_unit(tgt_capacity if tgt_capacity else "")

    if norm_capacity in ["kw", "mw", "kwh", "mwh"] and norm_target in ["kw", "mw", "kwh", "mwh"]:
        cap_factor = capacity_factor_func(norm_capacity, norm_target)
        new_capacity_part = norm_target  
    else:

        cap_factor = 1.0
        new_capacity_part = capacity_part

    factor = currency_factor * cap_factor

    # -----------------------------
    numeric_cols = []
    for col in row.index:
        if col != "year" and isinstance(col, str) and col.isdigit():
            numeric_cols.append(col)

    if not numeric_cols:
        for col in row.index:
            if isinstance(col, str) and "cost_" in col and col.split("_")[-1].isdigit():
                numeric_cols.append(col)

    possible_years = ["2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060"]
    for y in possible_years:
        if y in row.index and y not in numeric_cols:
            numeric_cols.append(y)

    for col in numeric_cols:
        val = row[col]
        if pd.notna(val):
            try:
                row[col] = float(val) * factor
            except ValueError:
                pass

    # -----------------------------
    # e.g. "EUR/kW", "EUR/ton", "EUR/m^2" ...
    row["unit"] = f"{tgt_currency.upper()}/{new_capacity_part}"

    return row


def apply_conversion(df, target_unit_installation, target_unit_storage):
    """
    Apply the unit conversion to each row. Preserves the original unit
    in 'original_unit' before overwriting 'unit'.
    """
    # Ensure essential columns exist
    for col in ["technology", "reference", "unit"]:
        if col not in df.columns:
            df[col] = f"Default_{col}"

    # Keep the original unit in a separate column if not present
    if "original_unit" not in df.columns:
        df["original_unit"] = df["unit"].copy()

    # Convert row by row
    df = df.apply(
        convert_row_units,
        axis=1,
        args=(target_unit_installation, target_unit_storage)
    )
    return df


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
    ref_df: pd.DataFrame,
    tech_colors: Optional[Dict[str, str]] = None,
    font_size: int = 14,
    plot_reference: bool = True
) -> plt.Figure:
    """Plot technology cost trends with literature comparison.

    Args:
        costs_df: DataFrame containing the main cost data.
        ref_df: DataFrame containing reference data for comparison.
        tech_colors: Dictionary mapping technology names to colors.
        font_size: Font size for plot elements.
        plot_reference: Boolean indicating whether to plot reference data.

    Returns:
        plt.Figure: The generated plot figure.

    Example:
        >>> costs_df = load_and_clean_data("costs.csv")
        >>> ref_df = load_reference_data("reference.csv")
        >>> fig = plot_technologies_by_category(costs_df, ref_df)
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

    ref_year_cols = [col for col in ref_df.columns if col.isdigit()]
    if not ref_year_cols:
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
        ref_tech_df = ref_df[ref_df["technology"] == tech]
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
        if plot_reference and ref_tech_df.shape[0] > 0:
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


def extract_costs_from_network(n: pypsa.Network) -> pd.DataFrame:
    """Extract cost data from a PyPSA network object.

    Extract investment cost data for all components from the PyPSA network object,
    including generators, storage units, transmission lines, and transformers.

    Args:
        n: PyPSA network object

    Returns:
        pd.DataFrame: DataFrame containing the following columns:
            - technology: Name of the technology
            - parameter: Parameter type (fixed as 'investment')
            - unit: Unit of cost (EUR/kW or EUR/kWh)
            - value: Cost value
    """
    costs_data = []
    
    if not n.generators.empty:
        for carrier in n.generators.carrier.unique():
            gen_mask = n.generators.carrier == carrier
            if 'capital_cost' in n.generators.columns:
                cost = n.generators.loc[gen_mask, 'capital_cost'].mean()
                if not pd.isna(cost):
                    costs_data.append({
                        'technology': carrier,
                        'parameter': 'investment',
                        'unit': 'EUR/kW',
                        'value': cost
                    })
    
    if not n.storage_units.empty:
        for carrier in n.storage_units.carrier.unique():
            store_mask = n.storage_units.carrier == carrier
            if 'capital_cost' in n.storage_units.columns:
                cost = n.storage_units.loc[store_mask, 'capital_cost'].mean()
                if not pd.isna(cost):
                    costs_data.append({
                        'technology': carrier,
                        'parameter': 'investment',
                        'unit': 'EUR/kWh',
                        'value': cost
                    })
    
    if not n.lines.empty and 'capital_cost' in n.lines.columns:
        cost = n.lines.capital_cost.mean()
        if not pd.isna(cost):
            costs_data.append({
                'technology': 'AC line',
                'parameter': 'investment',
                'unit': 'EUR/kW',
                'value': cost
            })
    
    if not n.transformers.empty and 'capital_cost' in n.transformers.columns:
        cost = n.transformers.capital_cost.mean()
        if not pd.isna(cost):
            costs_data.append({
                'technology': 'transformer',
                'parameter': 'investment',
                'unit': 'EUR/kW',
                'value': cost
            })
    
    return pd.DataFrame(costs_data)

# --------------------------------------------------
# 5) Main Execution (Snakemake or Standalone)
# --------------------------------------------------


if __name__ == "__main__":
    if 'snakemake' in globals():
        target_unit_installation = "eur/kW"
        target_unit_storage = "eur/kWh"
        default_config_path = "config/default_config.yaml"
        plot_config_path = "config/plot_config.yaml"

        tech_colors = load_plot_config(plot_config_path)
        all_data = pd.DataFrame()
        
        for year, file_path in snakemake.input.costs.items():
            n = pypsa.Network(file_path)
            df = extract_costs_from_network(n)
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
        filtered_data = filter_technologies_by_config(all_data, default_config_path)
        if filtered_data.empty:
            logging.warning("Warning: No data left after technology filtering!")
            sys.exit(0)

        # 4) Plot
        ref_df = None
        if hasattr(snakemake.input, 'reference_costs'):
            ref_df = load_reference_data(snakemake.input.reference_costs)
        
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

    else:
        # Standalone mode for testing
        target_unit_installation = "eur/kW"
        target_unit_storage = "eur/kWh"
        default_config_path = "config/default_config.yaml"
        plot_config_path = "config/plot_config.yaml"
        file_path = "tech_costs_subset.csv"

        tech_colors = load_plot_config(plot_config_path)
        df = load_and_clean_data(file_path)
        
        if not df.empty:
            # 1) Filter to keep only investment parameter
            df = filter_investment_parameter(df)

            # 2) Convert units
            df = apply_conversion(df, target_unit_installation, target_unit_storage)

            # 3) Filter technologies
            filtered_df = filter_technologies_by_config(df, default_config_path)
            if filtered_df.empty:
                logging.warning("Warning: No data left after technology filtering!")
                sys.exit(0)

            # 4) Plot
            ref_df = load_reference_data("resources/data/costs/reference_values/tech_costs_subset_litreview.csv")
            fig = plot_technologies_by_category(filtered_df, ref_df, tech_colors)
            plt.savefig("test_output.png", bbox_inches='tight', dpi=300)
            logging.info("Test plot successfully generated")
        else:
            logging.error("Error: Unable to load test data.")
