"""EV sector reference data generator

Generates reference data for the EV sector using a simplified Gompertz model.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class GompertzModel:
    """Simplified Gompertz model"""
    
    def __init__(self, saturation_level: float = 500, alpha: float = -5.58):
        self.saturation_level = saturation_level
        self.alpha = alpha
        self.beta = None
        self.fitted = False
    
    def gompertz_function(self, pgdp: np.ndarray, beta: float) -> np.ndarray:
        """Gompertz function"""
        return self.saturation_level * np.exp(self.alpha * np.exp(beta * pgdp))
    
    def fit_model(self, pgdp_data: np.ndarray, vehicle_data: np.ndarray) -> bool:
        """Fit model parameter beta"""
        try:
            def objective_function(pgdp, beta):
                return self.gompertz_function(pgdp, beta)
            
            popt, _ = curve_fit(objective_function, pgdp_data, vehicle_data, 
                                p0=[-0.0001], bounds=([-1], [0]))
            self.beta = popt[0]
            self.fitted = True
            logger.info(f"Gompertz model fitted successfully - α: {self.alpha}, β: {self.beta:.6f}")
            return True
        except Exception as e:
            logger.error(f"Gompertz model fitting failed: {e}")
            return False
    
    def predict_vehicles(self, pgdp: float, population: float) -> float:
        """Predict vehicle numbers (10,000 vehicles)"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        vehicle_per_capita = self.gompertz_function(np.array([pgdp]), self.beta)[0]
        return vehicle_per_capita * population / 1000  # output in 10,000 vehicles


def _load_historical_data(input_files: dict) -> pd.DataFrame:
    """Load historical data (per-capita GDP, vehicle per 1000 people)"""
    gdp = pd.read_csv(input_files['historical_gdp'], index_col=0, encoding="gbk", comment='#') 
    pop = pd.read_csv(input_files['historical_pop'], index_col=0, encoding="gbk", comment='#')
    cars = pd.read_csv(input_files['historical_cars'], index_col=0, encoding="gbk", comment='#')

    province_mapping = {"Innermonglia": "InnerMongolia", "Innermongolia": "InnerMongolia"}
    gdp.rename(index=province_mapping, inplace=True)
    pop.rename(index=province_mapping, inplace=True)
    cars.rename(index=province_mapping, inplace=True)

    gdp_long = gdp.stack().rename("gdp").reset_index()
    pop_long = pop.stack().rename("population").reset_index()
    cars_long = cars.stack().rename("cars").reset_index()

    gdp_long.columns = ["province", "year", "gdp"]
    pop_long.columns = ["province", "year", "population"]
    cars_long.columns = ["province", "year", "cars"]

    df = gdp_long.merge(pop_long, on=["province", "year"], how="inner") \
                 .merge(cars_long, on=["province", "year"], how="inner")

    df = df[df["year"].str.isdigit()]
    df["year"] = df["year"].astype(int)

    df["pgdp"] = df["gdp"] / df["population"]
    df["vehicle_per_capita"] = df["cars"] / df["population"] * 1000

    return df[["province", "year", "pgdp", "vehicle_per_capita"]]


def _load_future_data(input_files: dict, years: list) -> pd.DataFrame:
    """Load future projections (population, GDP, per-capita GDP)"""
    pop_data = pd.read_excel(input_files['ssp2_pop'], sheet_name='SSP2', index_col=0)
    gdp_data = pd.read_excel(input_files['ssp2_gdp'], sheet_name='SSP2', index_col=0)
    pop_data.columns = pop_data.columns.map(str)
    gdp_data.columns = gdp_data.columns.map(str)
    
    province_mapping = {"Innermonglia": "InnerMongolia", "Innermongolia": "InnerMongolia"}
    pop_data.rename(index=province_mapping, inplace=True)
    gdp_data.rename(index=province_mapping, inplace=True)

    target_years = [str(y) for y in years]
    pop_sel = pop_data[target_years]
    gdp_sel = gdp_data[target_years]

    pop_long = pop_sel.stack().rename("population").reset_index()
    gdp_long = gdp_sel.stack().rename("gdp").reset_index()
    pop_long.columns = ["province", "year", "population"]
    gdp_long.columns = ["province", "year", "gdp"]

    df = pd.merge(pop_long, gdp_long, on=["province", "year"], how="inner")

    df["population"] = df["population"] / 10000   # in 10,000 persons
    df["pgdp"] = df["gdp"] / df["population"]
    df["year"] = df["year"].astype(int)

    return df


def generate_reference(years: list, input_files: dict, output_dir: str, config: dict = None):
    """Generate EV sector reference data"""
    logger.info("Generating EV sector reference data")
    
    model = GompertzModel(
        saturation_level=config.get('saturation_level', 500) if config else 500, # Nature Geoscience: https://doi.org/10.1038/s41561-023-01350-9
        alpha=config.get('alpha', -5.58) if config else -5.58 #Energy Policy: https://doi.org/10.1016/j.enpol.2011.01.043
    )
    
    historical_data = _load_historical_data(input_files)
    pgdp_data = historical_data['pgdp'].values
    vehicle_data = historical_data['vehicle_per_capita'].values
    model.fit_model(pgdp_data, vehicle_data)
    
    future_data = _load_future_data(input_files, years)
    
    future_data['vehicles'] = future_data.apply(
        lambda row: model.predict_vehicles(row['pgdp'], row['population']), axis=1
    )
    
    shares_by_year = {}
    for year in years:
        year_data = future_data[future_data['year'] == year]
        total_vehicles = year_data['vehicles'].sum()
        shares = year_data['vehicles'] / total_vehicles
        shares_by_year[year] = dict(zip(year_data['province'], shares))
    
    shares_df = pd.DataFrame(shares_by_year)
    shares_df.to_csv(f"{output_dir}/ev_passenger_shares.csv")
    shares_df.to_csv(f"{output_dir}/ev_freight_shares.csv")
    
    logger.info(f"EV reference data saved to {output_dir}")
