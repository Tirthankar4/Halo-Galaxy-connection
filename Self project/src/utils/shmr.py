"""
SHMR Utilities for Phase 2: Universal Residual Learning

This module provides utilities for loading and using the computed
Stellar-Halo Mass Relation (SHMR) from the Golden Training Set.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Union, Optional, Tuple

# Default paths
BASE_DIR = Path(__file__).resolve().parents[2]
SHMR_DIR = BASE_DIR / "outputs" / "phase2_shmr"
LOOKUP_TABLE_PATH = SHMR_DIR / "shmr_lookup_table.json"
FIT_PARAMS_PATH = SHMR_DIR / "shmr_fit_parameters.json"


class SHMR:
    """
    Stellar-Halo Mass Relation utilities.
    
    Provides both lookup table interpolation and parametric fit evaluation
    for predicting median stellar mass from halo mass.
    """
    
    def __init__(
        self,
        lookup_table_path: Optional[Path] = None,
        fit_params_path: Optional[Path] = None
    ):
        """
        Initialize SHMR utilities.
        
        Args:
            lookup_table_path: Path to shmr_lookup_table.json. If None, uses default.
            fit_params_path: Path to shmr_fit_parameters.json. If None, uses default.
        """
        self.lookup_table_path = lookup_table_path or LOOKUP_TABLE_PATH
        self.fit_params_path = fit_params_path or FIT_PARAMS_PATH
        
        # Load data
        self._load_lookup_table()
        self._load_fit_parameters()
        
        # Create interpolation function
        self._create_interpolator()
    
    def _load_lookup_table(self):
        """Load the SHMR lookup table."""
        if not self.lookup_table_path.exists():
            raise FileNotFoundError(
                f"SHMR lookup table not found: {self.lookup_table_path}\n"
                "Please run utils/phase2_compute_shmr.py first."
            )
        
        with open(self.lookup_table_path, 'r') as f:
            data = json.load(f)
        
        self.log_mh_bins = np.array(data['log_Mh_bin_centers'])
        self.log_mstar_median = np.array(data['log_Mstar_median'])
        self.log_mstar_std = np.array(data['log_Mstar_std'])
        self.log_mstar_p16 = np.array(data['log_Mstar_p16'])
        self.log_mstar_p84 = np.array(data['log_Mstar_p84'])
        self.n_galaxies = np.array(data['n_galaxies'])
    
    def _load_fit_parameters(self):
        """Load the fitted SHMR parameters."""
        if not self.fit_params_path.exists():
            raise FileNotFoundError(
                f"SHMR fit parameters not found: {self.fit_params_path}\n"
                "Please run utils/phase2_compute_shmr.py first."
            )
        
        with open(self.fit_params_path, 'r') as f:
            params = json.load(f)
        
        self.alpha = params['parameters']['alpha']
        self.beta = params['parameters']['beta']
        self.r_squared = params['statistics']['r_squared']
        self.n_bins = params['statistics']['n_bins']
        self.n_galaxies = params['statistics']['n_galaxies']
    
    def _create_interpolator(self, kind: str = 'cubic'):
        """
        Create interpolation function from lookup table.
        
        Args:
            kind: Interpolation type ('linear', 'cubic', etc.)
        """
        self.interp_func = interp1d(
            self.log_mh_bins,
            self.log_mstar_median,
            kind=kind,
            bounds_error=False,
            fill_value=(self.log_mstar_median[0], self.log_mstar_median[-1])
        )
        
        # Also create interpolators for uncertainty bounds
        self.interp_p16 = interp1d(
            self.log_mh_bins,
            self.log_mstar_p16,
            kind=kind,
            bounds_error=False,
            fill_value=(self.log_mstar_p16[0], self.log_mstar_p16[-1])
        )
        
        self.interp_p84 = interp1d(
            self.log_mh_bins,
            self.log_mstar_p84,
            kind=kind,
            bounds_error=False,
            fill_value=(self.log_mstar_p84[0], self.log_mstar_p84[-1])
        )
    
    def predict_from_fit(
        self,
        log_mh: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Predict stellar mass using the fitted power law.
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
        
        Returns:
            Predicted stellar mass in log10(M_solar)
        """
        return self.alpha * log_mh + self.beta
    
    def predict_from_interpolation(
        self,
        log_mh: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Predict stellar mass using interpolation of lookup table.
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
        
        Returns:
            Predicted stellar mass in log10(M_solar)
        """
        return self.interp_func(log_mh)
    
    def predict(
        self,
        log_mh: Union[float, np.ndarray],
        method: str = 'fit'
    ) -> Union[float, np.ndarray]:
        """
        Predict stellar mass from halo mass.
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
            method: Prediction method ('fit' or 'interpolation')
        
        Returns:
            Predicted stellar mass in log10(M_solar)
        """
        if method == 'fit':
            return self.predict_from_fit(log_mh)
        elif method == 'interpolation':
            return self.predict_from_interpolation(log_mh)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fit' or 'interpolation'.")
    
    def get_uncertainty(
        self,
        log_mh: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Get uncertainty bounds (16th-84th percentile) for stellar mass prediction.
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
        
        Returns:
            Tuple of (lower_bound, upper_bound) in log10(M_solar)
        """
        return self.interp_p16(log_mh), self.interp_p84(log_mh)
    
    def compute_residuals(
        self,
        log_mh: Union[float, np.ndarray],
        log_mstar_true: Union[float, np.ndarray],
        method: str = 'fit'
    ) -> Union[float, np.ndarray]:
        """
        Compute residuals: Δ = log(M*_true) - log(M*_predicted).
        
        This is the "delta" that will be predicted by the residual learning model.
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
            log_mstar_true: True stellar mass in log10(M_solar). Can be scalar or array.
            method: Prediction method for baseline ('fit' or 'interpolation')
        
        Returns:
            Residuals in log10(M_solar)
        """
        log_mstar_pred = self.predict(log_mh, method=method)
        return log_mstar_true - log_mstar_pred
    
    def apply_residuals(
        self,
        log_mh: Union[float, np.ndarray],
        residuals: Union[float, np.ndarray],
        method: str = 'fit'
    ) -> Union[float, np.ndarray]:
        """
        Apply residuals to SHMR prediction.
        
        Final prediction: log(M*_final) = log(M*_SHMR) + Δ
        
        Args:
            log_mh: Halo mass in log10(M_solar/h). Can be scalar or array.
            residuals: Predicted residuals Δ. Can be scalar or array.
            method: Prediction method for baseline ('fit' or 'interpolation')
        
        Returns:
            Final stellar mass prediction in log10(M_solar)
        """
        log_mstar_base = self.predict(log_mh, method=method)
        return log_mstar_base + residuals
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SHMR(α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"R²={self.r_squared:.4f}, N={self.n_galaxies})"
        )


def load_shmr(
    lookup_table_path: Optional[Path] = None,
    fit_params_path: Optional[Path] = None
) -> SHMR:
    """
    Convenience function to load SHMR utilities.
    
    Args:
        lookup_table_path: Path to shmr_lookup_table.json. If None, uses default.
        fit_params_path: Path to shmr_fit_parameters.json. If None, uses default.
    
    Returns:
        SHMR instance
    """
    return SHMR(lookup_table_path, fit_params_path)


# Example usage
if __name__ == "__main__":
    # Load SHMR
    shmr = load_shmr()
    print(shmr)
    print()
    
    # Test predictions
    test_log_mh = np.array([10.5, 11.0, 11.5, 12.0, 12.5])
    
    print("Test predictions:")
    print("-" * 60)
    print(f"{'log(M_h)':<12} {'Fit':<12} {'Interp':<12} {'Difference':<12}")
    print("-" * 60)
    
    for mh in test_log_mh:
        pred_fit = shmr.predict_from_fit(mh)
        pred_interp = shmr.predict_from_interpolation(mh)
        diff = pred_fit - pred_interp
        print(f"{mh:<12.2f} {pred_fit:<12.4f} {pred_interp:<12.4f} {diff:<12.4f}")
    
    print()
    
    # Test uncertainty
    print("Uncertainty bounds:")
    print("-" * 80)
    print(f"{'log(M_h)':<12} {'Median':<12} {'16th %ile':<12} {'84th %ile':<12} {'Spread':<12}")
    print("-" * 80)
    
    for mh in test_log_mh:
        median = shmr.predict_from_interpolation(mh)
        p16, p84 = shmr.get_uncertainty(mh)
        spread = p84 - p16
        print(f"{mh:<12.2f} {median:<12.4f} {p16:<12.4f} {p84:<12.4f} {spread:<12.4f}")
    
    print()
    
    # Test residual computation
    print("Residual computation example:")
    log_mh_example = 11.5
    log_mstar_true = 9.5  # Example true stellar mass
    residual = shmr.compute_residuals(log_mh_example, log_mstar_true)
    print(f"  M_h = 10^{log_mh_example} M☉/h")
    print(f"  M_* (true) = 10^{log_mstar_true} M☉")
    print(f"  M_* (SHMR) = 10^{shmr.predict(log_mh_example):.4f} M☉")
    print(f"  Residual Δ = {residual:.4f}")

