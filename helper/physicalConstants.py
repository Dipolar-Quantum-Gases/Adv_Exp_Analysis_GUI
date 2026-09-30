# physicalConstants.py
import numpy as np

# Universal physical constants
const = {
    "c": 299792458,           # Speed of light [m/s]
    "e": 1.602176634e-19,     # Elementary charge [C]
    "h": 6.626e-34,
    "epsilon_0": 8.854187817e-12,  # Vacuum permittivity [F/m]
    "mu_0": 1.25663706212e-6, # Vacuum permeability [N/A²]
    "a_0": 5.29177210903e-11, # Bohr radius [m]
    "k_B": 1.380649e-23,      # Boltzmann constant [J/K]
    "m_e": 9.1093837015e-31,  # Electron mass [kg]
    "m_p": 1.67262192369e-27, # Proton mass [kg]
    "alpha": 7.2973525693e-3, # Fine-structure constant (unitless)
    "Ry": 2.1798723611035e-18, # Rydberg energy [J]
    "u": 1.66053906660e-27, #atomic mass [kg]
    "mu_B": 9.274009994e-24, # Bohr magneton [J/T]
}
const["hbar"] = const["h"]/(2*np.pi)
const["au"] = 2.48832e-8*const["h"]

# Define atoms and their relevant parameters
atom = {
    "Rb87": {
        "m": 1.44316060e-25,  # Mass [kg]
        "D1": 794.8e-9,  # D1 transition wavelength [m]
        "D2": 780.2e-9,  # D2 transition wavelength [m]
    },
    "K39": {
        "m": 38.963707*const["u"], # Mass [kg]
        "D1": 770.1e-9,  # D1 transition wavelength [m]
        "D2": 766.7e-9,  # D2 transition wavelength [m]
        "pola_1064": 598.71*const["au"], # Polarisability at 1064nm
        "pola_1030": 645.23*const["au"], # Polarisability at 1030nm
        "pola_1064_D1": -3155.17*const["au"], # Polarisability at 1064nm for D1 line
        "pola_1030_D1": -2363.98*const["au"], # Polarisability at 1030nm for D1 line
        "FS": {
            "D1": 0, # Finstructure splitting for D1 line [MHz]
            "D2": 0, # Fine structure splitting for D2 line [MHz]
        }
    },
    "K40": {
        "m": 39.963999*const["u"],  # Mass [kg]
        "D1": 770.1e-9,  # D1 transition wavelength [m]
        "D2": 766.7e-9,  # D2 transition wavelength [m]
        "pola_1064": 598.71*const["au"], # Polarisability at 1064nm
        "pola_1030": 645.23*const["au"], # Polarisability at 1030nm
        "pola_1064_D1": -3155.17*const["au"], # Polarisability at 1064nm for D1 line
        "pola_1030_D1": -2363.98*const["au"] # Polarisability at 1030nm for D1 line
    },
    "K41": {
        "m": 40.961826*const["u"],  # Mass [kg]
        "D1": 770.1e-9,  # D1 transition wavelength [m]
        "D2": 766.7e-9,  # D2 transition wavelength [m]
        "pola_1064": 598.71*const["au"], # Polarisability at 1064nm
        "pola_1030": 645.23*const["au"], # Polarisability at 1030nm
        "pola_1064_D1": -3155.17*const["au"], # Polarisability at 1064nm for D1 line
        "pola_1030_D1": -2363.98*const["au"], # Polarisability at 1030nm for D1 line
        "FS": {
            "D1": 235.5, # Finstructure splitting for D1 line [MHz]
            "D2": 236.2 # Fine structure splitting for D2 line [MHz]
        },
        "HFS": {
            "S1/2_F1": -158.8, # Hyperfine splitting for S1/2 F=1 state [MHz]
            "S1/2_F2": 95.3, # Hyperfine splitting for S1/2 F=2 state [MHz]
            "P1/2_F1": -19.1, # Hyperfine splitting for P1/2 F=1 state [MHz]
            "P1/2_F2": 11.4, # Hyperfine splitting for P1/2 F=2 state [MHz]
            "P3/2_F1": 0, # Hyperfine splitting for P3/2 F=1 state [MHz]
            "P3/2_F2": 63.4 # Hyperfine splitting for P3/2 F=2 state [MHz]
        }

    },
    "Er166": {
        "m": 165.930299*const["u"],  # Mass [kg]
        "D1": None,  # TBD
        "D2": None,  # TBD
        "pola_1064": 166*const["au"], # Polarisability at 1064nm
        "pola_532": 430*const["au"], # Polarisability at 532nm
    },
    "Er167": {
        "m": 166.932054*const["u"],  # Mass [kg]
        "D1": None,  # TBD
        "D2": None,  # TBD
        "pola_1064": 166*const["au"], # Polarisability at 1064nm
        "pola_532": 430*const["au"], # Polarisability at 532nm
    },
    "Er168": {
        "m": 167.932376*const["u"],  # Mass [kg]
        "D1": None,  # TBD
        "D2": None,  # TBD
        "pola_1064": 166*const["au"], # Polarisability at 1064nm
        "pola_532": 430*const["au"], # Polarisability at 532nm
    },
    "Er170": {
        "m": 169.935470*const["u"],  # Mass [kg]
        "D1": None,  # TBD
        "D2": None,  # TBD
        "pola_1064": 166*const["au"], # Polarisability at 1064nm
        "pola_532": 430*const["au"], # Polarisability at 532nm
    }
}

