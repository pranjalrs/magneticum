import numpy as np

import utils

class BaryonFractionHandler():
    """
    A class to handle baryon fraction data.

    Parameters:
    - path (str): The path to the data file.

    Attributes:
    - z (float): The redshift value.
    - mmin (float): The minimum mass value.
    - mmax (float): The maximum mass value.
    - mass_range_str (str): The mass range as a string in scientific notation.
    - halo_mass (ndarray): An array of halo masses.
    - fgas_r500c (ndarray): An array of gas fractions at r500c.
    - fgas_rvir (ndarray): An array of gas fractions at rvir.
    - fbar_r500c (ndarray): An array of baryon fractions at r500c.
    - fbar_rvir (ndarray): An array of baryon fractions at rvir.
    - fgas_r500c_mean (float): The mean gas fraction at r500c.
    - fgas_rvir_mean (float): The mean gas fraction at rvir.
    - fbar_r500c_mean (float): The mean baryon fraction at r500c.
    - fbar_rvir_mean (float): The mean baryon fraction at rvir.
    """

    def __init__(self, path) -> None:
        self._load_baryon_fraction(path)
    
    def _load_baryon_fraction(self, path):
        """
        Load baryon fraction data from the specified file.

        Parameters:
        - path (str): The path to the data file.
        """
        self.z = utils.search_z_in_string(path)
        self.mmin, self.mmax = utils.search_mass_range_in_string(path)

        # save mass range as string in scientific notation
        self.mass_range_str = f'{self.mmin:.2E}_{self.mmax:.2E}'

        data = self._loadtxt(path)

        self.halo_mass = data[:, 0]
        self.fgas_r500c = data[:, 1]
        self.fgas_rvir = data[:, 2]
        self.fbar_r500c = data[:, 3]
        self.fbar_rvir = data[:, 4]

        self.fgas_r500c_mean = np.mean(self.fgas_r500c)
        self.fgas_rvir_mean = np.mean(self.fgas_rvir)
        self.fbar_r500c_mean = np.mean(self.fbar_r500c)
        self.fbar_rvir_mean = np.mean(self.fbar_rvir)


    @staticmethod
    def _loadtxt(path, **kwargs):
        """
        Load data from a text file.

        Parameters:
        - path (str): The path to the text file.
        - **kwargs: Additional keyword arguments to pass to np.loadtxt.

        Returns:
        - ndarray: The loaded data.
        """
        try:
            return np.loadtxt(path, **kwargs)
        
        except FileNotFoundError:
            print(f'File not found {path}')
            return None, None
