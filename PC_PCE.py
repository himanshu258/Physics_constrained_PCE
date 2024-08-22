import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from UQpy.surrogates import *
from scipy import special as sp
from pyDOE2 import lhs
import random
from UQpy.distributions import Uniform, JointIndependent, Normal
from sklearn.linear_model import Lars

def generate_lhs_samples(lb, ub, total_samples, fixed_dimension, fixed_value):
    """
    Generate a Latin Hypercube Sampling (LHS) design for a given set of bounds and fixed dimension.

    Parameters:
    lb (numpy.ndarray): Lower bounds for each dimension. The shape should be (num_dimensions,).
    ub (numpy.ndarray): Upper bounds for each dimension. The shape should be (num_dimensions,).
    total_samples (int): The total number of samples to generate.
    fixed_dimension (int): The dimension for which the fixed value is applied.
    fixed_value (float): The fixed value to apply to the specified dimension.

    Returns:
    numpy.ndarray: A 2D numpy array containing the LHS samples. The shape is (total_samples, num_dimensions).
    """
    
    ar = np.arange(len(lb))
    arr = np.delete(ar, fixed_dimension)
    num_variables = len(arr)
    lhs_samples_unit_interval = lhs(num_variables, samples=total_samples, criterion='maximin')
    lhs_samples_scaled = np.zeros_like(lhs_samples_unit_interval)
    for i in range(num_variables):
        lhs_samples_scaled[:, i] = lhs_samples_unit_interval[:, i] * (ub[arr][i] - lb[arr][i]) + lb[arr][i]
        
    lhs_samples = np.insert(lhs_samples_scaled, fixed_dimension, fixed_value, axis=1)

    return lhs_samples

def random_pairs(U, count):
    """
    Generate random pairs of indices for an n-dimensional array (U).

    Parameters:
    U (numpy.ndarray): An n-dimensional array.
    count (int): The number of random pairs to generate.

    Returns:
    numpy.ndarray: A 2D numpy array with shape (count, n) containing the random pairs of indices.
        Each row represents a tuple of indices (i1, i2, ..., in) where 0 <= i < shape of U along that dimension.
    """
    # Get the shape and number of dimensions of U
    dims = U.shape
    n_dims = len(dims)
    
    # Generate random indices for each dimension
    random_pairs = np.array([[random.randint(0, dims[i] - 1) for i in range(n_dims)] for _ in range(count)])
    
    return random_pairs

import numpy as np

def model(U, count):
    """
    This function generates a random subset of data points from an n-dimensional array (U) 
    and their corresponding coordinates.

    Parameters:
    U (numpy.ndarray): An n-dimensional array containing the original data.
    count (int): The number of random data points to generate.

    Returns:
    numpy.ndarray: A 2D array containing the random data points and their coordinates. 
    """
    # Determine the shape and dimensionality of U
    dims = U.shape
    n_dims = len(dims)
    
    # Create a list of linspace arrays for each dimension
    grid_coords = [np.linspace(0, 1, dims[i]) for i in range(n_dims)]
    
    # Generate a meshgrid for all dimensions
    mesh = np.meshgrid(*grid_coords)
    
    random_indices = random_pairs(U, count)
    
    # Gather the coordinates and corresponding values
    coordinates = [mesh[i].T[tuple(random_indices[:,j] for j in range(n_dims))] for i in range(n_dims)]
    y = U[tuple(random_indices[:,j] for j in range(n_dims))]
    
    # Stack the coordinates and values into a single array
    data = np.column_stack(coordinates + [y])
    
    return data

def PolynomialBasisDerivative(Multindex, PolynomialTypes, U, lb, ub, dimension=0, order=0):
    """
    Generate a multivariate polynomial basis derivative using the given multivariate indices, polynomial types, and input samples.

    Parameters:
    Multindex (numpy.ndarray): A 2D numpy array containing the multivariate indices.
        The shape should be (P, nvar), where P is the number of polynomial terms and nvar is the number of variables.
    PolynomialTypes (list): A list of strings representing the type of polynomial for each variable.
        Supported types are 'Hermite' and 'Legendre'.
    U (numpy.ndarray): A 2D numpy array containing the input samples.
        The shape should be (nsim, nvar), where nsim is the number of samples and nvar is the number of variables.
    dimension (int): The dimension for which the derivative is calculated.
    order (int): The order of the derivative.
    lb (numpy.ndarray): A 1D numpy array containing the lower bounds for each variable for Uniform distribution and means for Normal distribution.
        The shape should be (nvar,).
    ub (numpy.ndarray): A 1D numpy array containing the upper bounds for each variable for Uniform distribution and standard deviations for Normal distribution.
        The shape should be (nvar,).

    Returns:
    numpy.ndarray: A 2D numpy array containing the multivariate polynomial basis derivative.
        The shape is (nsim, P), where nsim is the number of samples and P is the number of polynomial terms.
    """
    nsim, nvar = U.shape
    P, nvar = Multindex.shape
    MultivariateBasisderivatives = np.empty([nsim, P])
    
    if nvar-1 < dimension:
        raise ValueError("Dimension is bigger")

    for i in range(P):
        basis = np.ones(nsim)
        for j in range(nvar):
            if PolynomialTypes[j] == 'Hermite':
                basis *= sp.hermitenorm(Multindex[i, j])((U[:, j] - lb[j])/ub[j])
                basis /= np.sqrt(np.math.factorial(Multindex[i, j]))
            elif PolynomialTypes[j] == 'Legendre':
                if j == dimension:
                    basis *= ((2/(ub[j] - lb[j]))**order)*sp.legendre(Multindex[i, j]).deriv(m=order)((2*U[:, j]- lb[j] - ub[j]) / (ub[j] - lb[j]))*np.sqrt((2 * Multindex[i, j] + 1))
                 
                else:
                    
                    basis *= sp.legendre(Multindex[i, j])((2*U[:, j]- lb[j] - ub[j]) / (ub[j] - lb[j]))*np.sqrt((2 * Multindex[i, j] + 1))

            else:
                raise ValueError("Unsupported PolynomialType")

        MultivariateBasisderivatives[:, i] = basis

    return MultivariateBasisderivatives

def standardize_sample(x, joint_distribution):
        """
        Static method: Standardize data based on the joint probability distribution.

        :param x: Input data generated from a joint probability distribution.
        :param joint_distribution: joint probability distribution from :py:mod:`UQpy` distribution object
        :return: Standardized data.
        """

        s = np.zeros(x.shape)
        inputs_number = len(x[0, :])
        if inputs_number == 1:
            marginals = [joint_distribution]
        else:
            marginals = joint_distribution.marginals

        for i in range(inputs_number):
            if type(marginals[i]) == Normal:
                s[:, i] = Polynomials.standardize_normal(x[:, i], mean=marginals[i].parameters['loc'],
                                                         std=marginals[i].parameters['scale'])
            elif type(marginals[i]) == Uniform:
                s[:, i] = Polynomials.standardize_uniform(x[:, i], marginals[i])
            else:
                raise TypeError("standarize_sample is defined only for Uniform and Gaussian marginal distributions")
        return s
    
class ReducedPce:
    def __init__(self, pce, Multindex, coeff, n_det, determ_pos=None):
        
        """
        Initialize the ReducedPCE object.

        :param pce: Polynomial chaos expansion object.
        :param Multindex: Multivariate indices of the polynomial basis.
        :param coeff: Coefficients of the PC^2 expansion.
        :param n_det: Number of deterministic variables.
        :param determ_pos: Positions of deterministic variables in the multivariate indices.
        """

        if determ_pos is None:
            determ_pos = list(np.arange(n_det))

        self.original_multindex = Multindex
        self.original_beta = coeff
        self.original_P, self.nvar = self.original_multindex.shape
        self.original_pce = pce
        self.determ_pos = determ_pos

        # Select basis containing only deterministic variables
        determ_multi_index = np.zeros(self.original_multindex.shape)
        determ_selection_mask = [False] * self.nvar

        for i in (determ_pos):
            determ_selection_mask[i] = True

        determ_multi_index[:, determ_selection_mask] = self.original_multindex[:, determ_selection_mask]

        self.determ_multi_index = determ_multi_index.astype(int)
        self.determ_basis = polynomial_chaos.PolynomialBasis.construct_arbitrary_basis(self.nvar, self.original_pce.polynomial_basis.distributions, self.determ_multi_index)

        reduced_multi_mask = self.original_multindex > 0

        reduced_var_mask = [True] * self.nvar
        for i in (determ_pos):
            reduced_var_mask[i] = False

        reduced_multi_mask = reduced_multi_mask * reduced_var_mask

        reduced_multi_index = np.zeros(self.original_multindex.shape) + (self.original_multindex * reduced_multi_mask)
        reduced_multi_index = reduced_multi_index[:, reduced_var_mask]

        self.reduced_positions = reduced_multi_mask.sum(axis=1) > 0
        reduced_multi_index = reduced_multi_index[self.reduced_positions, :]

        unique_basis, unique_positions, self.unique_indices = np.unique(reduced_multi_index, axis=0, return_index=True, return_inverse=True)

        P_unique, nrand = unique_basis.shape
        self.unique_basis = np.concatenate((np.zeros((1, nrand)), unique_basis), axis=0)
    
    def eval_coord(self,coordinates,return_coeff=False):
        
        coord_x=np.zeros((1,self.nvar))
        coord_x[0,self.determ_pos]=coordinates
        
        determ_basis_eval=polynomial_chaos.PolynomialBasis(self.nvar,len(self.determ_multi_index),self.determ_multi_index,self.determ_basis,self.original_pce.polynomial_basis.distributions).evaluate_basis(coord_x)
        
        determ_beta=np.transpose(determ_basis_eval*self.original_beta)

        reduced_beta=determ_beta[self.reduced_positions]
        complement_beta=determ_beta[~self.reduced_positions]

        unique_beta=[]
        for ind in np.unique(self.unique_indices):
            sum_beta=np.sum(reduced_beta[self.unique_indices==ind,0])
            unique_beta.append(sum_beta)

        unique_beta=np.array([0]+unique_beta)
        unique_beta[0]=unique_beta[0]+np.sum(complement_beta)
        
        if return_coeff==False:
            mean=unique_beta[0]
            var=np.sum(unique_beta[1:]**2)
            return mean,var
        else:
            return unique_beta
        
    def derive_coord(self,coordinates,PolynomialType, lb, ub, der_var,der_order, return_coeff=False):
        coord_x=np.zeros((1,self.nvar))
        coord_x[0,self.determ_pos]=coordinates
        coord_s=polynomial_chaos.Polynomials.standardize_sample(coord_x,self.original_pce.polynomial_basis.distributions)
        
        determ_multi_index=np.zeros(self.original_multindex.shape)
        determ_selection_mask=np.arange(self.nvar)==self.determ_pos
        
        determ_multi_index[:,determ_selection_mask]=self.original_multindex[:,determ_selection_mask]
        determ_multi_index=determ_multi_index.astype(int)
        determ_basis_eval=PolynomialBasisDerivative(determ_multi_index,PolynomialType,coord_s, lb, ub, dimension=der_var, order=der_order)
        determ_beta=np.transpose(determ_basis_eval*self.original_beta)

        reduced_beta=determ_beta[self.reduced_positions]
        complement_beta=determ_beta[~self.reduced_positions]

        unique_beta=[]
        for ind in np.unique(self.unique_indices):
            sum_beta=np.sum(reduced_beta[self.unique_indices==ind,0])
            unique_beta.append(sum_beta)

        unique_beta=np.array([0]+unique_beta)
        unique_beta[0]=unique_beta[0]+np.sum(complement_beta)
        
        if return_coeff==False:
            mean=unique_beta[0]
            var=np.sum(unique_beta[1:]**2)
            return mean,var
        else:
            return unique_beta
        
    def variance_contributions(self,unique_beta):

        variance = np.sum(unique_beta[1:]**2)
        multi_index_set = self.unique_basis
        terms, inputs_number=multi_index_set.shape
        variances = np.zeros(inputs_number)
        # take all multi-indices except 0-index
        idx_no_0 = np.delete(multi_index_set, 0, axis=0)
        for nn in range(inputs_number):
            # remove nn-th column
            idx_no_0_nn = np.delete(idx_no_0, nn, axis=1)
            # we want the rows with all indices (except nn) equal to zero
            sum_idx_rows = np.sum(idx_no_0_nn, axis=1)
            zero_rows = np.asarray(np.where(sum_idx_rows == 0)).flatten() + 1
            variance_contribution = np.sum(unique_beta[zero_rows] ** 2, axis=0)

            variances[nn] = variance_contribution

        return variances    
def get_localUQ(reduced_pce, field_x, PolynomialType, n_rvs, lb, ub, sigma_mult=3, n_derivations=0, der_leadvar=0):
    """
    This function calculates the mean, variance, and confidence intervals (lower and upper quantiles) 
    for a given reduced polynomial chaos expansion (PCE) at different points in the field.

    Parameters:
    reduced_pce: Reduced polynomial chaos expansion object.
    field_x: List of points in the field where the calculations will be performed.
    PolynomialType: Type of polynomial used in the PCE.
    n_rvs: Number of random variables.
    lb: Lower bounds for the random variables.
    ub: Upper bounds for the random variables.
    sigma_mult (optional): Multiplier for the standard deviation to calculate the confidence intervals. Default is 3.
    n_derivations (optional): Number of derivatives to calculate. Default is 0.
    der_leadvar (optional): Leading variable for the derivative calculation. Default is 0.

    Returns:
    mean_res: Mean values at each point in the field.
    vartot_res: Total variance at each point in the field.
    lower_quantiles_modes: Lower quantiles (confidence intervals) at each point in the field.
    upper_quantiles_modes: Upper quantiles (confidence intervals) at each point in the field.
    """

    mean_res = []
    vartot_res = []
    lower_quantiles_modes = []
    upper_quantiles_modes = []

    for i in range(len(field_x)):
        x = field_x[i, :]
        mean = np.zeros(n_derivations + 1)
        var = np.zeros(n_derivations + 1)
        variances = np.zeros((n_derivations + 1, n_rvs))
        lq = np.zeros((1 + n_derivations, n_rvs))
        uq = np.zeros((1 + n_derivations, n_rvs))

        for d in range(1 + n_derivations):
            if d == 0:
                coeff = (reduced_pce.eval_coord(x, return_coeff=True))
            else:
                coeff = reduced_pce.derive_coord(x, PolynomialType, lb, ub, der_order=d, der_var=der_leadvar, return_coeff=True)

            mean[d] = coeff[0]
            var[d] = np.sum(coeff[1:] ** 2)
            variances[d, :] = reduced_pce.variance_contributions(coeff)

            for e in range(n_rvs):
                lq[d, e] = mean[d] + sigma_mult * np.sqrt(np.sum(variances[d, :e + 1]))
                uq[d, e] = mean[d] - sigma_mult * np.sqrt(np.sum(variances[d, :e + 1]))

        lower_quantiles_modes.append(lq)
        upper_quantiles_modes.append(uq)
        mean_res.append(mean)
        vartot_res.append(var)

    mean_res = np.array(mean_res)
    vartot_res = np.array(vartot_res)
    lower_quantiles_modes = np.array(lower_quantiles_modes)
    upper_quantiles_modes = np.array(upper_quantiles_modes)

    return mean_res, vartot_res, lower_quantiles_modes, upper_quantiles_modes

