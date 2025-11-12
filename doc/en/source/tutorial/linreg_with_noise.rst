Linear Regression and Noise Estimation with ODAT-SE
===================================================

Introduction
------------

This tutorial shows how to use the ODAT-SE framework to perform linear regression analysis and estimate noise levels in experimental data through Bayesian inference and partition function methods. The goal of this tutorial is to demonstrate the automatic determination of optimal noise levels in experimental data by maximizing the model evidence.

Theoretical Foundation
----------------------

Bayesian Framework for Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We consider experimental measurement data :math:`D \equiv \{(x_{\mu,i},y_{\mu,i})\}_{\mu,i}`, where:

- :math:`i` indexes each of the :math:`n` data points for a given dataset (such that :math:`i` runs from 1 to :math:`n`),
- :math:`\mu` indexes each of the :math:`N` datasets (such that :math:`\mu` runs from 1 to :math:`N`), and that the datasets are explained by the same model (e.g. data from different trials, data from different XRD spots, etc.),
- :math:`(x_{\mu,i},y_{\mu,i})` represents the :math:`i` -th measurement in the :math:`\mu` -th dataset.

Our goal is to estimate:

1. the parameters :math:`\theta` in some model that maps state quantities :math:`x` to observations :math:`y`, and 
2. the noise level :math:`\sigma` in the experimental data.

In this tutorial, we perform a linear regression on the observed data to obtain a linear model for the former, and use maximum likelihood estimation on the model evidence to determine the latter.

Defining the Likelihood Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To measure the distance between calculated and observed measurements, we use the sum of squared residuals:

.. math::

	f(\theta; D) = \sum_{\mu=1}^{N} w_\mu \sum_{i=1}^n \left(y_{\mu,i} - y_{\mu,i}^{(\text{cal})}(x_{\mu,i}; \theta)\right)^2

Assuming Gaussian noise with standard deviation :math:`\sigma`, the likelihood function (which is the likelihood of the observation :math:`y_{\mu,i}` under a given model) for a single data point is:

.. math::

	P(x_{\mu,i},y_{\mu,i}|\theta; \beta) = \sqrt{\frac{\beta w_\mu}{\pi}} \exp\left(-\beta w_\mu \left(y_{\mu,i} - y_{\mu,i}^{(\text{cal})}(x_{\mu,i}; \theta)\right)^2\right)

where :math:`\beta = (2\sigma^2)^{-1}` is the inverse temperature parameter and parametrizes the model together with the predicted state quantity :math:`x`.

The total likelihood for all measurements is the product of their likelihoods under a given model:

.. math::

	P(D|\theta; \beta) = \prod_{\mu=1}^{N} \prod_{i=1}^n P(x_{\mu,i},y_{\mu,i}|\theta; \beta) = \left(\prod_{\mu=1}^{N} \left(\frac{\beta w_\mu}{\pi}\right)^{n/2}\right) \exp\left(-\beta f(\theta; D)\right)

Defining the Model Evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We assume a uniform prior distribution over the parameter space :math:`\Omega`:

.. math::

	P(\theta) = \frac{1}{V_\Omega}

where :math:`V_\Omega = \int_\Omega dx` is the normalization volume. The model evidence (which corresponds to the marginal likelihood) is obtained by marginalizing over all possible parameter values:

.. math::

	P(D; \beta) = \int_\Omega P(D|\theta; \beta) P(\theta) d\theta=\frac{1}{V_\Omega} \left(\prod_{\mu=1}^{N} \left(\frac{\beta w_\mu}{\pi}\right)^{n/2}\right) \int_\Omega \exp\left(-\beta f(\theta; D)\right) d\theta

We can define the partition function to be:

.. math::

	Z(D; \beta) \equiv \int_\Omega \exp\left(-\beta f(\theta; D)\right) d\theta

This means that we can write the model evidence as:

.. math::

	P(D; \beta) = \frac{Z(D; \beta)}{V_\Omega} \prod_{\mu=1}^{N} \left(\frac{\beta w_\mu}{\pi}\right)^{n/2}

Analytical Solution for Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We assume that there is a single dataset. For the simple linear model :math:`y = ax`, the objective function (sum of squared residuals) is:

.. math::

	f(a; D) = \sum_{i=1}^n (y_i - ax_i)^2

If we expand the objective function, we obtain:

.. math::

	f(a; D) = \sum_{i=1}^n (y_i - ax_i)^2 = a^2 \left(\sum_{i=1}^n x_i^2\right) - 2a \left(\sum_{i=1}^n x_i y_i\right) + \sum_{i=1}^n y_i^2

We define the following intermediate quantities:

.. math::

	A \equiv \sum_{i=1}^n x_i^2, \quad B \equiv \sum_{i=1}^n x_i y_i, \quad C \equiv \sum_{i=1}^n y_i^2

Then, by completing the square, we can rewrite the objective as:

.. math::

	f(a; D) = A\left(a - \frac{B}{A}\right)^2 + C - \frac{B^2}{A} \equiv A(a - a^*)^2 + f^*

We can identify the following quantities:

- :math:`a^* = \frac{B}{A}` is the best-fitting parameter, and
- :math:`f^* = C - \frac{B^2}{A}` is the minimum value of the cost function :math:`f` at :math:`f(a^*; D)`.

Calculating the Model Evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate the model evidence, we must first compute the partition function :math:`Z(D|\beta)`. We first consider the partition function using the objective function :math:`f(a; D)`, where the model parameter :math:`a` has domain :math:`[-L/2, L/2]`.

.. math::

	Z(D; \beta) = \int_{-L/2}^{L/2} \exp(-\beta f(a; D)) da = \int_{-L/2}^{L/2} \exp(-\beta[A(a - a^*)^2 + f^*]) da

We perform a change of variables :math:`t = a - a^*` and move the constant factor out of the integral before evaluating:

.. math::

	Z(D; \beta) = e^{-\beta f^*} \int_{-L/2-a^*}^{L/2-a^*} \exp(-\beta A t^2) dt=e^{-\beta f^*} \frac{1}{\sqrt{\beta A}} \frac{\sqrt{\pi}}{2} \left[\text{erf}\left(\sqrt{\beta A}\left(\frac{L}{2} - a^*\right)\right) - \text{erf}\left(\sqrt{\beta A}\left(-\frac{L}{2} - a^*\right)\right)\right]

In the above, we used the definition of the error function as an integral:

.. math::

	\int_0^y e^{-\gamma t^2} dt = \frac{1}{\sqrt{\gamma}} \frac{\sqrt{\pi}}{2} \text{erf}(\sqrt{\gamma} y)

Thus, the model evidence can be written out:

.. math::

	P(D; \beta) = \frac{1}{L} \left(\frac{\beta}{\pi}\right)^{(n-1)/2} \frac{e^{-\beta f^*}}{2\sqrt{A}} \left[\text{erf}\left(\sqrt{\beta A}\left(\frac{L}{2} - a^*\right)\right) - \text{erf}\left(\sqrt{\beta A}\left(-\frac{L}{2} - a^*\right)\right)\right]

The error function approaches 1 and -1 as its argument tends to :math:`\infty` and :math:`-\infty` respectively. When :math:`L` is taken to be sufficiently large, we obtain the following result for the model evidence:

.. math::

	P(D; \beta) \approx \frac{1}{L} \left(\frac{\beta}{\pi}\right)^{(n-1)/2} \frac{e^{-\beta f^*}}{\sqrt{A}}

Estimating the Noise Level
^^^^^^^^^^^^^^^^^^^^^^^^^^

The principle of maximum likelihood provides a guide for how we might be able to obtain an estimate of the noise level. Maximizing the logarithm of the model evidence with respect to :math:`\beta` (in the large :math:`L` limit), we obtain:

.. math::

	\frac{d}{d\beta} \log P(D; \beta) = \frac{n-1}{2\beta} - f^* = 0

The optimal inverse temperature and associated noise level can then be determined:

.. math::

	\beta_{\text{opt}} = \frac{n-1}{2f^*}\implies\sigma_{\text{opt}} = \frac{1}{\sqrt{2\beta_{\text{opt}}}} = \sqrt{\frac{f^*}{n-1}}

This provides a principled, data-driven estimate of the noise level.

ODAT-SE Implementation
----------------------

Code Implementation
~~~~~~~~~~~~~~~~~~~

For this tutorial, files can be located in ``sample/linreg_with_noise`` relative to the root of the repository. The main script is named ``linreg_with_noise.py``.

User-defined Target Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ODAT-SE framework allows users to define custom solvers by inheriting from the ``SolverBase`` class. Here, we demonstrate how to create a linear regression solver.

.. code:: python

	#!/usr/bin/env python
	"""
	Integrated linear regression and model evidence analysis
	Combines custom ODAT-SE solver implementing linear regression with model evidence calculation
	"""
	
	import odatse
	import sys, os, argparse
	import numpy as np
	from matplotlib import pyplot as plt
	from odatse.algorithm import choose_algorithm
	sys.path.append("../../script")
	from plt_model_evidence import load_data, calc_log_pdb, print_log_pdb, plot_log_pdb
	
	class LinearRegression(odatse.solver.SolverBase):
		"""Linear regression solver class"""
		
		def __init__(self, info):
			super().__init__(info)
			data_file = info.solver["reference"]["path"]
			data = np.loadtxt(data_file, unpack=True)
			
			self.xdata = data[0]
			self.ydata = data[1]
			self.n = len(self.ydata)
	
		def evaluate(self, xs, args, nprocs=1, nthreads=1):
			loss = np.sum((xs*self.xdata - self.ydata)**2)
			return loss

The target function is defined in the ``evaluate`` function. Here, we use the quadratic loss function representing the sum of squared residuals.

Model Evidence Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^

For the calculation of the model evidence we import functions from the post-processing tool ``plt_model_evidence.py`` (for details, see :doc:`../post/index`). To plot the linear fit together with noise bands, we use the following function:

.. code:: python

	def plot_linear_fit_with_noise(xdata, ydata, a, noise_level, beta_opt, output_file):
		fig, ax = plt.subplots(figsize=(10, 6))
		
		# Plot original data points and fitted line
		ax.scatter(xdata, ydata, s=50, alpha=0.7, label='Data', color='blue')
		x_fit = np.linspace(xdata.min(), xdata.max(), 100)
		y_fit = a * x_fit
		ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: y = {a:.4f}x')
		
		# Add noise bands (\pm1\sigma, \pm2\sigma)
		ax.fill_between(x_fit, y_fit - noise_level, y_fit + noise_level, 
						alpha=0.3, color='red', label=f'$\\pm1\\sigma$ noise')
		ax.fill_between(x_fit, y_fit - 2*noise_level, y_fit + 2*noise_level, 
						alpha=0.15, color='red', label=f'$\\pm2\\sigma$ noise')
		
		ax.set_xlabel('x', fontsize=12)
		ax.set_ylabel('y', fontsize=12)
		ax.set_title(f'Linear Regression with Optimal Noise Level\n' + 
					f'$a$ = {a:.4f}, $\\beta_{{opt}}$ = {beta_opt:.2e}, $\\sigma$ = {noise_level:.4f}', fontsize=14)
		ax.legend(loc='best')
		ax.grid(True, alpha=0.3)
		
		fig.savefig(output_file, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Linear fit plot saved to: {output_file}")

The main function accepts a TOML file as input and optionally a log file. Other arguments include the normalization factor of the prior and arguments related to plot window selection when visualizing the model evidence plot.

.. code:: python

	if __name__ == "__main__":
		parser = argparse.ArgumentParser(description='Integrated linear regression and model evidence analysis')
		# Basic parameters
		parser.add_argument('--input', type=str, default='input.toml',
						   help='ODAT-SE input configuration file path')
		parser.add_argument('--logfile', type=str, default=None,
						   help='ODAT-SE run log file (default: output_dir/odatse_run.log)')
		# Model evidence calculation parameters
		parser.add_argument('-V', '--volume', type=float, default=1.0,
						   help='Normalization factor of prior probability distribution (default is 1.0)')
		# Auto-focus parameters
		parser.add_argument('--auto-focus', action='store_true',
						   help='Auto-focus on maximum model evidence region')
		parser.add_argument('--focus-factor', type=float, default=0.5,
						   help='Auto-focus tightness (0-1, smaller is tighter, default: 0.5)')
		args = parser.parse_args()
		
		# Run main program
		print("="*60)
		print("Starting Linear Regression and Model Evidence Analysis")
		print("="*60)
		
		# Step 1: Run ODAT-SE
		print("\nStep 1: Running ODAT-SE linear regression...")
		
		sys.argv = ["script.py", args.input, "--init"]
		
		original_stdout = sys.stdout
		original_stderr = sys.stderr
		
		# Initialize ODAT-SE to get output directory
		info, run_mode = odatse.initialize()
		output_dir = info.base.get("output_dir", "./output")
		
		# Create output directory if it doesn't exist
		os.makedirs(output_dir, exist_ok=True)
		
		# Set log file path
		if args.logfile is None:
			args.logfile = os.path.join(output_dir, "odatse_run.log")
		
		# Run ODAT-SE
		with open(args.logfile, "w") as f:
			sys.stdout = f
			sys.stderr = f
			
			solver = LinearRegression(info)
			runner = odatse.Runner(solver, info)
			alg_module = choose_algorithm(info.algorithm["name"])
			alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
			result = alg.main()
			
			sys.stdout = original_stdout
			sys.stderr = original_stderr
		print("ODAT-SE run completed")
		print(f"Output directory: {output_dir}")
		
		# Get fitting parameters
		a = result['x'][0]
		xdata = solver.xdata
		ydata = solver.ydata
		n_data = solver.n
		print(f"\nFitting results:")
		print(f"  Slope a = {a:.6f}")
		print(f"  Number of data points n = {n_data}")
		
		# Step 2: Load fx.txt data and calculate model evidence
		print("\nStep 2: Calculating model evidence...")
		fx_file = os.path.join(output_dir, "fx.txt")
		if not os.path.exists(fx_file):
			raise FileNotFoundError(f"Error: Cannot find file {fx_file}")
		beta, logz = load_data(fx_file)
		log_pdb = calc_log_pdb(beta, logz, np.asarray([n_data], dtype=np.int64), np.asarray([1], dtype=np.float64), args.volume)
		
		# Save model evidence data
		evidence_file = os.path.join(output_dir, "model_evidence.txt")
		print_log_pdb(evidence_file, beta, log_pdb)
		
		# Step 3: Find optimal beta value
		print("\nStep 3: Finding optimal beta value...")
		valid_mask = np.isfinite(log_pdb) & np.isfinite(beta)
		if not np.any(valid_mask):
			raise ValueError("No valid data points")
		max_idx = np.argmax(log_pdb[valid_mask])
		beta_opt = beta[valid_mask][max_idx]
		log_pdb_max = log_pdb[valid_mask][max_idx]
		print(f"\nOptimal parameters:")
		print(f"  beta_opt = {beta_opt:.6e}")
		print(f"  log P(D;beta_opt) = {log_pdb_max:.6f}")
		
		# Step 4: Calculate noise level
		noise_level = 1.0 / np.sqrt(2.0 * beta_opt)
		print(f"\nNoise level:")
		print(f"  std = 1/sqrt(2*beta_opt) = {noise_level:.6f}")
		
		# Calculate R^2 value
		y_pred = a * xdata
		ss_res = np.sum((ydata - y_pred)**2)
		ss_tot = np.sum((ydata - np.mean(ydata))**2)
		r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
		print(f"\nFitting quality:")
		print(f"  R^2 = {r_squared:.6f}")
		print(f"  Residual sum of squares = {ss_res:.6f}")
		
		# Step 5: Generate plots
		print("\nStep 5: Generating plots...")
		
		# Plot model evidence and linear fit with noise bands
		evidence_plot = os.path.join(output_dir, "model_evidence.png")
		plot_log_pdb(evidence_plot, beta, log_pdb, None, args.auto_focus, args.focus_factor)
		fit_plot = os.path.join(output_dir, "linear_fit_with_noise.png")
		plot_linear_fit_with_noise(xdata, ydata, a, noise_level, beta_opt, output_file=fit_plot)
		
		# Save results to file
		print("\nStep 6: Saving results...")
		results_file = os.path.join(output_dir, "analysis_results.txt")
		with open(results_file, "w") as f:
			f.write("Linear regression and model evidence analysis results\n")
			f.write("="*50 + "\n\n")
			f.write(f"Fitting parameters:\n")
			f.write(f"  Slope a = {a:.6f}\n")
			f.write(f"  Number of data points n = {n_data}\n\n")
			f.write(f"Optimal parameters:\n")
			f.write(f"  beta_opt = {beta_opt:.6e}\n")
			f.write(f"  log P(D;beta_opt) = {log_pdb_max:.6f}\n\n")
			f.write(f"Noise level:\n")
			f.write(f"  std = {noise_level:.6f}\n\n")
			f.write(f"Fitting quality:\n")
			f.write(f"  R^2 = {r_squared:.6f}\n")
			f.write(f"  Residual sum of squares = {ss_res:.6f}\n")
		print(f"Results saved to: {results_file}")
		print("\n" + "="*60)
		print("Analysis completed!")
		print(f"All output files are saved in: {output_dir}")
		print("="*60)

Usage Examples
--------------

Example 1: Six-Point Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Prepare Input Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a data file ``data.txt`` with two columns (x and y values):

.. code:: text

	1.0 1.1
	2.0 1.9
	3.0 3.1
	4.0 4.2
	5.0 4.8
	6.0 6.1

Step 2: Create the ODAT-SE Input File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``input.toml`` according to the ODATSE configuration (model evidence is available for Replica Exchange MC and Population Annealing MC algorithms):

.. code:: toml

	[base]
	dimension = 1
	output_dir = "./output_data"

	[solver]
	name = "user"

	[solver.reference]
	path = "./data.txt"

	[algorithm]
	name = "pamc"
	seed = 12345
	label_list = ["a"]

	[algorithm.param]
	min_list = [-20.0]
	max_list = [20.0]
	step_list = [0.05]

	[algorithm.pamc]
	Tnum = 100
	bmin = 1e-5
	bmax = 1e2
	Tlogspace = true
	numsteps_annealing = 100
	nreplica_per_proc = 10

Step 3: Run the Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

Execute the script, passing ``input.toml`` as the input argument:

.. code:: bash

	python linreg_with_noise.py input.toml

To use the auto-focus feature that determines a suitable plot window that includes the maximum of the model evidence plot according to a focus factor from 0 to 1, we can supply the optional ``--auto-focus`` and ``--focus-factor`` (default: 0.5) parameters:

.. code:: bash

	python linreg_with_noise.py input.toml --auto-focus --focus-factor 0.3

Theoretical Calculation
^^^^^^^^^^^^^^^^^^^^^^^

Using the data file described previously, we consider six points:

.. math::

	(x_1, y_1) = (1, 1.1), \quad (x_2, y_2) = (2, 1.9), \quad (x_3, y_3) = (3, 3.1), \quad (x_4, y_4) = (4, 4.2), \quad (x_5, y_5) = (5, 4.8), \quad (x_6, y_6) = (6, 6.1)

By applying the formulas presented earlier, we can obtain theoretical values for :math:`a^*`, :math:`f^*`, and :math:`\sigma_{\text{opt}}` (through :math:`\beta_{\text{opt}}`):

.. math::

	a^* = \frac{\sum_{i=1}^6 x_i y_i}{\sum_{i=1}^6 x_i^2} = \frac{91.6}{91} \approx 1.00659, \quad f^* = \sum_{i=1}^6 y_i^2 - \frac{(\sum_{i=1}^6 x_i y_i)^2}{\sum_{i=1}^6 x_i^2} = 92.32 - \frac{(92.32)^2}{91} \approx 0.11604

.. math::

	\beta_{\text{opt}} = \frac{n-1}{2f^*} \approx 21.544 \implies \sigma_{\text{opt}} = \frac{1}{\sqrt{2\beta_{\text{opt}}}} \approx 0.15234

Numerical Results
^^^^^^^^^^^^^^^^^

The PAMC algorithm yields:

- :math:`a^* = 1.0066` (numerical)
- :math:`\beta_{\text{opt}} = 23.101` (numerical)
- :math:`\sigma_{\text{opt}} = 0.14712` (from :math:`\beta_{\text{opt}}`)

Factors that cause the slight difference in :math:`\beta` values include the use of a discretized temperature range in the annealing schedule and statistical fluctuations in the partition function estimation due to finite sampling in the Monte Carlo method.

.. figure:: ../../../common/img/linreg_with_noise/example_fit.png
	:name: example_fit
	:width: 55%
	
	Figure 1: Linear regression plot for the dataset used in this example.

The estimated noise level :math:`\sigma \approx 0.1` is reasonable given the deviations of :math:`y_i` from the fitted line.

Example 2: Artificial Data with Different Noise Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the robustness of the noise estimation method by maximizing model evidence using data at three different noise levels.

Step 1: Prepare Input Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a Python script that generates some linear data combined with Gaussian noise at various noise levels.

.. code:: python

	import numpy as np
	import matplotlib.pyplot as plt

	np.random.seed(42)

	n_samples = 50
	x = np.linspace(0, 10, n_samples)

	a = 2.5
	y_true = a * x

	noise_levels = [1, 3, 5]

	file_names = []
	for noise in noise_levels:
		y_noisy = y_true + np.random.normal(0, noise, size=n_samples)
		data = np.column_stack((x, y_noisy))
		file_name = f"linear_data_gaussian_noise{noise}.txt"
		np.savetxt(file_name, data, comments='', fmt="%.6f")
		file_names.append(file_name)
		plt.scatter(x, y_noisy, label=f'Noise level: {noise}')

	plt.plot(x, y_true, color='red', label='True line', linewidth=2)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Linear Data with Gaussian Noise')
	plt.legend()
	plt.savefig("linear_data_gaussian_noise.png")

The datasets generated by this script are plotted below.

.. figure:: ../../../common/img/linreg_with_noise/linear_data_gaussian_noise.png
	:name: linear_data_gaussian_noise
	:width: 50%
	
	Figure 2: Artificial data generated from the function :math:`y=2.5x` with Gaussian noise at levels :math:`\sigma^*=1,3,5`.

Step 2: Create the ODAT-SE Input File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use the same input file as in the previous example, replacing the value of ``output_dir`` under ``[base]`` and ``path`` under ``[solver.reference]`` to the appropriate paths to the output directory and input file (whose filenames are of the form ``data_noise#.txt``, where ``#`` denotes the noise level as taken from the above script for generating noisy input data).

Step 3: Run the Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

As in the previous example, we then run the script for each of the three datasets, specifying ``input.toml`` as the input argument (after making the appropriate changes to the input file):

.. code:: bash

	python linreg_with_noise.py input.toml

With the auto-focus feature, the invocation could look like:

.. code:: bash

	python linreg_with_noise.py input.toml --auto-focus --focus-factor 0.3

Numerical Results
^^^^^^^^^^^^^^^^^

After performing a linear regression and noise estimation on each of the three datasets, we obtain the following results:

.. raw:: html

	<style>
	table {
		width: 40%;
		margin-left: auto;
		margin-right: auto;
		border-collapse: collapse;
	}
	thead {
		border-bottom: 2px solid black;
	}
	</style>
	<table>
		<colgroup>
			<col>
			<col style="border-right: 2px solid black;">
			<col>
			<col>
		</colgroup>
		<thead>
			<tr>
				<td style="text-align: center;">Slope</td>
				<td style="text-align: center;">Noise level</td>
				<td style="text-align: center;">Fitted slope</td>
				<td style="text-align: center;">Estimated noise level</td>
			</tr>
		</thead>
		<tbody>
			<tr>
				<td style="text-align: center;">$$a^*=2.5$$</td>
				<td style="text-align: center;">$$\sigma^*=1$$</td>
				<td style="text-align: center;">$$a=2.4516$$</td>
				<td style="text-align: center;">$$\sigma=0.8820$$</td>
			</tr>
			<tr>
				<td style="text-align: center;">$$a^*=2.5$$</td>
				<td style="text-align: center;">$$\sigma^*=3$$</td>
				<td style="text-align: center;">$$a=2.4844$$</td>
				<td style="text-align: center;">$$\sigma=2.5412$$</td>
			</tr>
			<tr>
				<td style="text-align: center;">$$a^*=2.5$$</td>
				<td style="text-align: center;">$$\sigma^*=5$$</td>
				<td style="text-align: center;">$$a=2.4552$$</td>
				<td style="text-align: center;">$$\sigma=4.8738$$</td>
			</tr>
		</tbody>
	</table>

.. raw:: latex

	\begin{tabular}{cc|cc}
	Slope & Noise level & Fitted slope & Estimated noise level \\
	\hline
	$a^*=2.5$ & $\sigma^*=1$ & $a=2.4516$ & $\sigma=0.8820$ \\
	$a^*=2.5$ & $\sigma^*=3$ & $a=2.4844$ & $\sigma=2.5412$ \\
	$a^*=2.5$ & $\sigma^*=5$ & $a=2.4552$ & $\sigma=4.8738$ \\
	\end{tabular}

The regression plot and model evidence plot for the case where the model has noise level :math:`\sigma=5` are presented below.

.. figure:: ../../../common/img/linreg_with_noise/gaussian_noise5.png
	:name: gaussian_noise5
	
	Figure 3: Linear regression plot and model evidence plot for the dataset at noise level :math:`\sigma^*=5`.