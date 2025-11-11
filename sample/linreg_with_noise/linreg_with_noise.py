#!/usr/bin/env python
"""
Integrated linear regression and model evidence analysis
Combines a custom ODAT-SE solver implementing linear regression with model evidence calculation
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