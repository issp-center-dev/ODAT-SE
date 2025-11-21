import sys, os, argparse
import numpy as np
import odatse
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

def himmelblau(x, y):
    return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2

def make_data(f, N, xrange, yrange, noise = 0.01):
    x = np.random.uniform(*xrange, N)
    y = np.random.uniform(*yrange, N)
    t = (f(x, y) + noise * np.random.randn(N)).astype(np.float32).reshape(-1, 1)
    X = np.stack([x, y], axis=1).astype(np.float32)
    return X, t

def split_data(X, t, n_train, n_test):
    ids = np.random.choice(len(t), len(t), replace=False)
    id_tr = ids[:n_train]
    id_te = ids[n_train:n_train + n_test]
    return X[id_tr], t[id_tr], X[id_te], t[id_te]

def gplearn(Xtr, ttr, seed):
    kernel = ConstantKernel(1.0, (0.1, 1e6)) * RBF(1.0, (0.1, 10.0)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-16, 1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=False, random_state=seed)
    gpr.fit(Xtr, ttr)
    print(f"Hyperparameters: {np.exp(gpr.kernel_.theta)}")
    print(f"Fitted kernel: {gpr.kernel_}")

    def predictor(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ts = gpr.predict(X).astype(np.float32)
        return ts  # (N,)
    return predictor

class Converter:
    def __init__(self, X, t):
        self.scaler_X = StandardScaler()
        self.scaler_t = StandardScaler()

        self.scaler_X.fit(X)
        self.scaler_t.fit(t)

    def convert_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.scaler_X.transform(X).astype(np.float32)

    def convert_t(self, t):
        return self.scaler_t.transform(t).astype(np.float32)

    def revert_t(self, t_scaled):
        if t_scaled.ndim == 1:
            t_scaled = t_scaled.reshape(-1, 1)
        t_proc = self.scaler_t.inverse_transform(t_scaled)
        return t_proc.reshape(-1).astype(np.float32)

    def gen_predictor(self, predictor_raw):
        def predictor(X):
            Xs = self.convert_X(X)
            ts = predictor_raw(Xs)
            return self.revert_t(ts)
        return predictor

class PredictorSolver(odatse.solver.SolverBase):
    def __init__(self, info, predictor):
        super().__init__(info)
        self.__name = "predict"
        self.predictor = predictor

    def evaluate(self, xs, _=()):
        return self.predictor(xs)[0]

def plot_res(Xtr, predictor):
    # Grid over the domain
    x = np.linspace(xrange[0], xrange[1], 251)
    y = np.linspace(yrange[0], yrange[1], 251)
    Xg, Yg = np.meshgrid(x, y)
    XY = np.stack((Xg, Yg), axis=-1).reshape(-1, 2).astype(np.float32)
    
    # Compute maps
    Z_pred = predictor(XY).reshape(Xg.shape)
    Z_true = himmelblau(Xg, Yg).astype(np.float32)
    Z_err = Z_pred - Z_true
    
    # Shared limits for true/pred
    vmin = float(min(Z_true.min(), Z_pred.min()))
    vmax = float(max(Z_true.max(), Z_pred.max()))
    
    # Plots: True | Pred | Error
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax_t, ax_p, ax_e = axes
    
    im_t = ax_t.imshow(Z_true, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax_t.set_title('True')
    ax_t.set_xlabel('x')
    ax_t.set_ylabel('y')
    ax_t.scatter(Xtr[:, 0], Xtr[:, 1], s=8, c='w', edgecolor='k', linewidths=0.3, alpha=0.8)
    plt.colorbar(im_t, ax=ax_t)
    
    im_p = ax_p.imshow(Z_pred, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax_p.set_title('Predicted')
    ax_p.set_xlabel('x')
    ax_p.set_ylabel('y')
    ax_p.scatter(Xtr[:, 0], Xtr[:, 1], s=8, c='w', edgecolor='k', linewidths=0.3, alpha=0.8)
    plt.colorbar(im_p, ax=ax_p)
    
    im_e = ax_e.imshow(Z_err, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='coolwarm', norm=CenteredNorm(vcenter=0.0))
    ax_e.set_title('Error (pred - true)')
    ax_e.set_xlabel('x')
    ax_e.set_ylabel('y')
    ax_e.scatter(Xtr[:, 0], Xtr[:, 1], s=8, c='k', alpha=0.25, linewidths=0)
    plt.colorbar(im_e, ax=ax_e)
    
    # plt.show()
    fig.savefig("res.png")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.toml",
                       help="ODAT-SE input configuration file path")
    parser.add_argument("--logfile", type=str, default=None,
                       help="ODAT-SE run log file (default: odatse_run.log)")
    parser.add_argument("--ntrain", type=int, default=500,
                       help="number of training data points (default: 500)")
    parser.add_argument("--ntest", type=int, default=500,
                       help="number of test data points (default: 500)")
    args = parser.parse_args()
    
    # Initialize ODAT-SE to get output directory
    sys.argv = ["script.py", args.input, "--init"]
    info, run_mode = odatse.initialize()
    output_dir = info.base.get("output_dir", "./output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set log file path
    if args.logfile is None:
        args.logfile = os.path.join(output_dir, "odatse_run.log")
    
    np.random.seed(info.algorithm["seed"])
    xrange, yrange = zip(info.algorithm["param"]["min_list"],info.algorithm["param"]["max_list"])
    
    # Data
    X, t = make_data(himmelblau, args.ntrain + args.ntest, xrange, yrange, 0.01)
    
    # Split
    Xtr, ttr, Xte, _ = split_data(X, t, args.ntrain, args.ntest)
    conv = Converter(Xtr, ttr)
    Xtr_s = conv.convert_X(Xtr)
    ttr_s = conv.convert_t(ttr)
    
    predictor_raw = gplearn(Xtr_s, ttr_s, info.algorithm["seed"])
    predictor = conv.gen_predictor(predictor_raw)
    
    t_pred = predictor(Xte)
    t_true = himmelblau(Xte[:, 0], Xte[:, 1]).astype(np.float32)
    mse = float(np.mean((t_pred - t_true) ** 2))
    var_true = float(np.var(t_true))
    nmse = mse / (var_true + 1e-12)
    print(f"MSE (n_test={len(t_true)}): {mse:.6e}")
    print(f"NMSE: {nmse:.6e}")

    # Plot output
    plot_res(Xtr, predictor)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Run ODAT-SE
    with open(args.logfile, "w") as f:
        sys.stdout = f
        sys.stderr = f
        
        solver = PredictorSolver(info, predictor=predictor)
        runner = odatse.Runner(solver, info)
        alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])
        alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
        result = alg.main()
        
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    print(f"Best solution: x^* = {result['x']}")
    print(f"Surrogate f(x^*) = {result['fx']}")
    print(f"True f(x^*) = {himmelblau(*tuple(result['x']))}")