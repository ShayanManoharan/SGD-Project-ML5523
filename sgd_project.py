import os
import math
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Project constants
# =========================
D = 4                   # feature dimension
W_DIM = D + 1           # parameter dimension including bias
SIGMAS = [0.2, 0.4]
N_VALUES = [50, 100, 500, 1000]
NUM_TRIALS = 30
TEST_SIZE = 400

# Create output folder
os.makedirs("results", exist_ok=True)


# =========================
# Utility / projection
# =========================
def project_to_unit_ball(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the unit ball:
      Pi(v) = v            if ||v|| <= 1
              v / ||v||    otherwise
    """
    norm = np.linalg.norm(v)
    if norm <= 1.0:
        return v.copy()
    return v / norm


def augment_x(x: np.ndarray) -> np.ndarray:
    """Append bias term 1 to x."""
    return np.append(x, 1.0)


# =========================
# Data generation
# =========================
def generate_example(sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """
    Generate one example (x, y) exactly as the project specifies.
    """
    if rng.random() < 0.5:
        y = -1
        mu = -0.25 * np.ones(D)
    else:
        y = 1
        mu = 0.25 * np.ones(D)

    u = rng.normal(loc=mu, scale=sigma, size=D)
    x = project_to_unit_ball(u)   # project onto X
    return x, y


def generate_dataset(size: int, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of the given size.
    Returns:
      X shape: (size, 4)
      y shape: (size,)
    """
    X = np.zeros((size, D))
    y = np.zeros(size, dtype=int)

    for i in range(size):
        xi, yi = generate_example(sigma, rng)
        X[i] = xi
        y[i] = yi

    return X, y


# =========================
# Logistic loss / gradient
# =========================
def logistic_loss_single(w: np.ndarray, x: np.ndarray, y: int) -> float:
    """
    Logistic loss:
      log(1 + exp(-y <w, x_tilde>))
    """
    x_tilde = augment_x(x)
    yz = y * np.dot(w, x_tilde)

    # numerically stable version
    if yz >= 0:
        return math.log1p(math.exp(-yz))
    return -yz + math.log1p(math.exp(yz))


def logistic_grad_single(w: np.ndarray, x: np.ndarray, y: int) -> np.ndarray:
    """
    Gradient of logistic loss wrt w:
      -(y x_tilde) / (1 + exp(y <w, x_tilde>))
    """
    x_tilde = augment_x(x)
    yz = y * np.dot(w, x_tilde)

    # stable sigmoid-style computation
    if yz >= 0:
        coeff = 1.0 / (1.0 + math.exp(yz))
    else:
        exp_yz = math.exp(yz)
        coeff = 1.0 / (1.0 + exp_yz)

    return -(y * x_tilde) * coeff


def average_logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    losses = [logistic_loss_single(w, X[i], int(y[i])) for i in range(len(y))]
    return float(np.mean(losses))


def classification_error(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    preds = []
    for i in range(len(y)):
        score = np.dot(w, augment_x(X[i]))
        pred = 1 if score >= 0 else -1
        preds.append(pred)

    preds = np.array(preds)
    return float(np.mean(preds != y))


# =========================
# SGD
# =========================
def sgd_train(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Projected SGD for logistic regression.
    Uses eta_t = 1 / sqrt(t)
    Returns the average iterate.
    """
    n = len(y_train)
    w = np.zeros(W_DIM)
    w_sum = np.zeros(W_DIM)

    for t in range(1, n + 1):
        eta_t = 1.0 / math.sqrt(t)
        g_t = logistic_grad_single(w, X_train[t - 1], int(y_train[t - 1]))
        w = project_to_unit_ball(w - eta_t * g_t)   # project onto C
        w_sum += w

    return w_sum / n


# =========================
# Experiment runner
# =========================
def run_one_setting(sigma: float, n: int, test_X: np.ndarray, test_y: np.ndarray, base_seed: int = 12345):
    """
    Run 30 trials for one (sigma, n) setting.
    Returns summary stats needed for the table and plots.
    """
    risk_values = []
    err_values = []

    for trial in range(NUM_TRIALS):
        rng = np.random.default_rng(base_seed + int(1000 * sigma) + 10000 * n + trial)
        train_X, train_y = generate_dataset(n, sigma, rng)
        w_hat = sgd_train(train_X, train_y)

        risk = average_logistic_loss(w_hat, test_X, test_y)
        err = classification_error(w_hat, test_X, test_y)

        risk_values.append(risk)
        err_values.append(err)

    risk_values = np.array(risk_values)
    err_values = np.array(err_values)

    risk_mean = float(np.mean(risk_values))
    risk_std = float(np.std(risk_values, ddof=0))
    risk_min = float(np.min(risk_values))
    excess_risk = risk_mean - risk_min

    err_mean = float(np.mean(err_values))
    err_std = float(np.std(err_values, ddof=0))

    return {
        "sigma": sigma,
        "n": n,
        "N": TEST_SIZE,
        "trials": NUM_TRIALS,
        "risk_mean": risk_mean,
        "risk_std": risk_std,
        "risk_min": risk_min,
        "excess_risk": excess_risk,
        "err_mean": err_mean,
        "err_std": err_std,
    }


def print_table(results: list[dict]) -> None:
    print("\nRESULTS TABLE")
    print("-" * 110)
    header = (
        f"{'sigma':<8}{'n':<8}{'N':<8}{'trials':<10}"
        f"{'loss mean':<15}{'loss std':<15}{'loss min':<15}"
        f"{'excess risk':<15}{'err mean':<12}{'err std':<12}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        print(
            f"{r['sigma']:<8}"
            f"{r['n']:<8}"
            f"{r['N']:<8}"
            f"{r['trials']:<10}"
            f"{r['risk_mean']:<15.6f}"
            f"{r['risk_std']:<15.6f}"
            f"{r['risk_min']:<15.6f}"
            f"{r['excess_risk']:<15.6f}"
            f"{r['err_mean']:<12.6f}"
            f"{r['err_std']:<12.6f}"
        )

    print("-" * 110)


# =========================
# Plotting
# =========================
def make_plots(results: list[dict]) -> None:
    # Organize by sigma
    for_plot = {}
    for sigma in SIGMAS:
        rows = [r for r in results if r["sigma"] == sigma]
        rows.sort(key=lambda r: r["n"])
        for_plot[sigma] = rows

    # -------- Plot 1: Excess Risk --------
    plt.figure(figsize=(8, 5))
    for sigma in SIGMAS:
        rows = for_plot[sigma]
        x = [r["n"] for r in rows]
        y = [r["excess_risk"] for r in rows]
        yerr = [r["risk_std"] for r in rows]
        plt.errorbar(x, y, yerr=yerr, marker='o', capsize=4, label=f"sigma = {sigma}")

    plt.xlabel("Training set size n")
    plt.ylabel("Estimated excess risk")
    plt.title("Estimated Excess Risk vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/excess_risk_plot.png", dpi=300)

    # -------- Plot 2: Classification Error --------
    plt.figure(figsize=(8, 5))
    for sigma in SIGMAS:
        rows = for_plot[sigma]
        x = [r["n"] for r in rows]
        y = [r["err_mean"] for r in rows]
        yerr = [r["err_std"] for r in rows]
        plt.errorbar(x, y, yerr=yerr, marker='o', capsize=4, label=f"sigma = {sigma}")

    plt.xlabel("Training set size n")
    plt.ylabel("Estimated classification error")
    plt.title("Estimated Classification Error vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/classification_error_plot.png", dpi=300)

    # 🔥 Show BOTH plots at the end
    plt.show()
    

# =========================
# Main
# =========================
def main():
    all_results = []

    for sigma in SIGMAS:
        # One fixed test set per sigma, exactly as the project says
        test_rng = np.random.default_rng(999 + int(1000 * sigma))
        test_X, test_y = generate_dataset(TEST_SIZE, sigma, test_rng)

        for n in N_VALUES:
            print(f"Running sigma={sigma}, n={n} ...")
            result = run_one_setting(sigma, n, test_X, test_y)
            all_results.append(result)

    # Sort nicely for printing
    all_results.sort(key=lambda r: (r["sigma"], r["n"]))

    print_table(all_results)
    make_plots(all_results)

    print("\nSaved plots to:")
    print("  results/excess_risk_plot.png")
    print("  results/classification_error_plot.png")


if __name__ == "__main__":
    main()