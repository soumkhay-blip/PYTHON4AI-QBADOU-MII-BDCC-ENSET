"""
Réseau de neurones XOR — NumPy only
Architecture : 2 -> 4 -> 1  (Sigmoid / Sigmoid)
Algorithme   : SGD avec rétropropagation
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Données ─────────────────────────────────────────────────────────────────
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0],    [1],    [1],    [0]],    dtype=float)


# ── Fonctions d'activation ──────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


# ── Paramètres ──────────────────────────────────────────────────────────────
def init_weights(n_input=2, n_hidden=4, n_output=1):
    W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((n_output, 1))
    return W1, b1, W2, b2


# ── Forward ─────────────────────────────────────────────────────────────────
def forward(X, W1, b1, W2, b2):
    Xt = X.T
    Z1 = W1 @ Xt + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return A2, (Z1, A1, Z2, A2, Xt)


# ── Perte ────────────────────────────────────────────────────────────────────
def compute_loss(A2, y):
    eps = 1e-8
    yT = y.T
    return -np.mean(yT * np.log(A2 + eps) + (1 - yT) * np.log(1 - A2 + eps))


# ── Backward ─────────────────────────────────────────────────────────────────
def backward(A2, y, cache, W2):
    Z1, A1, _, _, Xt = cache
    m = y.shape[0]
    yT = y.T
    dZ2 = A2 - yT
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.mean(dZ2, axis=1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * sigmoid_prime(Z1)
    dW1 = (dZ1 @ Xt.T) / m
    db1 = np.mean(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


# ── Entraînement ─────────────────────────────────────────────────────────────
def train(X, y, n_hidden=4, lr=1.0, epochs=10000, print_every=2000):
    W1, b1, W2, b2 = init_weights(n_hidden=n_hidden)
    history = []

    for epoch in range(epochs + 1):
        A2, cache = forward(X, W1, b1, W2, b2)
        loss = compute_loss(A2, y)
        history.append(loss)
        dW1, db1g, dW2, db2g = backward(A2, y, cache, W2)
        W1 -= lr * dW1
        b1 -= lr * db1g
        W2 -= lr * dW2
        b2 -= lr * db2g

        if epoch % print_every == 0:
            preds = (A2.T > 0.5).astype(int).flatten()
            acc = np.mean(preds == y.flatten().astype(int)) * 100
            print(f"Époque {epoch:6d} | Loss={loss:.6f} | Précision={acc:.1f}%")

    return W1, b1, W2, b2, history


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_all(history, W1, b1, W2, b2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Courbe d'apprentissage
    axes[0].plot(history, color='steelblue', linewidth=1.5)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Courbe d'apprentissage")
    axes[0].grid(alpha=0.3)

    # Frontière de décision
    res = 300
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, res),
                         np.linspace(-0.5, 1.5, res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    A2_grid, _ = forward(grid, W1, b1, W2, b2)
    Z = A2_grid.reshape(res, res)

    axes[1].contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    colors = ['red' if yi == 0 else 'green' for yi in y.flatten()]
    axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', zorder=5)
    for xi, yi in zip(X, y):
        axes[1].annotate(f"({xi[0]:.0f},{xi[1]:.0f})→{yi[0]:.0f}",
                         (xi[0] + 0.05, xi[1] + 0.05), fontsize=9)
    axes[1].set_title("Frontière de décision")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    plt.suptitle("Réseau XOR — NumPy", fontsize=14)
    plt.tight_layout()
    plt.savefig("xor_results.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Réseau de neurones XOR — NumPy")
    print("=" * 50)

    W1, b1, W2, b2, history = train(X, y, n_hidden=4, lr=1.0, epochs=10000)

    print("\nRésultats finaux :")
    print(f"{'x1':>4} {'x2':>4} {'Attendu':>9} {'Prédit':>7} {'Confiance':>10}")
    print("-" * 42)
    A2_final, _ = forward(X, W1, b1, W2, b2)
    preds = (A2_final.T > 0.5).astype(int).flatten()
    for i in range(4):
        conf = A2_final[0, i]
        exp  = int(y[i, 0])
        pred = preds[i]
        ok   = "OK" if exp == pred else "ERREUR"
        print(f"{X[i,0]:>4.0f} {X[i,1]:>4.0f} {exp:>9d} {pred:>7d} {conf:>9.4f}  {ok}")

    acc = np.mean(preds == y.flatten().astype(int)) * 100
    print(f"\nPrécision finale : {acc:.1f}%")

    plot_all(history, W1, b1, W2, b2)
