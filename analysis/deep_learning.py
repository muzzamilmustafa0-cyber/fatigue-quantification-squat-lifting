import sys
import os
import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE   = 42
BATCH_SIZE     = 32
EPOCHS         = 100
PATIENCE       = 15
DS_FACTOR      = 10   # Downsample EMG windows 1020→102 Hz for CNN/LSTM speed


def _get_tf():
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow not installed. Run: pip install tensorflow\n"
            "Deep learning models will be skipped."
        )


def build_mlp_classifier(input_dim: int) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers, regularizers

    inp = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out, name="MLP_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_mlp_regressor(input_dim: int) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers, regularizers

    inp = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inp, out, name="MLP_Regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_cnn_classifier(timesteps: int, n_channels: int = 8) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(timesteps, n_channels))
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out, name="CNN1D_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_regressor(timesteps: int, n_channels: int = 8) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(timesteps, n_channels))
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inp, out, name="CNN1D_Regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_lstm_classifier(timesteps: int, n_channels: int = 8) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(timesteps, n_channels))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out, name="BiLSTM_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_regressor(timesteps: int, n_channels: int = 8) -> "tf.keras.Model":
    tf = _get_tf()
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(timesteps, n_channels))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inp, out, name="BiLSTM_Regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val,
                epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                patience: int = PATIENCE) -> dict:
    tf = _get_tf()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=patience // 2, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return history.history


def eval_clf(model, X_test, y_test) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  precision_score, recall_score, cohen_kappa_score)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Kappa":     cohen_kappa_score(y_test, y_pred),
    }
    try:
        metrics["AUC"] = roc_auc_score(y_test, y_prob)
    except Exception:
        metrics["AUC"] = np.nan
    return metrics, y_pred, y_prob


def eval_reg(model, X_test, y_test) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = model.predict(X_test, verbose=0).ravel()
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mse),
        "R2":   r2_score(y_test, y_pred),
        "MAPE": np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100,
    }
    return metrics, y_pred


def subject_stratified_split(subjects: np.ndarray, test_frac: float = 0.2,
                               random_state: int = RANDOM_STATE):
    np.random.seed(random_state)
    unique = np.unique(subjects)
    np.random.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_subjs  = set(unique[:n_test])
    train_mask = np.array([s not in test_subjs for s in subjects])
    test_mask  = ~train_mask
    return train_mask, test_mask


def normalize_windows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu  = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mu) / std, mu, std


def loso_dl(model_fn, X_windows, y, subjects, task="clf",
            n_loso: int = 5) -> pd.DataFrame:
    unique_subjs = np.unique(subjects)
    np.random.seed(RANDOM_STATE)
    loso_subjs = np.random.choice(unique_subjs, min(n_loso, len(unique_subjs)), replace=False)

    records = []
    for subj in loso_subjs:
        test_mask  = subjects == subj
        train_mask = ~test_mask

        Xtr, Xte = X_windows[train_mask], X_windows[test_mask]
        ytr, yte = y[train_mask], y[test_mask]

        if len(Xte) < 2:
            continue

        # Split train further for validation
        val_size = max(1, int(len(Xtr) * 0.15))
        Xval, yval = Xtr[-val_size:], ytr[-val_size:]
        Xtr, ytr   = Xtr[:-val_size], ytr[:-val_size]

        timesteps  = Xtr.shape[1]
        n_channels = Xtr.shape[2]
        model = model_fn(timesteps, n_channels)
        history = train_model(model, Xtr, ytr, Xval, yval)

        if task == "clf":
            m, _, _ = eval_clf(model, Xte, yte)
        else:
            m, _ = eval_reg(model, Xte, yte)

        records.append(m)
        # Free memory
        del model

    return pd.DataFrame(records)


def run_deep_learning(verbose: bool = True):
    try:
        _get_tf()
    except ImportError as e:
        print(f"[SKIP] {e}")
        return None

    clf_path = RESULTS_DIR / "dataset_classification.csv"
    dl_path  = RESULTS_DIR / "dataset_dl_windows.npz"

    if not clf_path.exists():
        print("[ERROR] Run 04_dataset_builder.py first.")
        return None

    clf_df = pd.read_csv(clf_path)
    meta_cols = ["Subject", "Subject_idx", "Load_kg", "Squat_No",
                 "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
    feat_cols = [c for c in clf_df.columns if c not in meta_cols]

    X_tab  = clf_df[feat_cols].values.astype(np.float32)
    y_clf  = clf_df["Fatigued"].values.astype(np.float32)
    y_rpe  = clf_df["RPE"].values.astype(np.float32)
    subj   = clf_df["Subject_idx"].values.astype(int)

    # Normalize tabular features
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_tab = imp.fit_transform(X_tab).astype(np.float32)
    scl = StandardScaler()
    X_tab = scl.fit_transform(X_tab).astype(np.float32)

    # Train/test split (subject-stratified)
    tr_mask, te_mask = subject_stratified_split(subj)
    val_size = max(1, int(tr_mask.sum() * 0.15))

    all_histories = {}
    clf_records   = []
    reg_records   = []

    import sys as _sys
    if verbose: print("\n[1/6] MLP Classifier ..."); _sys.stdout.flush()
    model = build_mlp_classifier(X_tab.shape[1])
    Xtr, ytr = X_tab[tr_mask], y_clf[tr_mask]
    Xte, yte = X_tab[te_mask], y_clf[te_mask]
    Xval, yval = Xtr[-val_size:], ytr[-val_size:]
    Xtr_, ytr_ = Xtr[:-val_size], ytr[:-val_size]
    hist = train_model(model, Xtr_, ytr_, Xval, yval)
    all_histories["MLP_clf"] = hist
    m, _, _ = eval_clf(model, Xte, yte)
    clf_records.append({"Model": "MLP", **m})
    if verbose: print(f"   Accuracy={m['Accuracy']:.3f}  F1={m['F1']:.3f}  AUC={m.get('AUC', np.nan):.3f}"); _sys.stdout.flush()

    if verbose: print("[2/6] MLP Regressor ..."); _sys.stdout.flush()
    model_r = build_mlp_regressor(X_tab.shape[1])
    ytr_r, yte_r, yval_r = y_rpe[tr_mask], y_rpe[te_mask], y_rpe[tr_mask][-val_size:]
    ytr_r_ = y_rpe[tr_mask][:-val_size]
    hist_r = train_model(model_r, Xtr_, ytr_r_, Xval, yval_r)
    all_histories["MLP_reg"] = hist_r
    m_r, _ = eval_reg(model_r, Xte, yte_r)
    reg_records.append({"Model": "MLP", **m_r})
    if verbose: print(f"   MAE={m_r['MAE']:.3f}  RMSE={m_r['RMSE']:.3f}  R2={m_r['R2']:.3f}"); _sys.stdout.flush()

    has_dl = dl_path.exists()
    if has_dl:
        dl = np.load(dl_path)
        X_win  = dl["windows"].astype(np.float32)
        y_win_clf = dl["labels"].astype(np.float32)
        y_win_rpe = dl["rpe"].astype(np.float32)
        subj_win  = dl["subjects"].astype(int)

        # Drop samples with NaN/inf in windows or labels
        valid_mask = (
            np.isfinite(y_win_rpe) &
            np.isfinite(y_win_clf) &
            np.isfinite(X_win).all(axis=(1, 2))
        )
        X_win, y_win_clf, y_win_rpe, subj_win = (
            X_win[valid_mask], y_win_clf[valid_mask],
            y_win_rpe[valid_mask], subj_win[valid_mask]
        )

        # Downsample timesteps for CNN/LSTM speed on CPU
        X_win = X_win[:, ::DS_FACTOR, :]   # 2040 → 204 timesteps

        # Subsample to max 10k windows for speed
        if len(X_win) > 10000:
            idx = np.random.choice(len(X_win), 10000, replace=False)
            idx = np.sort(idx)
            X_win, y_win_clf, y_win_rpe, subj_win = (
                X_win[idx], y_win_clf[idx], y_win_rpe[idx], subj_win[idx]
            )

        X_win, _, _ = normalize_windows(X_win)
        tr_win, te_win = subject_stratified_split(subj_win)
        val_win_size = max(1, int(tr_win.sum() * 0.15))

        Xtr_w  = X_win[tr_win];  Xte_w  = X_win[te_win]
        ytr_wc = y_win_clf[tr_win]; yte_wc = y_win_clf[te_win]
        ytr_wr = y_win_rpe[tr_win]; yte_wr = y_win_rpe[te_win]
        Xval_w  = Xtr_w[-val_win_size:]
        yvval_c = ytr_wc[-val_win_size:]
        yvval_r = ytr_wr[-val_win_size:]
        Xtr_w_ = Xtr_w[:-val_win_size]
        ytr_wc_ = ytr_wc[:-val_win_size]
        ytr_wr_ = ytr_wr[:-val_win_size]

        ts = X_win.shape[1]
        nc = X_win.shape[2]

        if verbose: print("[3/6] 1D-CNN Classifier ..."); _sys.stdout.flush()
        cnn_c = build_cnn_classifier(ts, nc)
        h = train_model(cnn_c, Xtr_w_, ytr_wc_, Xval_w, yvval_c)
        all_histories["CNN_clf"] = h
        m, _, _ = eval_clf(cnn_c, Xte_w, yte_wc)
        clf_records.append({"Model": "1D-CNN", **m})
        if verbose: print(f"   Accuracy={m['Accuracy']:.3f}  F1={m['F1']:.3f}  AUC={m.get('AUC', np.nan):.3f}")

        if verbose: print("[4/6] 1D-CNN Regressor ..."); _sys.stdout.flush()
        cnn_r = build_cnn_regressor(ts, nc)
        h = train_model(cnn_r, Xtr_w_, ytr_wr_, Xval_w, yvval_r)
        all_histories["CNN_reg"] = h
        m_r, _ = eval_reg(cnn_r, Xte_w, yte_wr)
        reg_records.append({"Model": "1D-CNN", **m_r})
        if verbose: print(f"   MAE={m_r['MAE']:.3f}  RMSE={m_r['RMSE']:.3f}  R2={m_r['R2']:.3f}")

        if verbose: print("[5/6] BiLSTM Classifier ..."); _sys.stdout.flush()
        lstm_c = build_lstm_classifier(ts, nc)
        h = train_model(lstm_c, Xtr_w_, ytr_wc_, Xval_w, yvval_c)
        all_histories["LSTM_clf"] = h
        m, _, _ = eval_clf(lstm_c, Xte_w, yte_wc)
        clf_records.append({"Model": "BiLSTM", **m})
        if verbose: print(f"   Accuracy={m['Accuracy']:.3f}  F1={m['F1']:.3f}  AUC={m.get('AUC', np.nan):.3f}")

        if verbose: print("[6/6] BiLSTM Regressor ..."); _sys.stdout.flush()
        lstm_r = build_lstm_regressor(ts, nc)
        h = train_model(lstm_r, Xtr_w_, ytr_wr_, Xval_w, yvval_r)
        all_histories["LSTM_reg"] = h
        m_r, _ = eval_reg(lstm_r, Xte_w, yte_wr)
        reg_records.append({"Model": "BiLSTM", **m_r})
        if verbose: print(f"   MAE={m_r['MAE']:.3f}  RMSE={m_r['RMSE']:.3f}  R2={m_r['R2']:.3f}")

    else:
        if verbose:
            print("  [INFO] DL window dataset not found — CNN/LSTM skipped.")
            print("         Run 04_dataset_builder.py to generate it.")

    clf_out = pd.DataFrame(clf_records)
    reg_out = pd.DataFrame(reg_records)

    for col in clf_out.select_dtypes(include=np.number).columns:
        clf_out[col] = clf_out[col].round(4)
    for col in reg_out.select_dtypes(include=np.number).columns:
        reg_out[col] = reg_out[col].round(4)

    clf_out.to_csv(RESULTS_DIR / "dl_results_classification.csv", index=False)
    reg_out.to_csv(RESULTS_DIR / "dl_results_regression.csv",     index=False)

    with open(RESULTS_DIR / "dl_training_history.pkl", "wb") as f:
        pickle.dump(all_histories, f)

    if verbose:
        print(f"\n  DL results saved to: {RESULTS_DIR}")

    return {
        "clf_results": clf_out,
        "reg_results": reg_out,
        "histories":   all_histories,
    }


if __name__ == "__main__":
    results = run_deep_learning(verbose=True)
