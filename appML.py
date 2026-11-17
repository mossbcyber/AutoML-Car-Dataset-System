#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          🚗  AutoML Car Dataset System  🚗                      ║
║  ML Pipeline: RF · XGBoost · LightGBM · GRU · LSTM             ║
║  Adversarial MLP · Adversarial CNN                              ║
║  Metrics: F1, Recall, Precision, Accuracy, Confusion Matrix,    ║
║           ROC, AUC, PR Curve, Time Analysis                     ║
║  Train/Test Split: 70/30                                        ║
╚══════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, GRU, LSTM, Dropout, Conv1D,
    GlobalAveragePooling1D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    DATA GENERATOR                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def generate_car_data(n_samples=20):
    """Generate synthetic car dataset for ML training."""
    np.random.seed(42)

    brands = [
        'Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford',
        'Audi', 'Tesla', 'Hyundai', 'Kia', 'Chevrolet'
    ]
    fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
    transmissions = ['Manual', 'Automatic', 'CVT']
    colors = ['White', 'Black', 'Silver', 'Red', 'Blue', 'Gray']
    conditions = ['New', 'Used', 'Refurbished']

    data = {
        'Car_ID': range(1, n_samples + 1),
        'Brand': np.random.choice(brands, n_samples),
        'Year': np.random.randint(2010, 2024, n_samples),
        'Mileage_km': np.random.randint(0, 200000, n_samples),
        'Engine_CC': np.random.choice(
            [1000, 1200, 1500, 1800, 2000, 2500, 3000, 4000], n_samples
        ),
        'Horsepower': np.random.randint(70, 450, n_samples),
        'Torque_Nm': np.random.randint(100, 600, n_samples),
        'Fuel_Type': np.random.choice(fuel_types, n_samples),
        'Transmission': np.random.choice(transmissions, n_samples),
        'Color': np.random.choice(colors, n_samples),
        'Condition': np.random.choice(conditions, n_samples),
        'Doors': np.random.choice([2, 4, 5], n_samples),
        'Seats': np.random.choice([2, 4, 5, 7], n_samples),
        'Fuel_Efficiency_kmpl': np.round(
            np.random.uniform(8, 35, n_samples), 2
        ),
        'Safety_Rating': np.random.randint(1, 6, n_samples),
        'Insurance_Cost_USD': np.random.randint(500, 5000, n_samples),
        'Maintenance_Cost_USD': np.random.randint(200, 3000, n_samples),
        'Num_Owners': np.random.randint(0, 5, n_samples),
        'Service_History': np.random.choice(
            ['Complete', 'Partial', 'None'], n_samples
        ),
        'Accident_History': np.random.choice(
            [0, 1], n_samples, p=[0.75, 0.25]
        ),
    }

    df = pd.DataFrame(data)

    base_price = (
        (df['Year'] - 2010) * 1500
        + df['Horsepower'] * 80
        + df['Engine_CC'] * 2
        + df['Safety_Rating'] * 500
        - df['Mileage_km'] * 0.05
        - df['Num_Owners'] * 800
        - df['Accident_History'] * 2000
        + df['Fuel_Efficiency_kmpl'] * 200
        + np.random.normal(0, 2000, n_samples)
    )

    df['Price_USD'] = np.clip(base_price, 3000, 150000).astype(int)

    df['Price_Category'] = pd.cut(
        df['Price_USD'],
        bins=[0, 15000, 40000, 200000],
        labels=['Low', 'Medium', 'High']
    )

    df['Is_Premium'] = (df['Price_USD'] > 30000).astype(int)

    return df


def preprocess_data(df, target_col='Is_Premium'):
    """Preprocess the car dataset for ML."""
    df_processed = df.copy()

    drop_cols = ['Car_ID', 'Price_USD', 'Price_Category', 'Is_Premium']
    feature_cols = [
        c for c in df_processed.columns if c not in drop_cols
    ]

    cat_cols = (
        df_processed[feature_cols]
        .select_dtypes(include=['object', 'category'])
        .columns.tolist()
    )
    le_dict = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(
            df_processed[col].astype(str)
        )
        le_dict[col] = le

    X = df_processed[feature_cols]
    y = df_processed[target_col]

    if y.dtype == 'object' or str(y.dtype) == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    else:
        y = y.values
        le_target = None

    return X, y, le_dict, le_target, feature_cols


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    ML MODEL TRAINERS                             ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 1. RANDOM FOREST ──────────────────────────────────────────────

def train_random_forest(
    X_train, y_train, X_test, y_test,
    n_estimators=100, max_depth=None, min_samples_split=2
):
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start

    pred_start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    pred_time = time.time() - pred_start

    cols = (
        X_train.columns
        if hasattr(X_train, 'columns')
        else range(X_train.shape[1])
    )
    feature_importance = dict(zip(cols, model.feature_importances_))

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'feature_importance': feature_importance,
        'model_name': 'Random Forest'
    }


# ── 2. XGBOOST ───────────────────────────────────────────────────

def train_xgboost(
    X_train, y_train, X_test, y_test,
    n_estimators=100, max_depth=6,
    learning_rate=0.1, subsample=0.8
):
    start = time.time()

    n_classes = len(np.unique(y_train))
    objective = (
        'binary:logistic' if n_classes == 2 else 'multi:softprob'
    )
    eval_metric = 'logloss' if n_classes == 2 else 'mlogloss'

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        objective=objective,
        eval_metric=eval_metric,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_time = time.time() - start

    pred_start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    pred_time = time.time() - pred_start

    cols = (
        X_train.columns
        if hasattr(X_train, 'columns')
        else range(X_train.shape[1])
    )
    feature_importance = dict(zip(cols, model.feature_importances_))

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'feature_importance': feature_importance,
        'model_name': 'XGBoost'
    }


# ── 3. LIGHTGBM ──────────────────────────────────────────────────

def train_lightgbm(
    X_train, y_train, X_test, y_test,
    n_estimators=100, max_depth=-1,
    learning_rate=0.1, num_leaves=31
):
    start = time.time()

    n_classes = len(np.unique(y_train))
    objective = 'binary' if n_classes == 2 else 'multiclass'

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        objective=objective,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )
    train_time = time.time() - start

    pred_start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    pred_time = time.time() - pred_start

    cols = (
        X_train.columns
        if hasattr(X_train, 'columns')
        else range(X_train.shape[1])
    )
    feature_importance = dict(zip(cols, model.feature_importances_))

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'feature_importance': feature_importance,
        'model_name': 'LightGBM'
    }


# ── HELPER: reshape for sequence models ──────────────────────────

def _to_sequence(X):
    arr = X.values if hasattr(X, 'values') else X
    return arr.reshape(arr.shape[0], 1, arr.shape[1])


# ── 4. GRU ────────────────────────────────────────────────────────

def train_gru(
    X_train, y_train, X_test, y_test,
    units=64, dropout=0.2, epochs=50,
    batch_size=16, learning_rate=0.001
):
    start = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    X_tr_seq = _to_sequence(X_tr)
    X_te_seq = _to_sequence(X_te)
    n_features = X_tr.shape[1]

    if n_classes == 2:
        y_tr = y_train
        y_te_k = y_test
        out_units, activation, loss = 1, 'sigmoid', 'binary_crossentropy'
    else:
        y_tr = to_categorical(y_train, n_classes)
        y_te_k = to_categorical(y_test, n_classes)
        out_units, activation, loss = (
            n_classes, 'softmax', 'categorical_crossentropy'
        )

    model = Sequential([
        GRU(units, input_shape=(1, n_features), return_sequences=True),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(out_units, activation=activation)
    ])
    model.compile(
        optimizer=Adam(learning_rate), loss=loss, metrics=['accuracy']
    )

    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_tr_seq, y_tr,
        validation_data=(X_te_seq, y_te_k),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )
    train_time = time.time() - start

    pred_start = time.time()
    raw = model.predict(X_te_seq, verbose=0)
    pred_time = time.time() - pred_start

    if n_classes == 2:
        y_prob = np.column_stack([1 - raw.ravel(), raw.ravel()])
        y_pred = (raw.ravel() > 0.5).astype(int)
    else:
        y_prob = raw
        y_pred = np.argmax(raw, axis=1)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'history': history.history,
        'feature_importance': {},
        'model_name': 'GRU'
    }


# ── 5. LSTM ───────────────────────────────────────────────────────

def train_lstm(
    X_train, y_train, X_test, y_test,
    units=64, dropout=0.2, epochs=50,
    batch_size=16, learning_rate=0.001
):
    start = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    X_tr_seq = _to_sequence(X_tr)
    X_te_seq = _to_sequence(X_te)
    n_features = X_tr.shape[1]

    if n_classes == 2:
        y_tr = y_train
        y_te_k = y_test
        out_units, activation, loss = 1, 'sigmoid', 'binary_crossentropy'
    else:
        y_tr = to_categorical(y_train, n_classes)
        y_te_k = to_categorical(y_test, n_classes)
        out_units, activation, loss = (
            n_classes, 'softmax', 'categorical_crossentropy'
        )

    model = Sequential([
        LSTM(units, input_shape=(1, n_features), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(out_units, activation=activation)
    ])
    model.compile(
        optimizer=Adam(learning_rate), loss=loss, metrics=['accuracy']
    )

    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_tr_seq, y_tr,
        validation_data=(X_te_seq, y_te_k),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )
    train_time = time.time() - start

    pred_start = time.time()
    raw = model.predict(X_te_seq, verbose=0)
    pred_time = time.time() - pred_start

    if n_classes == 2:
        y_prob = np.column_stack([1 - raw.ravel(), raw.ravel()])
        y_pred = (raw.ravel() > 0.5).astype(int)
    else:
        y_prob = raw
        y_pred = np.argmax(raw, axis=1)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'history': history.history,
        'feature_importance': {},
        'model_name': 'LSTM'
    }


# ── 6. ROBUST MLP (Adversarial - FGSM) ──────────────────────────

def train_robust_mlp(
    X_train, y_train, X_test, y_test,
    hidden_units=128, dropout=0.3,
    epochs=50, batch_size=16,
    learning_rate=0.001, epsilon=0.01
):
    start = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    n_classes = len(np.unique(y_train))

    if n_classes == 2:
        y_tr = y_train.astype(np.float32)
        y_te_k = y_test.astype(np.float32)
        out_units, activation, loss = 1, 'sigmoid', 'binary_crossentropy'
    else:
        y_tr = to_categorical(y_train, n_classes).astype(np.float32)
        y_te_k = to_categorical(y_test, n_classes).astype(np.float32)
        out_units, activation, loss = (
            n_classes, 'softmax', 'categorical_crossentropy'
        )

    n_features = X_tr.shape[1]

    inp = Input(shape=(n_features,))
    x = Dense(hidden_units, activation='relu')(inp)
    x = Dropout(dropout)(x)
    x = Dense(hidden_units // 2, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_units // 4, activation='relu')(x)
    out = Dense(out_units, activation=activation)(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate), loss=loss, metrics=['accuracy']
    )

    # FGSM adversarial perturbation
    loss_fn = (
        tf.keras.losses.BinaryCrossentropy()
        if n_classes == 2
        else tf.keras.losses.CategoricalCrossentropy()
    )

    def fgsm_perturbation(x_batch, y_batch):
        x_tf = tf.Variable(x_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            pred = model(x_tf, training=False)
            l = loss_fn(
                tf.reshape(y_batch, pred.shape),
                pred
            )
        grads = tape.gradient(l, x_tf)
        if grads is not None:
            perturbation = epsilon * tf.sign(grads)
            return (x_batch + perturbation.numpy()).astype(np.float32)
        return x_batch

    histories = {
        'loss': [], 'val_loss': [],
        'accuracy': [], 'val_accuracy': []
    }

    for epoch in range(epochs):
        try:
            X_adv = fgsm_perturbation(X_tr, y_tr)
            X_combined = np.vstack([X_tr, X_adv])
            if n_classes == 2:
                y_combined = np.concatenate([y_tr, y_tr])
            else:
                y_combined = np.vstack([y_tr, y_tr])
        except Exception:
            X_combined = X_tr
            y_combined = y_tr

        hist = model.fit(
            X_combined, y_combined,
            validation_data=(X_te, y_te_k),
            epochs=1, batch_size=batch_size, verbose=0
        )

        for k in histories:
            if k in hist.history:
                histories[k].extend(hist.history[k])

        if len(histories['val_loss']) > 12:
            recent = histories['val_loss'][-10:]
            if min(recent) == histories['val_loss'][-10]:
                break

    train_time = time.time() - start

    pred_start = time.time()
    raw = model.predict(X_te, verbose=0)
    pred_time = time.time() - pred_start

    if n_classes == 2:
        y_prob = np.column_stack([1 - raw.ravel(), raw.ravel()])
        y_pred = (raw.ravel() > 0.5).astype(int)
    else:
        y_prob = raw
        y_pred = np.argmax(raw, axis=1)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'history': histories,
        'feature_importance': {},
        'model_name': 'Robust MLP (Adversarial)'
    }


# ── 7. ROBUST CNN (1D Adversarial-inspired) ─────────────────────

def train_robust_cnn(
    X_train, y_train, X_test, y_test,
    filters=64, kernel_size=3,
    dropout=0.3, epochs=50,
    batch_size=16, learning_rate=0.001
):
    start = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    X_tr_c = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_te_c = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)
    n_features = X_tr.shape[1]

    if n_classes == 2:
        y_tr = y_train
        y_te_k = y_test
        out_units, activation, loss = 1, 'sigmoid', 'binary_crossentropy'
    else:
        y_tr = to_categorical(y_train, n_classes)
        y_te_k = to_categorical(y_test, n_classes)
        out_units, activation, loss = (
            n_classes, 'softmax', 'categorical_crossentropy'
        )

    ks = min(kernel_size, n_features)

    model = Sequential([
        Conv1D(
            filters, kernel_size=ks, activation='relu',
            input_shape=(n_features, 1), padding='same'
        ),
        Dropout(dropout),
        Conv1D(
            filters // 2, kernel_size=ks,
            activation='relu', padding='same'
        ),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(dropout),
        Dense(out_units, activation=activation)
    ])
    model.compile(
        optimizer=Adam(learning_rate), loss=loss, metrics=['accuracy']
    )

    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_tr_c, y_tr,
        validation_data=(X_te_c, y_te_k),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )
    train_time = time.time() - start

    pred_start = time.time()
    raw = model.predict(X_te_c, verbose=0)
    pred_time = time.time() - pred_start

    if n_classes == 2:
        y_prob = np.column_stack([1 - raw.ravel(), raw.ravel()])
        y_pred = (raw.ravel() > 0.5).astype(int)
    else:
        y_prob = raw
        y_pred = np.argmax(raw, axis=1)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_time': train_time,
        'pred_time': pred_time,
        'history': history.history,
        'feature_importance': {},
        'model_name': 'Robust CNN (Adversarial)'
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    EVALUATION FUNCTIONS                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_metrics(y_true, y_pred, y_prob, model_name='Model'):
    n_classes = len(np.unique(y_true))
    avg = 'binary' if n_classes == 2 else 'weighted'

    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(
            precision_score(y_true, y_pred, average=avg, zero_division=0), 4
        ),
        'Recall': round(
            recall_score(y_true, y_pred, average=avg, zero_division=0), 4
        ),
        'F1_Score': round(
            f1_score(y_true, y_pred, average=avg, zero_division=0), 4
        ),
    }

    try:
        if n_classes == 2:
            metrics['ROC_AUC'] = round(
                roc_auc_score(y_true, y_prob[:, 1]), 4
            )
        else:
            metrics['ROC_AUC'] = round(
                roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr', average='weighted'
                ), 4
            )
    except Exception:
        metrics['ROC_AUC'] = None

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(c) for c in np.unique(y_true)]

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_norm = cm.astype(float) / cm_sum

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Confusion Matrix (Count)',
            'Confusion Matrix (Normalized)'
        ]
    )

    fig.add_trace(go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale='Blues',
        text=cm, texttemplate='%{text}',
        showscale=True, name='Count'
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=np.round(cm_norm, 2),
        x=class_names, y=class_names,
        colorscale='RdYlGn',
        text=np.round(cm_norm, 2),
        texttemplate='%{text:.2f}',
        showscale=True, name='Normalized'
    ), row=1, col=2)

    fig.update_layout(
        title=f'Confusion Matrix — {model_name}',
        height=420, template='plotly_dark'
    )
    return fig


def plot_roc_curves(results_list, y_test):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Random Classifier'
    ))

    colors = px.colors.qualitative.Plotly

    for i, res in enumerate(results_list):
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, res['y_prob'][:, 1])
                roc_auc_val = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f"{res['model_name']} (AUC={roc_auc_val:.3f})",
                    line=dict(
                        color=colors[i % len(colors)], width=2
                    )
                ))
            else:
                classes = np.unique(y_test)
                y_bin = label_binarize(y_test, classes=classes)
                for j, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(
                        y_bin[:, j], res['y_prob'][:, j]
                    )
                    roc_auc_val = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=(
                            f"{res['model_name']} "
                            f"cls-{cls} (AUC={roc_auc_val:.3f})"
                        ),
                        line=dict(width=1.5)
                    ))
        except Exception:
            pass

    fig.update_layout(
        title='ROC Curves — All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=520, template='plotly_dark',
        legend=dict(x=0.55, y=0.05)
    )
    return fig


def plot_pr_curves(results_list, y_test):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    n_classes = len(np.unique(y_test))

    for i, res in enumerate(results_list):
        try:
            if n_classes == 2:
                prec, rec, _ = precision_recall_curve(
                    y_test, res['y_prob'][:, 1]
                )
                ap = average_precision_score(
                    y_test, res['y_prob'][:, 1]
                )
                fig.add_trace(go.Scatter(
                    x=rec, y=prec, mode='lines',
                    name=f"{res['model_name']} (AP={ap:.3f})",
                    line=dict(
                        color=colors[i % len(colors)], width=2
                    )
                ))
        except Exception:
            pass

    fig.update_layout(
        title='Precision-Recall Curves — All Models',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=480, template='plotly_dark'
    )
    return fig


def plot_metrics_comparison(metrics_df):
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    available = [c for c in metric_cols if c in metrics_df.columns]

    fig = go.Figure()
    colors_map = {
        'Accuracy': '#636EFA', 'Precision': '#EF553B',
        'Recall': '#00CC96', 'F1_Score': '#AB63FA',
        'ROC_AUC': '#FFA15A'
    }

    for metric in available:
        fig.add_trace(go.Bar(
            name=metric, x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='outside',
            marker_color=colors_map.get(metric, '#19D3F3')
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        xaxis_title='Model', yaxis_title='Score',
        yaxis=dict(range=[0, 1.15]),
        height=500, template='plotly_dark',
        legend=dict(orientation='h', y=1.12)
    )
    return fig


def plot_time_comparison(metrics_df):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Time (s)', 'Inference Time (s)']
    )

    if 'Train_Time_s' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Train_Time_s'],
            marker_color='#636EFA',
            text=metrics_df['Train_Time_s'].round(3),
            textposition='outside', name='Train Time'
        ), row=1, col=1)

    if 'Pred_Time_s' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Pred_Time_s'],
            marker_color='#EF553B',
            text=metrics_df['Pred_Time_s'].round(4),
            textposition='outside', name='Predict Time'
        ), row=1, col=2)

    fig.update_layout(
        height=420, template='plotly_dark',
        title='Time Comparison Across Models'
    )
    return fig


def plot_feature_importance(feature_importance, model_name, top_n=15):
    if not feature_importance:
        return None

    df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['Feature', 'Importance']
    )
    df = df.sort_values('Importance', ascending=False).head(top_n)

    fig = px.bar(
        df, x='Importance', y='Feature', orientation='h',
        title=f'Top {top_n} Feature Importances — {model_name}',
        color='Importance', color_continuous_scale='Viridis',
        template='plotly_dark'
    )
    fig.update_layout(height=500)
    return fig


def plot_training_history(history, model_name):
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=['Loss', 'Accuracy']
    )

    epochs = list(range(1, len(history.get('loss', [])) + 1))

    if 'loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history['loss'], mode='lines',
            name='Train Loss', line=dict(color='#636EFA')
        ), row=1, col=1)
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'], mode='lines',
            name='Val Loss',
            line=dict(color='#EF553B', dash='dash')
        ), row=1, col=1)
    if 'accuracy' in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history['accuracy'], mode='lines',
            name='Train Acc', line=dict(color='#00CC96')
        ), row=1, col=2)
    if 'val_accuracy' in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_accuracy'], mode='lines',
            name='Val Acc',
            line=dict(color='#FFA15A', dash='dash')
        ), row=1, col=2)

    fig.update_layout(
        title=f'Training History — {model_name}',
        height=420, template='plotly_dark'
    )
    return fig


def plot_radar_chart(metrics_df):
    categories = [
        'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'
    ]
    available = [c for c in categories if c in metrics_df.columns]

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, row in metrics_df.iterrows():
        values = [row.get(c, 0) or 0 for c in available]
        values += [values[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available + [available[0]],
            fill='toself',
            name=row['Model'],
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Radar Chart — Model Comparison',
        height=520, template='plotly_dark'
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    STREAMLIT APP                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="🚗 AutoML Car Dataset System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background: #0e1117; color: #fafafa; }

.metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #3d4270;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card h3 {
    color: #a0aec0; font-size: 0.85rem; margin: 0;
}
.metric-card h2 {
    color: #63b3ed; font-size: 1.8rem;
    font-weight: 700; margin: 4px 0;
}

.section-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 16px 0 12px 0;
}

.banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-bottom: 24px;
    border: 1px solid #4a5568;
}
.banner h1 { color: #90cdf4; font-size: 2.2rem; margin: 0; }
.banner p  { color: #a0aec0; margin: 8px 0 0; }

.badge-success {
    background:#276749; color:#9ae6b4;
    padding:3px 10px; border-radius:20px; font-size:0.8rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none;
    border-radius: 8px; padding: 10px 24px;
    font-weight: 600; width: 100%;
    transition: all 0.3s;
}
.stButton > button:hover {
    opacity:0.85; transform:translateY(-1px);
}

.stProgress > div > div > div {
    background: linear-gradient(90deg,#667eea,#764ba2);
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────

for key in [
    'df', 'results', 'metrics_df', 'X_test',
    'y_test', 'selected_models', 'target_col', 'trained'
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.trained is None:
    st.session_state.trained = False


# ── Banner ────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>🚗 AutoML Car Dataset System</h1>
  <p>ML Pipeline · Random Forest · XGBoost · LightGBM ·
     GRU · LSTM · Adversarial MLP / CNN</p>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    SIDEBAR                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.markdown("---")

    # ── Data Settings ─────────────────────────
    st.markdown("### 📊 Data Settings")
    n_cars = st.slider(
        "Number of Cars (rows)", 20, 500, 20, step=10
    )
    test_size = st.slider("Test Split %", 10, 40, 30, step=5)
    target_option = st.selectbox(
        "Target Variable",
        ['Is_Premium (Binary)', 'Price_Category (Multi-class)']
    )
    target_col = (
        'Is_Premium' if 'Binary' in target_option
        else 'Price_Category'
    )

    st.markdown("---")

    # ── Model Selection ───────────────────────
    st.markdown("### 🤖 Select Models")
    use_rf   = st.checkbox("Random Forest",             value=True)
    use_xgb  = st.checkbox("XGBoost",                  value=True)
    use_lgbm = st.checkbox("LightGBM",                 value=True)
    use_gru  = st.checkbox("GRU",                      value=True)
    use_lstm = st.checkbox("LSTM",                     value=True)
    use_mlp  = st.checkbox("Robust MLP (Adversarial)", value=True)
    use_cnn  = st.checkbox("Robust CNN (Adversarial)", value=True)

    st.markdown("---")

    # ── Hyperparameters ───────────────────────
    st.markdown("### 🔧 Hyperparameters")

    with st.expander("Random Forest"):
        rf_n_est = st.slider(
            "n_estimators", 50, 500, 100, 50, key='rf_n'
        )
        rf_depth = st.slider(
            "max_depth (0=None)", 0, 30, 0, key='rf_d'
        )
        rf_split = st.slider(
            "min_samples_split", 2, 20, 2, key='rf_s'
        )
        rf_max_depth = None if rf_depth == 0 else rf_depth

    with st.expander("XGBoost"):
        xgb_n_est = st.slider(
            "n_estimators", 50, 500, 100, 50, key='xgb_n'
        )
        xgb_depth = st.slider(
            "max_depth", 3, 12, 6, key='xgb_d'
        )
        xgb_lr = st.select_slider(
            "learning_rate",
            [0.001, 0.01, 0.05, 0.1, 0.2, 0.3], 0.1,
            key='xgb_lr'
        )
        xgb_sub = st.slider(
            "subsample", 0.5, 1.0, 0.8, 0.1, key='xgb_s'
        )

    with st.expander("LightGBM"):
        lgb_n_est = st.slider(
            "n_estimators", 50, 500, 100, 50, key='lgb_n'
        )
        lgb_leaves = st.slider(
            "num_leaves", 10, 300, 31, key='lgb_l'
        )
        lgb_lr = st.select_slider(
            "learning_rate",
            [0.001, 0.01, 0.05, 0.1, 0.2, 0.3], 0.1,
            key='lgb_lr'
        )

    with st.expander("GRU / LSTM"):
        rnn_units = st.select_slider(
            "Units", [16, 32, 64, 128, 256], 64
        )
        rnn_drop = st.slider(
            "Dropout", 0.0, 0.5, 0.2, 0.05, key='rnn_d'
        )
        rnn_epochs = st.slider(
            "Epochs", 10, 200, 50, 10, key='rnn_e'
        )
        rnn_batch = st.select_slider(
            "Batch Size", [8, 16, 32, 64], 16
        )
        rnn_lr = st.select_slider(
            "LR", [0.0001, 0.001, 0.01, 0.1], 0.001,
            key='rnn_lr'
        )

    with st.expander("Robust MLP / CNN"):
        nn_hidden = st.select_slider(
            "MLP Hidden Units",
            [32, 64, 128, 256, 512], 128
        )
        nn_drop = st.slider(
            "Dropout", 0.0, 0.5, 0.3, 0.05, key='nn_d'
        )
        nn_epochs = st.slider(
            "Epochs", 10, 200, 50, 10, key='nn_e'
        )
        nn_batch = st.select_slider(
            "Batch Size", [8, 16, 32, 64], 16, key='nn_b'
        )
        nn_lr = st.select_slider(
            "LR", [0.0001, 0.001, 0.01, 0.1], 0.001,
            key='nn_lr'
        )
        nn_eps = st.select_slider(
            "FGSM ε", [0.001, 0.01, 0.05, 0.1, 0.2], 0.01
        )
        cnn_filters = st.select_slider(
            "CNN Filters", [16, 32, 64, 128], 64
        )

    st.markdown("---")
    run_btn = st.button("🚀 Run ML Pipeline")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    MAIN TABS                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

tabs = st.tabs([
    "📋 Dataset",
    "🤖 Train Models",
    "📊 Metrics",
    "🔀 Confusion Matrix",
    "📈 ROC / AUC",
    "📉 PR Curve",
    "⏱ Time Analysis",
    "🔍 Feature Importance",
    "📚 Training History",
    "🕸 Radar Chart",
    "📥 Export"
])

(tab_data, tab_train, tab_metrics, tab_cm, tab_roc,
 tab_pr, tab_time, tab_fi, tab_hist, tab_radar, tab_export) = tabs


# ─────────────────────────────────────────────────────────────────
# Helper guard
# ─────────────────────────────────────────────────────────────────
def _not_trained():
    st.info(
        "⬅️ Configure settings in the sidebar and "
        "click **🚀 Run ML Pipeline**."
    )


# ═══════════════════════════════════════════════
# TAB 1 — DATASET
# ═══════════════════════════════════════════════
with tab_data:
    st.markdown(
        '<div class="section-header">📊 Car Dataset</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🎲 Generate New Dataset"):
            st.session_state.df = generate_car_data(n_cars)
            st.session_state.trained = False
            st.session_state.results = None

    with col2:
        uploaded = st.file_uploader(
            "📁 Upload CSV", type=['csv'], key='upload_csv'
        )
        if uploaded:
            try:
                st.session_state.df = pd.read_csv(uploaded)
                st.session_state.trained = False
                st.session_state.results = None
                st.success(
                    f"✅ Loaded {st.session_state.df.shape[0]} rows"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    with col3:
        st.markdown("#### Add / Update Row")
        if st.session_state.df is not None:
            with st.form("add_row_form"):
                brand = st.selectbox("Brand", [
                    'Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford',
                    'Audi', 'Tesla', 'Hyundai', 'Kia', 'Chevrolet'
                ])
                year    = st.number_input("Year", 2000, 2024, 2020)
                mileage = st.number_input("Mileage (km)", 0, 300000, 50000)
                hp      = st.number_input("Horsepower", 50, 600, 150)
                fuel    = st.selectbox("Fuel Type", [
                    'Petrol', 'Diesel', 'Electric', 'Hybrid'
                ])
                price   = st.number_input(
                    "Price (USD)", 1000, 200000, 25000
                )
                submit  = st.form_submit_button("➕ Add Row")

                if submit:
                    new_id = (
                        st.session_state.df['Car_ID'].max() + 1
                        if 'Car_ID' in st.session_state.df
                        else 1
                    )
                    new_row = pd.DataFrame([{
                        'Car_ID': new_id,
                        'Brand': brand, 'Year': year,
                        'Mileage_km': mileage,
                        'Horsepower': hp,
                        'Fuel_Type': fuel,
                        'Price_USD': price,
                        'Is_Premium': int(price > 30000),
                        'Price_Category': (
                            'Low' if price < 15000
                            else ('Medium' if price < 40000 else 'High')
                        ),
                        'Engine_CC': 1500,
                        'Torque_Nm': 200,
                        'Transmission': 'Automatic',
                        'Color': 'White',
                        'Condition': 'Used',
                        'Doors': 4, 'Seats': 5,
                        'Fuel_Efficiency_kmpl': 15.0,
                        'Safety_Rating': 4,
                        'Insurance_Cost_USD': 1200,
                        'Maintenance_Cost_USD': 600,
                        'Num_Owners': 1,
                        'Service_History': 'Complete',
                        'Accident_History': 0
                    }])
                    st.session_state.df = pd.concat(
                        [st.session_state.df, new_row],
                        ignore_index=True
                    )
                    st.success("✅ Row added!")

    # Auto-generate if no data
    if st.session_state.df is None:
        st.session_state.df = generate_car_data(n_cars)

    df = st.session_state.df

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><h3>Total Cars</h3>'
        f'<h2>{len(df)}</h2></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><h3>Features</h3>'
        f'<h2>{df.shape[1]}</h2></div>',
        unsafe_allow_html=True
    )
    if 'Price_USD' in df.columns:
        c3.markdown(
            f'<div class="metric-card"><h3>Avg Price</h3>'
            f'<h2>${df["Price_USD"].mean():,.0f}</h2></div>',
            unsafe_allow_html=True
        )
    if 'Is_Premium' in df.columns:
        c4.markdown(
            f'<div class="metric-card"><h3>Premium Cars</h3>'
            f'<h2>{df["Is_Premium"].sum()}</h2></div>',
            unsafe_allow_html=True
        )

    st.markdown("#### 🗂 Full Dataset")
    st.dataframe(df, use_container_width=True, height=400)

    # Edit existing row
    st.markdown("#### ✏️ Edit Existing Row")
    if df is not None and len(df) > 0:
        row_idx = st.number_input(
            "Row index to edit", 0, len(df) - 1, 0
        )
        with st.expander("Edit Row"):
            row = df.iloc[row_idx]
            new_vals = {}
            edit_cols = st.columns(3)
            col_list = [
                c for c in df.columns
                if c not in ['Car_ID', 'Price_Category', 'Is_Premium']
            ]
            for i_c, col in enumerate(col_list):
                with edit_cols[i_c % 3]:
                    val = row[col]
                    if df[col].dtype in ['int64', 'int32']:
                        new_vals[col] = st.number_input(
                            col, value=int(val), key=f'edit_{col}'
                        )
                    elif df[col].dtype in ['float64', 'float32']:
                        new_vals[col] = st.number_input(
                            col, value=float(val), key=f'edit_{col}'
                        )
                    else:
                        new_vals[col] = st.text_input(
                            col, value=str(val), key=f'edit_{col}'
                        )

            if st.button("💾 Save Changes"):
                for col, val in new_vals.items():
                    st.session_state.df.at[row_idx, col] = val
                if 'Price_USD' in new_vals:
                    p = new_vals['Price_USD']
                    st.session_state.df.at[row_idx, 'Is_Premium'] = (
                        int(p > 30000)
                    )
                    st.session_state.df.at[row_idx, 'Price_Category'] = (
                        'Low' if p < 15000
                        else ('Medium' if p < 40000 else 'High')
                    )
                st.success("✅ Row updated!")
                st.rerun()

    # Statistics
    st.markdown("#### 📈 Quick Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df.describe(), use_container_width=True)
    with c2:
        if 'Price_Category' in df.columns:
            fig_dist = px.histogram(
                df, x='Price_USD', color='Price_Category',
                nbins=20, title='Price Distribution',
                template='plotly_dark',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════
if run_btn and st.session_state.df is not None:
    df = st.session_state.df

    with st.spinner("⚙️ Preparing data…"):
        try:
            X, y, le_dict, le_target, feature_cols = preprocess_data(
                df, target_col
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size / 100,
                random_state=42,
                stratify=y
            )
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.target_col = target_col
        except Exception as e:
            st.error(f"Data preparation error: {e}")
            st.stop()

    selected = []
    if use_rf:   selected.append('RF')
    if use_xgb:  selected.append('XGB')
    if use_lgbm: selected.append('LGBM')
    if use_gru:  selected.append('GRU')
    if use_lstm: selected.append('LSTM')
    if use_mlp:  selected.append('MLP')
    if use_cnn:  selected.append('CNN')

    results = []
    all_metrics = []

    progress = st.progress(0)
    status   = st.empty()

    model_fns = {
        'RF': lambda: train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=rf_n_est,
            max_depth=rf_max_depth,
            min_samples_split=rf_split
        ),
        'XGB': lambda: train_xgboost(
            X_train, y_train, X_test, y_test,
            n_estimators=xgb_n_est,
            max_depth=xgb_depth,
            learning_rate=xgb_lr,
            subsample=xgb_sub
        ),
        'LGBM': lambda: train_lightgbm(
            X_train, y_train, X_test, y_test,
            n_estimators=lgb_n_est,
            num_leaves=lgb_leaves,
            learning_rate=lgb_lr
        ),
        'GRU': lambda: train_gru(
            X_train, y_train, X_test, y_test,
            units=rnn_units, dropout=rnn_drop,
            epochs=rnn_epochs, batch_size=rnn_batch,
            learning_rate=rnn_lr
        ),
        'LSTM': lambda: train_lstm(
            X_train, y_train, X_test, y_test,
            units=rnn_units, dropout=rnn_drop,
            epochs=rnn_epochs, batch_size=rnn_batch,
            learning_rate=rnn_lr
        ),
        'MLP': lambda: train_robust_mlp(
            X_train, y_train, X_test, y_test,
            hidden_units=nn_hidden,
            dropout=nn_drop,
            epochs=nn_epochs,
            batch_size=nn_batch,
            learning_rate=nn_lr,
            epsilon=nn_eps
        ),
        'CNN': lambda: train_robust_cnn(
            X_train, y_train, X_test, y_test,
            filters=cnn_filters,
            dropout=nn_drop,
            epochs=nn_epochs,
            batch_size=nn_batch,
            learning_rate=nn_lr
        ),
    }

    name_map = {
        'RF': 'Random Forest', 'XGB': 'XGBoost',
        'LGBM': 'LightGBM', 'GRU': 'GRU', 'LSTM': 'LSTM',
        'MLP': 'Robust MLP', 'CNN': 'Robust CNN'
    }

    for i, key in enumerate(selected):
        name = name_map[key]
        status.markdown(
            f"⚙️ Training **{name}**… ({i+1}/{len(selected)})"
        )

        try:
            res = model_fns[key]()
            results.append(res)

            m = compute_metrics(
                y_test, res['y_pred'], res['y_prob'],
                res['model_name']
            )
            m['Train_Time_s'] = round(res['train_time'], 4)
            m['Pred_Time_s']  = round(res['pred_time'], 5)
            all_metrics.append(m)

            status.markdown(
                f"✅ **{name}** done — "
                f"Acc: {m['Accuracy']:.3f} | F1: {m['F1_Score']:.3f}"
            )
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")

        progress.progress((i + 1) / len(selected))

    st.session_state.results        = results
    st.session_state.metrics_df     = pd.DataFrame(all_metrics)
    st.session_state.trained        = True
    st.session_state.selected_models = selected

    progress.progress(1.0)
    status.markdown(
        "🎉 **Pipeline complete!** Navigate tabs to explore results."
    )


# ═══════════════════════════════════════════════
# TAB 2 — TRAIN MODELS (summary)
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown(
        '<div class="section-header">🤖 Training Summary</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        df_m    = st.session_state.metrics_df
        results = st.session_state.results

        cols = st.columns(len(results))
        for i, res in enumerate(results):
            row = df_m[df_m['Model'] == res['model_name']].iloc[0]
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <h3>{res['model_name']}</h3>
                  <h2>{row['Accuracy']:.3f}</h2>
                  <small style='color:#a0aec0'>Accuracy</small><br>
                  <small class='badge-success'>
                    F1: {row['F1_Score']:.3f}
                  </small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📋 Full Results Table")
        styled = df_m.style.highlight_max(
            subset=[
                'Accuracy', 'Precision', 'Recall',
                'F1_Score', 'ROC_AUC'
            ],
            color='#276749'
        ).highlight_min(
            subset=['Train_Time_s'],
            color='#2a4365'
        )
        st.dataframe(styled, use_container_width=True)

        st.markdown("#### 📝 Classification Reports")
        for res in results:
            with st.expander(f"📄 {res['model_name']}"):
                report = classification_report(
                    st.session_state.y_test, res['y_pred'],
                    output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report).T.round(4)
                st.dataframe(report_df, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — METRICS
# ═══════════════════════════════════════════════
with tab_metrics:
    st.markdown(
        '<div class="section-header">📊 Evaluation Metrics</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        df_m = st.session_state.metrics_df

        best_idx   = df_m['Accuracy'].idxmax()
        best_model = df_m.loc[best_idx, 'Model']
        best_acc   = df_m.loc[best_idx, 'Accuracy']

        st.success(
            f"🏆 Best Model: **{best_model}** "
            f"with Accuracy = **{best_acc:.4f}**"
        )

        row = df_m.loc[best_idx]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col_w, metric_name, label in zip(
            [c1, c2, c3, c4, c5],
            ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
            ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        ):
            val = row.get(metric_name, 'N/A')
            col_w.markdown(f"""
            <div class="metric-card">
              <h3>{label}</h3>
              <h2>{val if val is not None else 'N/A'}</h2>
            </div>""", unsafe_allow_html=True)

        st.plotly_chart(
            plot_metrics_comparison(df_m), use_container_width=True
        )

        metric_sel = st.selectbox(
            "Drill-down metric",
            ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        )
        fig_bar = px.bar(
            df_m.sort_values(metric_sel, ascending=False),
            x='Model', y=metric_sel,
            color=metric_sel,
            color_continuous_scale='Viridis',
            title=f'{metric_sel} per Model',
            template='plotly_dark',
            text=metric_sel
        )
        fig_bar.update_traces(
            texttemplate='%{text:.3f}', textposition='outside'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4 — CONFUSION MATRIX
# ═══════════════════════════════════════════════
with tab_cm:
    st.markdown(
        '<div class="section-header">🔀 Confusion Matrices</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        results = st.session_state.results
        y_test  = st.session_state.y_test

        model_names = [r['model_name'] for r in results]
        sel_model = st.selectbox(
            "Select Model", model_names, key='cm_sel'
        )
        res = next(
            r for r in results if r['model_name'] == sel_model
        )

        n_classes = len(np.unique(y_test))
        cls_names = (
            ['Non-Premium', 'Premium']
            if n_classes == 2
            else ['Low', 'Medium', 'High']
        )

        fig_cm = plot_confusion_matrix(
            y_test, res['y_pred'], sel_model, cls_names
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        if st.checkbox("Show All Models"):
            for res in results:
                st.plotly_chart(
                    plot_confusion_matrix(
                        y_test, res['y_pred'],
                        res['model_name'], cls_names
                    ),
                    use_container_width=True
                )


# ═══════════════════════════════════════════════
# TAB 5 — ROC / AUC
# ═══════════════════════════════════════════════
with tab_roc:
    st.markdown(
        '<div class="section-header">📈 ROC Curves</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        results = st.session_state.results
        y_test  = st.session_state.y_test

        st.plotly_chart(
            plot_roc_curves(results, y_test),
            use_container_width=True
        )

        st.markdown("#### AUC Scores")
        auc_data = []
        for res in results:
            try:
                n_cls = len(np.unique(y_test))
                if n_cls == 2:
                    auc_val = roc_auc_score(
                        y_test, res['y_prob'][:, 1]
                    )
                else:
                    auc_val = roc_auc_score(
                        y_test, res['y_prob'],
                        multi_class='ovr', average='weighted'
                    )
                auc_data.append({
                    'Model': res['model_name'],
                    'AUC': round(auc_val, 4)
                })
            except Exception:
                auc_data.append({
                    'Model': res['model_name'], 'AUC': 'N/A'
                })

        st.dataframe(
            pd.DataFrame(auc_data).sort_values(
                'AUC', ascending=False
            ),
            use_container_width=True
        )


# ═══════════════════════════════════════════════
# TAB 6 — PR CURVE
# ═══════════════════════════════════════════════
with tab_pr:
    st.markdown(
        '<div class="section-header">'
        '📉 Precision-Recall Curves</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        results = st.session_state.results
        y_test  = st.session_state.y_test

        if len(np.unique(y_test)) > 2:
            st.info(
                "PR curves shown for binary classification only."
            )
        else:
            st.plotly_chart(
                plot_pr_curves(results, y_test),
                use_container_width=True
            )


# ═══════════════════════════════════════════════
# TAB 7 — TIME ANALYSIS
# ═══════════════════════════════════════════════
with tab_time:
    st.markdown(
        '<div class="section-header">⏱ Time Analysis</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        df_m = st.session_state.metrics_df
        st.plotly_chart(
            plot_time_comparison(df_m), use_container_width=True
        )

        time_df = df_m[['Model', 'Train_Time_s', 'Pred_Time_s']].copy()
        time_df['Train_Time_ms'] = (
            time_df['Train_Time_s'] * 1000
        ).round(2)
        time_df['Pred_Time_ms'] = (
            time_df['Pred_Time_s'] * 1000
        ).round(3)
        st.dataframe(time_df, use_container_width=True)

        if len(time_df) > 0:
            fastest = time_df.loc[
                time_df['Train_Time_s'].idxmin(), 'Model'
            ]
            slowest = time_df.loc[
                time_df['Train_Time_s'].idxmax(), 'Model'
            ]
            c1, c2 = st.columns(2)
            c1.success(f"⚡ Fastest Training: **{fastest}**")
            c2.warning(f"🐌 Slowest Training: **{slowest}**")


# ═══════════════════════════════════════════════
# TAB 8 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════
with tab_fi:
    st.markdown(
        '<div class="section-header">'
        '🔍 Feature Importance</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        results = st.session_state.results
        fi_models = [
            r for r in results if r.get('feature_importance')
        ]

        if not fi_models:
            st.info(
                "Feature importance available for "
                "tree-based models (RF, XGB, LGBM)."
            )
        else:
            sel = st.selectbox(
                "Select Model",
                [r['model_name'] for r in fi_models]
            )
            res = next(
                r for r in fi_models if r['model_name'] == sel
            )
            top_n = st.slider("Top N features", 5, 20, 15)
            fig_fi = plot_feature_importance(
                res['feature_importance'], sel, top_n
            )
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)

            if st.checkbox("Compare All Tree Models"):
                fi_cols = st.columns(len(fi_models))
                for i_fi, r in enumerate(fi_models):
                    with fi_cols[i_fi]:
                        fig = plot_feature_importance(
                            r['feature_importance'],
                            r['model_name'], 10
                        )
                        if fig:
                            st.plotly_chart(
                                fig, use_container_width=True
                            )


# ═══════════════════════════════════════════════
# TAB 9 — TRAINING HISTORY
# ═══════════════════════════════════════════════
with tab_hist:
    st.markdown(
        '<div class="section-header">'
        '📚 Training History (Neural Networks)</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        results = st.session_state.results
        nn_results = [
            r for r in results
            if 'history' in r and r['history']
        ]

        if not nn_results:
            st.info(
                "Training history available for "
                "GRU, LSTM, MLP, CNN models."
            )
        else:
            sel = st.selectbox(
                "Select Model",
                [r['model_name'] for r in nn_results]
            )
            res = next(
                r for r in nn_results if r['model_name'] == sel
            )
            st.plotly_chart(
                plot_training_history(res['history'], sel),
                use_container_width=True
            )

            if st.checkbox("Show All Neural Network Histories"):
                for r in nn_results:
                    st.plotly_chart(
                        plot_training_history(
                            r['history'], r['model_name']
                        ),
                        use_container_width=True
                    )


# ═══════════════════════════════════════════════
# TAB 10 — RADAR CHART
# ═══════════════════════════════════════════════
with tab_radar:
    st.markdown(
        '<div class="section-header">'
        '🕸 Radar Chart — Model Comparison</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.trained:
        _not_trained()
    else:
        df_m = st.session_state.metrics_df
        st.plotly_chart(
            plot_radar_chart(df_m), use_container_width=True
        )

        st.markdown("#### 🌡 Metrics Heatmap")
        metric_cols = [
            'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'
        ]
        avail = [c for c in metric_cols if c in df_m.columns]

        heat_df = df_m[['Model'] + avail].set_index('Model')

        fig_heat = px.imshow(
            heat_df.astype(float),
            text_auto='.3f',
            color_continuous_scale='RdYlGn',
            title='Metrics Heatmap',
            template='plotly_dark',
            aspect='auto'
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 11 — EXPORT
# ═══════════════════════════════════════════════
with tab_export:
    st.markdown(
        '<div class="section-header">📥 Export Results</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📊 Dataset Export")
        if st.session_state.df is not None:
            csv_data = st.session_state.df.to_csv(index=False)
            st.download_button(
                "📥 Download Dataset CSV",
                csv_data, "car_dataset.csv", "text/csv"
            )

    with c2:
        st.markdown("#### 📈 Metrics Export")
        if (
            st.session_state.trained
            and st.session_state.metrics_df is not None
        ):
            metrics_csv = st.session_state.metrics_df.to_csv(
                index=False
            )
            st.download_button(
                "📥 Download Metrics CSV",
                metrics_csv, "model_metrics.csv", "text/csv"
            )

    if (
        st.session_state.trained
        and st.session_state.metrics_df is not None
    ):
        st.markdown("#### 📋 Full Results Preview")
        st.dataframe(
            st.session_state.metrics_df, use_container_width=True
        )

        # JSON
        metrics_json = st.session_state.metrics_df.to_json(
            orient='records', indent=2
        )
        st.download_button(
            "📥 Download Metrics JSON",
            metrics_json, "model_metrics.json",
            "application/json"
        )

        # Text report
        if st.button("📝 Generate Text Report"):
            df_m = st.session_state.metrics_df
            report_lines = [
                "=" * 60,
                "     AutoML Car Dataset System — Report",
                "=" * 60,
                f"Total Models Trained : {len(df_m)}",
                f"Dataset Rows         : {len(st.session_state.df)}",
                f"Target Variable      : {st.session_state.target_col}",
                f"Train/Test Split     : {100-test_size}% / {test_size}%",
                "",
                "─" * 60,
                "PERFORMANCE METRICS",
                "─" * 60,
            ]
            for _, row in df_m.iterrows():
                report_lines += [
                    f"\n▶ {row['Model']}",
                    f"  Accuracy  : {row.get('Accuracy','N/A')}",
                    f"  Precision : {row.get('Precision','N/A')}",
                    f"  Recall    : {row.get('Recall','N/A')}",
                    f"  F1 Score  : {row.get('F1_Score','N/A')}",
                    f"  ROC AUC   : {row.get('ROC_AUC','N/A')}",
                    f"  Train Time: {row.get('Train_Time_s','N/A')}s",
                    f"  Pred Time : {row.get('Pred_Time_s','N/A')}s",
                ]

            best_row = df_m.loc[df_m['Accuracy'].idxmax()]
            report_lines += [
                "",
                "─" * 60,
                f"BEST MODEL   : {best_row['Model']}",
                f"BEST ACCURACY: {best_row['Accuracy']}",
                "=" * 60,
            ]
            report_text = "\n".join(report_lines)
            st.text(report_text)
            st.download_button(
                "📥 Download Report",
                report_text, "ml_report.txt", "text/plain"
            )


# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;'>"
    "🚗 AutoML Car Dataset System · Built with Streamlit, "
    "scikit-learn, XGBoost, LightGBM, TensorFlow"
    "</center>",
    unsafe_allow_html=True
)