import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from io import StringIO
from mlxtend.plotting import plot_decision_regions
from matplotlib.lines import Line2D

class PLSDA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None
        self.le_ = LabelEncoder()

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_encoded = self.le_.fit_transform(y)
        y_dummy = pd.get_dummies(y_encoded).values
        self.pls.fit(X, y_dummy)
        return self

    def predict(self, X):
        y_pred_prob = self.pls.predict(X)
        y_pred_encoded = np.argmax(y_pred_prob, axis=1)
        y_pred = self.le_.inverse_transform(y_pred_encoded)
        return y_pred

st.title("Spectroscopic Data Fusion App")

fusion_level = st.radio("Select Fusion Level:", ["Low-level (Preprocessed Spectra)", "Mid-level (PCA Scores)"])

# Shared ML section function
@st.cache_data
def run_ml(X_fused, common_labels, target_type):
    if len(common_labels) < 10:
        st.warning("Few samples; results may not be reliable.")
        return None, None, None, None, None, None, None

    def parse_target(label, t_type):
        date = label[:5]
        sex = label[5]
        age_str = label[6:]
        age = int(age_str)
        if t_type == "Individual":
            return date
        elif t_type == "Sex":
            return 'male' if sex == 'm' else 'female'
        elif t_type == "Age":
            return age
        return 'unknown'

    y_str = [parse_target(l, target_type) for l in common_labels]
    y_series = pd.Series(y_str, index=common_labels)
    if 'unknown' in y_series.values:
        st.error("Some labels have invalid format.")
        return None, None, None, None, None, None, None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series)

    # Scale and PCA for ML (10 components)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fused)
    n_comp_ml = min(10, X_fused.shape[0] - 1, X_fused.shape[1])
    pca_ml = PCA(n_components=n_comp_ml)
    X_ml = pca_ml.fit_transform(X_scaled)

    # For 2D plotting
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_ml, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    X_2d_train, X_2d_test, _, _ = train_test_split(X_2d, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    return X_train, X_test, y_train, y_test, X_2d_train, le.classes_, X_2d

# Model definitions
models_dict = {
    "LDA": (LinearDiscriminantAnalysis(), {}),
    "PLS-DA": (PLSDA(), {'n_components': [1, 2, 3, 5]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
    "FNN": (MLPClassifier(max_iter=500, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
}

def group_by_base_label(features_df):
    def get_base_label(label):
        return label.rsplit('_', 1)[0] if '_' in label else label
    features_df['base_label'] = features_df.index.map(get_base_label)
    grouped = features_df.groupby('base_label').mean().drop('base_label', axis=1)
    grouped.index.name = 'label'
    return grouped

if fusion_level == "Low-level (Preprocessed Spectra)":
    st.header("Low-level Fusion: Upload Preprocessed Spectra")
    ftir_file = st.file_uploader("Upload FTIR Spectra CSV (rows: samples/labels, columns: features)", type="csv")
    msp_file = st.file_uploader("Upload MSP Spectra CSV (rows: samples/labels, columns: features)", type="csv")
    
    if ftir_file is not None and msp_file is not None:
        # Read CSVs, assuming first column is 'label', rest are features
        ftir_df = pd.read_csv(ftir_file)
        msp_df = pd.read_csv(msp_file)
        
        # Ensure 'label' column exists; if not, use index as label
        if 'label' not in ftir_df.columns:
            ftir_df.insert(0, 'label', ftir_df.index.astype(str))
        if 'label' not in msp_df.columns:
            msp_df.insert(0, 'label', msp_df.index.astype(str))
        
        # Set label as index for grouping
        ftir_features = ftir_df.set_index('label').select_dtypes(include=[np.number])
        msp_features = msp_df.set_index('label').select_dtypes(include=[np.number])
        
        # Group by base label (exclude replicate)
        ftir_grouped = group_by_base_label(ftir_features)
        msp_grouped = group_by_base_label(msp_features)
        
        # Find common base labels
        common_labels = ftir_grouped.index.intersection(msp_grouped.index)
        if len(common_labels) == 0:
            st.error("No matching base labels found between FTIR and MSP files.")
        else:
            st.info(f"Found {len(common_labels)} common base samples after grouping.")
            
            ftir_sub = ftir_grouped.loc[common_labels]
            msp_sub = msp_grouped.loc[common_labels]
            
            # Fuse: concatenate horizontally
            X_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_fused)
            
            # PCA by default
            pca = PCA()
            scores = pca.fit_transform(X_scaled)
            
            # Explained variance ratio for scree plot
            evr = pca.explained_variance_ratio_
            
            # Create tabs for plots
            tab1, tab2 = st.tabs(["Scree Plot", "PC Scores Plot"])
            
            with tab1:
                fig_scree, ax_scree = plt.subplots()
                ax_scree.plot(range(1, len(evr) + 1), np.cumsum(evr), 'bo-')
                ax_scree.set_xlabel('Number of Components')
                ax_scree.set_ylabel('Cumulative Explained Variance Ratio')
                ax_scree.set_title('Scree Plot (Cumulative)')
                st.pyplot(fig_scree)
            
            with tab2:
                fig_scores, ax_scores = plt.subplots()
                y_sex = ['male' if l[5] == 'm' else 'female' for l in common_labels]
                colors = ['blue' if s == 'male' else 'red' for s in y_sex]
                scatter = ax_scores.scatter(scores[:, 0], scores[:, 1], c=colors, alpha=0.7)
                ax_scores.set_xlabel('PC1')
                ax_scores.set_ylabel('PC2')
                ax_scores.set_title('PCA Scores Plot (PC1 vs PC2)')
                legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male'),
                                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female')]
                ax_scores.legend(handles=legend_elements)
                st.pyplot(fig_scores)
            
            # Fused data option
            st.subheader("Fused Data")
            st.dataframe(X_fused)
            csv = X_fused.to_csv()
            st.download_button("Download Fused CSV", csv, "fused_data.csv")

            # ML Section
            st.subheader("Machine Learning Evaluation")
            target = st.selectbox("Select Target", ["Individual", "Sex", "Age"])
            selected_models = st.multiselect("Select Models", list(models_dict.keys()))

            if st.button("Run ML Evaluation") and len(selected_models) > 0:
                ml_data = run_ml(X_fused, common_labels, target)
                if ml_data[0] is None:
                    st.stop()

                X_train, X_test, y_train, y_test, X_2d_train, class_names, X_2d = ml_data

                for model_name in selected_models:
                    st.header(f"Results for {model_name}")
                    model_cls, param_grid = models_dict[model_name]

                    # If no params, use empty dict
                    if not param_grid:
                        param_grid = {}

                    gs = GridSearchCV(model_cls, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, y_train)

                    st.write(f"Best Parameters: {gs.best_params_}")
                    st.write(f"Cross-Validation Accuracy: {gs.best_score_:.3f}")

                    best_model = gs.best_estimator_

                    # Test predictions
                    y_test_pred = best_model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    st.write(f"Test Accuracy: {test_acc:.3f}")

                    # Test Confusion Matrix
                    cm_test = confusion_matrix(y_test, y_test_pred)
                    fig_test, ax_test = plt.subplots()
                    disp_test = ConfusionMatrixDisplay(cm_test, display_labels=class_names)
                    disp_test.plot(ax=ax_test)
                    st.pyplot(fig_test)

                    # Train Confusion Matrix
                    y_train_pred = best_model.predict(X_train)
                    cm_train = confusion_matrix(y_train, y_train_pred)
                    fig_train, ax_train = plt.subplots()
                    disp_train = ConfusionMatrixDisplay(cm_train, display_labels=class_names)
                    disp_train.plot(ax=ax_train)
                    st.pyplot(fig_train)

                    # Decision Boundary on 2D (fit new model on 2D with best params)
                    best_params = gs.best_params_
                    if model_name == "PLS-DA":
                        model_2d = PLSDA(**best_params)
                    else:
                        model_2d = type(model_cls())(**best_params)
                    model_2d.fit(X_2d_train, y_train)

                    fig_db, ax_db = plt.subplots(figsize=(8, 6))
                    plot_decision_regions(X_2d_train, y_train, model_2d, legend=1, ax=ax_db)
                    ax_db.set_xlabel('PC1')
                    ax_db.set_ylabel('PC2')
                    ax_db.set_title(f'Decision Boundary for {model_name} (on PCA 2D)')
                    st.pyplot(fig_db)

elif fusion_level == "Mid-level (PCA Scores)":
    st.header("Mid-level Fusion: Upload PCA Scores")
    ftir_pc_file = st.file_uploader("Upload FTIR PCA Scores CSV (rows: samples/labels, columns: PCs)", type="csv")
    msp_pc_file = st.file_uploader("Upload MSP PCA Scores CSV (rows: samples/labels, columns: PCs)", type="csv")
    
    if ftir_pc_file is not None and msp_pc_file is not None:
        # Read CSVs
        ftir_pc = pd.read_csv(ftir_pc_file)
        msp_pc = pd.read_csv(msp_pc_file)
        
        # Ensure 'label' column
        if 'label' not in ftir_pc.columns:
            ftir_pc.insert(0, 'label', ftir_pc.index.astype(str))
        if 'label' not in msp_pc.columns:
            msp_pc.insert(0, 'label', msp_pc.index.astype(str))
        
        # Set index to label, select numeric PC columns
        ftir_pc_idx = ftir_pc.set_index('label').select_dtypes(include=[np.number])
        msp_pc_idx = msp_pc.set_index('label').select_dtypes(include=[np.number])
        
        # Group by base label (exclude replicate)
        ftir_grouped = group_by_base_label(ftir_pc_idx)
        msp_grouped = group_by_base_label(msp_pc_idx)
        
        # Common labels
        common_labels = ftir_grouped.index.intersection(msp_grouped.index)
        if len(common_labels) == 0:
            st.error("No matching base labels found.")
        else:
            st.info(f"Found {len(common_labels)} common base samples after grouping.")
            
            ftir_sub = ftir_grouped.loc[common_labels]
            msp_sub = msp_grouped.loc[common_labels]
            
            # Plot MSP PC1 vs FTIR PC1, colored by sex
            if 'PC1' in msp_sub.columns and 'PC1' in ftir_sub.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                y_sex = ['male' if l[5] == 'm' else 'female' for l in common_labels]
                colors = ['blue' if s == 'male' else 'red' for s in y_sex]
                scatter = ax.scatter(msp_sub['PC1'], ftir_sub['PC1'], c=colors, alpha=0.7)
                ax.set_xlabel('MSP PC1')
                ax.set_ylabel('FTIR PC1')
                ax.set_title('MSP PC1 vs FTIR PC1')
                legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male'),
                                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female')]
                ax.legend(handles=legend_elements)
                st.pyplot(fig)
            else:
                st.warning("Expected 'PC1' in both FTIR and MSP columns. Adjust column names if needed.")
            
            # Show fused mid-level data (concat PCs)
            X_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            st.subheader("Fused Mid-level Data")
            st.dataframe(X_fused)
            csv_mid = X_fused.to_csv()
            st.download_button("Download Fused Mid-level CSV", csv_mid, "fused_midlevel.csv")

            # ML Section (same as low-level)
            st.subheader("Machine Learning Evaluation")
            target = st.selectbox("Select Target", ["Individual", "Sex", "Age"])
            selected_models = st.multiselect("Select Models", list(models_dict.keys()))

            if st.button("Run ML Evaluation") and len(selected_models) > 0:
                ml_data = run_ml(X_fused, common_labels, target)
                if ml_data[0] is None:
                    st.stop()

                X_train, X_test, y_train, y_test, X_2d_train, class_names, X_2d = ml_data

                for model_name in selected_models:
                    st.header(f"Results for {model_name}")
                    model_cls, param_grid = models_dict[model_name]

                    if not param_grid:
                        param_grid = {}

                    gs = GridSearchCV(model_cls, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, y_train)

                    st.write(f"Best Parameters: {gs.best_params_}")
                    st.write(f"Cross-Validation Accuracy: {gs.best_score_:.3f}")

                    best_model = gs.best_estimator_

                    # Test predictions
                    y_test_pred = best_model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    st.write(f"Test Accuracy: {test_acc:.3f}")

                    # Test Confusion Matrix
                    cm_test = confusion_matrix(y_test, y_test_pred)
                    fig_test, ax_test = plt.subplots()
                    disp_test = ConfusionMatrixDisplay(cm_test, display_labels=class_names)
                    disp_test.plot(ax=ax_test)
                    st.pyplot(fig_test)

                    # Train Confusion Matrix
                    y_train_pred = best_model.predict(X_train)
                    cm_train = confusion_matrix(y_train, y_train_pred)
                    fig_train, ax_train = plt.subplots()
                    disp_train = ConfusionMatrixDisplay(cm_train, display_labels=class_names)
                    disp_train.plot(ax=ax_train)
                    st.pyplot(fig_train)

                    # Decision Boundary on 2D
                    best_params = gs.best_params_
                    if model_name == "PLS-DA":
                        model_2d = PLSDA(**best_params)
                    else:
                        model_2d = type(model_cls())(**best_params)
                    model_2d.fit(X_2d_train, y_train)

                    fig_db, ax_db = plt.subplots(figsize=(8, 6))
                    plot_decision_regions(X_2d_train, y_train, model_2d, legend=1, ax=ax_db)
                    ax_db.set_xlabel('PC1')
                    ax_db.set_ylabel('PC2')
                    ax_db.set_title(f'Decision Boundary for {model_name} (on PCA 2D)')
                    st.pyplot(fig_db)
