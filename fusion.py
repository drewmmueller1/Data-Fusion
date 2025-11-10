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
def run_ml(X_fused, y_encoded, class_names):
    if len(class_names) < 2:
        st.warning("Less than 2 classes; classification not meaningful.")
        return None, None, None, None, None, None, None

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

    return X_train, X_test, y_train, y_test, X_2d_train, class_names, X_2d

# Model definitions
models_dict = {
    "LDA": (LinearDiscriminantAnalysis, {}),
    "PLS-DA": (PLSDA, {'n_components': [1, 2, 3, 5]}),
    "KNN": (KNeighborsClassifier, {'n_neighbors': [3, 5, 7, 9]}),
    "FNN": (MLPClassifier, {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
}

if fusion_level == "Low-level (Preprocessed Spectra)":
    st.header("Low-level Fusion: Upload Preprocessed Spectra")
    ftir_file = st.file_uploader("Upload FTIR Spectra CSV (first column: wavenumbers, columns: samples, first row per column: labels)", type="csv")
    msp_file = st.file_uploader("Upload MSP Spectra CSV (first column: wavelengths, columns: samples, first row per column: labels)", type="csv")
    
    if ftir_file is not None and msp_file is not None:
        # Read CSVs: index_col=0 for wavenumbers/wavelengths, columns are sample labels
        ftir_df = pd.read_csv(ftir_file, index_col=0)
        msp_df = pd.read_csv(msp_file, index_col=0)
        
        # Transpose to have samples as rows, features as columns
        ftir_features = ftir_df.T
        msp_features = msp_df.T
        
        # Select numeric only
        ftir_features = ftir_features.select_dtypes(include=[np.number])
        msp_features = msp_features.select_dtypes(include=[np.number])
        
        # Now index is labels (samples), find exact matching labels
        common_labels = ftir_features.index.intersection(msp_features.index)
        if len(common_labels) == 0:
            st.error("No matching labels found between FTIR and MSP files.")
        else:
            st.info(f"Found {len(common_labels)} common samples.")
            
            ftir_sub = ftir_features.loc[common_labels]
            msp_sub = msp_features.loc[common_labels]
            
            # Fuse: concatenate horizontally
            X_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            
            # Fused data option
            st.subheader("Fused Data")
            st.dataframe(X_fused)
            csv = X_fused.to_csv()
            st.download_button("Download Fused CSV", csv, "fused_data.csv")

            # Target selection for visualization and ML
            target = st.selectbox("Select Target", ["Individual", "Sex", "Age"])

            def parse_target(label, t_type):
                base = label.rsplit('_', 1)[0] if '_' in label else label
                date = base[:5]
                sex = base[5]
                age_str = base[6:]
                try:
                    age = int(age_str)
                except ValueError:
                    age = None
                if t_type == "Individual":
                    return date
                elif t_type == "Sex":
                    return 'male' if sex == 'm' else 'female'
                elif t_type == "Age":
                    if age is None:
                        return 'unknown'
                    return age
                return 'unknown'

            y_str = [parse_target(l, target) for l in common_labels]
            if 'unknown' in set(y_str):
                st.error("Some labels have invalid format for selected target.")
            else:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_str)
                class_names = le.classes_

                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_fused)

                # PCA for visualization (2 components)
                pca_2d = PCA(n_components=2)
                scores = pca_2d.fit_transform(X_scaled)

                # Full PCA for scree
                pca_full = PCA()
                pca_full.fit(X_scaled)
                evr = pca_full.explained_variance_ratio_

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
                    show_legend = st.checkbox("Show legend on plot", value=True if len(class_names) <= 20 else False)
                    fig_scores, ax_scores = plt.subplots()
                    scatter = ax_scores.scatter(scores[:, 0], scores[:, 1], c=y_encoded, cmap='tab10', alpha=0.7)
                    ax_scores.set_xlabel('PC1')
                    ax_scores.set_ylabel('PC2')
                    ax_scores.set_title(f'PCA Scores Plot (PC1 vs PC2) - Colored by {target}')
                    if show_legend:
                        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                                                  markerfacecolor=plt.cm.tab10(i / len(class_names)), markersize=10)
                                           for i, cls in enumerate(class_names)]
                        ax_scores.legend(handles=legend_elements)
                    st.pyplot(fig_scores)

                    # Always display labels table
                    label_df = pd.DataFrame({target: y_str}, index=common_labels)
                    st.subheader("Labels")
                    st.dataframe(label_df)

                # ML Section
                st.subheader("Machine Learning Evaluation")
                selected_models = st.multiselect("Select Models", list(models_dict.keys()))

                if st.button("Run ML Evaluation") and len(selected_models) > 0:
                    ml_data = run_ml(X_fused, y_encoded, class_names)
                    if ml_data[0] is None:
                        st.stop()

                    X_train, X_test, y_train, y_test, X_2d_train, class_names, X_2d = ml_data

                    for model_name in selected_models:
                        st.header(f"Results for {model_name}")
                        model_cls, param_grid = models_dict[model_name]

                        if not param_grid:
                            param_grid = {}

                        if model_name == "FNN":
                            estimator = MLPClassifier(max_iter=500, random_state=42)
                        else:
                            estimator = model_cls()
                        gs = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
                        if model_name == "FNN":
                            model_2d = MLPClassifier(max_iter=500, random_state=42, **best_params)
                        elif model_name == "PLS-DA":
                            model_2d = PLSDA(**best_params)
                        else:
                            model_2d = model_cls(**best_params)
                        model_2d.fit(X_2d_train, y_train)

                        fig_db, ax_db = plt.subplots(figsize=(8, 6))
                        plot_decision_regions(X_2d_train, y_train, model_2d, legend=1, ax=ax_db)
                        ax_db.set_xlabel('PC1')
                        ax_db.set_ylabel('PC2')
                        ax_db.set_title(f'Decision Boundary for {model_name} (on PCA 2D)')
                        st.pyplot(fig_db)

elif fusion_level == "Mid-level (PCA Scores)":
    st.header("Mid-level Fusion: Upload PCA Scores")
    ftir_pc_file = st.file_uploader("Upload FTIR PCA Scores CSV (columns: PCs then labels)", type="csv")
    msp_pc_file = st.file_uploader("Upload MSP PCA Scores CSV (columns: PCs then labels)", type="csv")
    
    if ftir_pc_file is not None and msp_pc_file is not None:
        # Read CSVs
        ftir_pc_df = pd.read_csv(ftir_pc_file)
        msp_pc_df = pd.read_csv(msp_pc_file)
        
        # Extract numeric PCs and labels
        ftir_numeric = ftir_pc_df.iloc[:, :-1].select_dtypes(include=[np.number])
        ftir_labels = ftir_pc_df.iloc[:, -1]
        
        msp_numeric = msp_pc_df.iloc[:, :-1].select_dtypes(include=[np.number])
        msp_labels = msp_pc_df.iloc[:, -1]
        
        # Sort by labels to align
        sort_idx_f = ftir_labels.argsort()
        ftir_numeric_sorted = ftir_numeric.iloc[sort_idx_f].reset_index(drop=True)
        ftir_labels_sorted = ftir_labels.iloc[sort_idx_f].reset_index(drop=True)
        
        sort_idx_m = msp_labels.argsort()
        msp_numeric_sorted = msp_numeric.iloc[sort_idx_m].reset_index(drop=True)
        msp_labels_sorted = msp_labels.iloc[sort_idx_m].reset_index(drop=True)
        
        # Take min length to match
        min_len = min(len(ftir_labels_sorted), len(msp_labels_sorted))
        ftir_sub = ftir_numeric_sorted.iloc[:min_len]
        msp_sub = msp_numeric_sorted.iloc[:min_len]
        common_labels = ftir_labels_sorted.iloc[:min_len].values
        
        if len(common_labels) == 0:
            st.error("No matching samples after alignment.")
        else:
            st.info(f"Found {len(common_labels)} aligned samples.")
            
            # Rename columns to avoid duplicates
            ftir_sub.columns = ['FTIR_' + col if col != 'Unnamed: 0' else 'FTIR_index' for col in ftir_sub.columns]
            msp_sub.columns = ['MSP_' + col if col != 'Unnamed: 0' else 'MSP_index' for col in msp_sub.columns]
            
            # Fuse: concatenate horizontally
            X_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            X_fused['label'] = common_labels
            
            # Plot MSP PC1 vs FTIR PC1, colored by label
            if ftir_sub.shape[1] > 0 and len(common_labels) > 0:
                show_legend = st.checkbox("Show legend on plot", value=True if len(np.unique(common_labels)) <= 20 else False)
                fig, ax = plt.subplots(figsize=(8, 6))
                ftir_pc1 = ftir_sub.iloc[:, 0]
                msp_pc1 = msp_sub.iloc[:, 0]
                le_temp = LabelEncoder()
                colors = le_temp.fit_transform(common_labels)
                scatter = ax.scatter(msp_pc1, ftir_pc1, c=colors, cmap='tab10', alpha=0.7)
                ax.set_xlabel('MSP PC1')
                ax.set_ylabel('FTIR PC1')
                ax.set_title('MSP PC1 vs FTIR PC1 - Colored by Label')
                if show_legend:
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                                              markerfacecolor=plt.cm.tab10(i / len(le_temp.classes_)), markersize=10)
                                       for i, cls in enumerate(le_temp.classes_)]
                    ax.legend(handles=legend_elements)
                st.pyplot(fig)

                # Always display labels table
                st.subheader("Labels")
                label_df = pd.DataFrame({'Label': common_labels})
                st.dataframe(label_df)
            else:
                st.warning("Insufficient columns for plotting PC1.")
            
            # Show fused mid-level data
            st.subheader("Fused Mid-level Data")
            st.dataframe(X_fused)
            csv_mid = X_fused.to_csv()
            st.download_button("Download Fused Mid-level CSV", csv_mid, "fused_midlevel.csv")

            # Prepare y for ML
            le = LabelEncoder()
            y_encoded = le.fit_transform(common_labels)
            class_names = le.classes_

            # ML Section
            st.subheader("Machine Learning Evaluation")
            selected_models = st.multiselect("Select Models", list(models_dict.keys()))

            if st.button("Run ML Evaluation") and len(selected_models) > 0:
                ml_data = run_ml(X_fused.iloc[:, :-1], y_encoded, class_names)  # Exclude label column
                if ml_data[0] is None:
                    st.stop()

                X_train, X_test, y_train, y_test, X_2d_train, class_names, X_2d = ml_data

                for model_name in selected_models:
                    st.header(f"Results for {model_name}")
                    model_cls, param_grid = models_dict[model_name]

                    if not param_grid:
                        param_grid = {}

                    if model_name == "FNN":
                        estimator = MLPClassifier(max_iter=500, random_state=42)
                    else:
                        estimator = model_cls()
                    gs = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
                    if model_name == "FNN":
                        model_2d = MLPClassifier(max_iter=500, random_state=42, **best_params)
                    elif model_name == "PLS-DA":
                        model_2d = PLSDA(**best_params)
                    else:
                        model_2d = model_cls(**best_params)
                    model_2d.fit(X_2d_train, y_train)

                    fig_db, ax_db = plt.subplots(figsize=(8, 6))
                    plot_decision_regions(X_2d_train, y_train, model_2d, legend=1, ax=ax_db)
                    ax_db.set_xlabel('PC1')
                    ax_db.set_ylabel('PC2')
                    ax_db.set_title(f'Decision Boundary for {model_name} (on PCA 2D)')
                    st.pyplot(fig_db)
