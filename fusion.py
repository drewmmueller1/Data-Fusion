import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import StringIO

st.title("Spectroscopic Data Fusion App")

fusion_level = st.radio("Select Fusion Level:", ["Low-level (Preprocessed Spectra)", "Mid-level (PCA Scores)"])

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
            ftir_df['label'] = ftir_df.index.astype(str)
        if 'label' not in msp_df.columns:
            msp_df['label'] = msp_df.index.astype(str)
        
        # Set label as index for merging
        ftir_features = ftir_df.set_index('label').select_dtypes(include=[np.number])
        msp_features = msp_df.set_index('label').select_dtypes(include=[np.number])
        
        # Find common labels
        common_labels = ftir_features.index.intersection(msp_features.index)
        if len(common_labels) == 0:
            st.error("No matching labels found between FTIR and MSP files.")
        else:
            st.info(f"Found {len(common_labels)} common samples.")
            
            ftir_sub = ftir_features.loc[common_labels]
            msp_sub = msp_features.loc[common_labels]
            
            # Fuse: concatenate horizontally
            X_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            
            # Standardize? Optional, but good for PCA
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_fused)
            
            if st.button("Run PCA on Fused Data"):
                pca = PCA()
                scores = pca.fit_transform(X_scaled)
                
                # Explained variance ratio for scree plot
                evr = pca.explained_variance_ratio_
                
                # Create tabs for plots
                tab1, tab2, tab3 = st.tabs(["Scree Plot", "PC Scores Plot", "Factor Loadings"])
                
                with tab1:
                    fig_scree, ax_scree = plt.subplots()
                    ax_scree.plot(range(1, len(evr) + 1), np.cumsum(evr), 'bo-')
                    ax_scree.set_xlabel('Number of Components')
                    ax_scree.set_ylabel('Cumulative Explained Variance Ratio')
                    ax_scree.set_title('Scree Plot (Cumulative)')
                    st.pyplot(fig_scree)
                
                with tab2:
                    fig_scores, ax_scores = plt.subplots()
                    ax_scores.scatter(scores[:, 0], scores[:, 1], c=range(len(common_labels)), cmap='viridis')
                    ax_scores.set_xlabel('PC1')
                    ax_scores.set_ylabel('PC2')
                    ax_scores.set_title('PCA Scores Plot (PC1 vs PC2)')
                    st.pyplot(fig_scores)
                
                with tab3:
                    # Loadings: for first few PCs, plot vs variables (need variable names)
                    n_vars = X_fused.shape[1]
                    var_names = X_fused.columns[:min(50, n_vars)]  # Limit for plot
                    loadings_pc1 = pca.components_[0, :len(var_names)]
                    
                    fig_load, ax_load = plt.subplots()
                    ax_load.bar(range(len(var_names)), loadings_pc1)
                    ax_load.set_xlabel('Variables')
                    ax_load.set_ylabel('Loadings (PC1)')
                    ax_load.set_title('Factor Loadings for PC1')
                    ax_load.set_xticks(range(0, len(var_names), 10))
                    ax_load.set_xticklabels(var_names[::10], rotation=45)
                    st.pyplot(fig_load)
                
                # Download fused data option
                st.subheader("Fused Data")
                st.dataframe(X_fused)
                csv = X_fused.to_csv()
                st.download_button("Download Fused CSV", csv, "fused_data.csv")

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
            ftir_pc['label'] = ftir_pc.index.astype(str)
        if 'label' not in msp_pc.columns:
            msp_pc['label'] = msp_pc.index.astype(str)
        
        # Set index to label, select numeric PC columns
        ftir_pc_idx = ftir_pc.set_index('label').select_dtypes(include=[np.number])
        msp_pc_idx = msp_pc.set_index('label').select_dtypes(include=[np.number])
        
        # Common labels
        common_labels = ftir_pc_idx.index.intersection(msp_pc_idx.index)
        if len(common_labels) == 0:
            st.error("No matching labels found.")
        else:
            st.info(f"Found {len(common_labels)} common samples.")
            
            ftir_sub = ftir_pc_idx.loc[common_labels]
            msp_sub = msp_pc_idx.loc[common_labels]
            
            # Assume columns include 'PC1', 'PC2', etc.; warn if not
            if 'PC1' not in msp_sub.columns or 'PC2' not in ftir_sub.columns:
                st.warning("Expected 'PC1' in MSP and 'PC2' in FTIR columns. Adjust column names if needed.")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(msp_sub['PC1'], ftir_sub['PC2'], alpha=0.7)
                ax.set_xlabel('MSP PC1')
                ax.set_ylabel('FTIR PC2')
                ax.set_title('MSP PC1 vs FTIR PC2')
                plt.colorbar(scatter)
                st.pyplot(fig)
            
            # Show fused mid-level data (concat PCs)
            X_mid_fused = pd.concat([ftir_sub, msp_sub], axis=1)
            st.subheader("Fused Mid-level Data")
            st.dataframe(X_mid_fused)
            csv_mid = X_mid_fused.to_csv()
            st.download_button("Download Fused Mid-level CSV", csv_mid, "fused_midlevel.csv")
