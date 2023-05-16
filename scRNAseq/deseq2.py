import anndata
# import scvi
import pandas as pd
import numpy as np

from diffexpr.py_deseq import py_DESeq2

def nd(arr):
    """
    Function to transform numpy matrix to nd array.
    """
    return np.asarray(arr).reshape(-1)

# Define number of subsampling runs
subsampling_runs = 10

# Load data
adata = anndata.read_h5ad("../../finchseq_data/all_celltype.h5ad")
control_mask = np.logical_or(adata.obs["batch"]=="control1", adata.obs["batch"]=="control2")
experiment_mask = np.logical_or(adata.obs["batch"]=="experiment1", adata.obs["batch"]=="experiment2")
adata.obs["batch_g"] = ""
adata.obs.loc[control_mask, "batch_g"] = "control"
adata.obs.loc[experiment_mask, "batch_g"] = "experiment"
# Create columns containing general celltype assignment - ignoring cluster separation
adata.obs['celltype_g'] = adata.obs['celltype'].str.replace('\d+', '')

for cluster in adata.obs.celltype.values.unique():   
    # Define subsample size
    cluster_size = len(adata.obs[adata.obs["celltype"]==cluster])
    # If cluster includes > 400 cells, subsample size = 100 cells per condition
    if cluster_size >= 400:
        subsample_size = 100
    # If cluster includes < 400 cells, do not subsample
    elif cluster_size < 400 and cluster_size > 100:
        subsample_size = False
    # Exclude clusters with < 100 cells
    else:
        continue
    
    print(f"Running DESeq2 on celltype cluster {cluster} with subsample size {subsample_size}.")
        
    for i in np.arange(subsampling_runs):
        # Create masks to only retain cells from control or experiment, and one cluster
        celltype_mask_control = np.char.startswith(nd(adata.obs.celltype.values).astype(str), cluster) & np.logical_or(adata.obs["batch"]=="control1", adata.obs["batch"]=="control2")
        celltype_mask_experiment = np.char.startswith(nd(adata.obs.celltype.values).astype(str), cluster) & np.logical_or(adata.obs["batch"]=="experiment1", adata.obs["batch"]=="experiment2")

        # Apply masks to raw data
        control_data = adata.raw[celltype_mask_control]
        experiment_data = adata.raw[celltype_mask_experiment]

        # Filter for only highly variable genes
        control_data = control_data[:,adata.var['highly_variable']]
        experiment_data = experiment_data[:,adata.var['highly_variable']]

        ### Create count matrix
        count_matrix = pd.DataFrame()
        count_matrix["gene"] = adata.raw.var[adata.var['highly_variable']==True].index.values

        ## Create matrix including raw expression of all highly variable genes for each cell from both control animals
        count_matrix_control = pd.DataFrame(control_data.X.todense().T.astype(int))
        
        if subsample_size:
            # Randombly subsample contol data
            count_matrix_control = count_matrix_control.sample(n=subsample_size, axis="columns")
            # Relabel columns 
            count_matrix_control.columns = np.arange(subsample_size)+1
            
        # Add prefix to mark these cells as control
        count_matrix_control = count_matrix_control.add_prefix("C_")

        ## Get raw expression of all highly variable genes for each cell from both experiment animals
        count_matrix_experiment = pd.DataFrame(experiment_data.X.todense().T.astype(int))
        
        if subsample_size:
            # Randombly subsample experiment data
            count_matrix_experiment = count_matrix_experiment.sample(n=subsample_size, axis="columns")
            # Relabel columns 
            count_matrix_experiment.columns = np.arange(subsample_size)+1
            
        # Add prefix to mark these cells as experiment
        count_matrix_experiment = count_matrix_experiment.add_prefix("E_")

        ## Concatenate experiment and control data
        count_matrix = pd.concat(
            [count_matrix, count_matrix_control, count_matrix_experiment], axis=1
        )

        # Show the first count matrix for this cluster
        if i==0:
            print(count_matrix)

        ### Create design matrix
        design_matrix = pd.DataFrame()
        design_matrix["samplename"] = count_matrix.columns.values[1:]
        # Get sample ID and replicate number from count matrix column names
        design_matrix = design_matrix.assign(
            sample=lambda d: d.samplename.str.extract("([CE])_", expand=False)
        )
        design_matrix["replicate"] = design_matrix["samplename"].str.extractall("(\d+)").values
        # Set sample name as index
        design_matrix.index = design_matrix.samplename

        # Show the first design matrix for this cluster
        if i==0:
            print(design_matrix)

        ### Run DESeq2
        dds = py_DESeq2(
            count_matrix=count_matrix,
            design_matrix=design_matrix,
            design_formula="~ replicate + sample",
            gene_column="gene",
        )
        dds.run_deseq(
            test="LRT",                            # Likelihood ratio test on the difference in deviance between a full and reduced model formula (defined by nbinomLRT)
            reduced="~ replicate",                 # Reduced model without condition as variable
            # fitType="parametric",                # Either "parametric", "local", "mean", or "glmGamPoi" for the type of fitting of dispersions to the mean intensity (default: parametric)
            sfType="poscounts",                    # Type of size factor estimation
            betaPrior=False,                       # Whether or not to put a zero-mean normal prior on the non-intercept coefficients (default: False)
            minReplicatesForReplace=float("inf"),  # The minimum number of replicates required in order to use replaceOutliers on a sample. 
            useT=True,                             # Default is FALSE, where Wald statistics are assumed to follow a standard Normal
            minmu=1e-06,                           # Lower bound on the estimated count for fitting gene-wise dispersion (default: 1e-06 for glmGamPoi)

        )
        dds.get_deseq_result(contrast=["sample", "C", "E"])
        res = dds.deseq_result
        
        res.to_csv(f"deseq2_results/deseq2_{cluster.replace(' ', '-').replace('/-', '')}_subsample_{i}.csv")
        
        # Do not run again if cluster was not subsampled
        if subsample_size is False:
            break
            
