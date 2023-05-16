import numpy as np
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd

def nd(arr):
    """
    Funciton to transform numpy matrix to nd array.
    """
    return np.asarray(arr).reshape(-1)

def violinplot(adata_exp, adata_ctrl, genes, labels, celltypes, fig_name, alpha=0.05, fold_change_min=2, figsize=None):
    if figsize == None:
        figsize = (len(celltypes)*2, len(genes))
        fig, axs = plt.subplots(figsize=figsize, nrows=len(genes))
    else:
        fig, axs = plt.subplots(figsize=figsize, nrows=len(genes))
    
    ## Find the indeces of the celltypes/clusters to be used
    # If the first celltype does not contain a number (e.g. "microglia" versus "microglia_1"),
    # use general celltype (celltype_g) to find all of the clusters for that celltye
    # (This is the case for data from the separately clustered dataset)
    if any(map(str.isdigit, celltypes[0])) == False:
        print("Including all clusters for each celltype (obs column: celltype_g).")
        # Define celltypes and celltype indeces in both datasets
        celltype_exp_idx = [np.where(adata_exp.obs.celltype_g == i)[0] for i in celltypes]
        celltype_ctrl_idx = [np.where(adata_ctrl.obs.celltype_g == i)[0] for i in celltypes]
    # Else, use individual clusters
    else:
        print("Individual cluster analysis (obs column: celltype).")
        # Define celltypes and celltype indeces in both datasets
        celltype_exp_idx = [np.where(adata_exp.obs.celltype == i)[0] for i in celltypes]
        celltype_ctrl_idx = [np.where(adata_ctrl.obs.celltype == i)[0] for i in celltypes]

    lidx = np.arange(len(celltypes))*2
    
    fontsize_star = 20
    
    if len(genes) < 2:
        axs = [axs]

    for cidx, (gene, ax) in enumerate(zip(genes, axs)):
        ## Get counts for this gene for all EXP cells
        x_exp_temp = nd(adata_exp.X[:, adata_exp.var.index.str.contains(gene)].todense())
        # Group EXP normalized UMI counts per celltype
        x_exp=[]
        for idx_array in celltype_exp_idx:
            x_exp.append([x_exp_temp[i] for i in idx_array])

        v1 = ax.violinplot(x_exp, showmedians=False, showextrema=False, positions=lidx+0.3)

        ## Get counts for this gene for all CTRL cells
        x_ctrl_temp = nd(adata_ctrl.X[:, adata_ctrl.var.index.str.contains(gene)].todense())
        # Group CTRL normalized UMI counts per celltype
        x_ctrl=[]
        for idx_array in celltype_ctrl_idx:
            x_ctrl.append([x_ctrl_temp[i] for i in idx_array])

        v2 = ax.violinplot(x_ctrl, showmedians=False, showextrema=False, positions=lidx-0.3)
          
        ## Welch's t-test and fold change of mean calculation
        fold_changes = [] 
        p_values = []
        for index, cell_array in enumerate(x_exp):
            # Perform Welch’s t-test, which does not assume equal population variance
            s, p = stats.ttest_ind(cell_array, x_ctrl[index], equal_var=False)
            # Save p-value for violin plot body transparency and heatmap
            p_values.append(p)
            
            if np.mean(cell_array) > np.mean(x_ctrl[index]):
                fold_change = np.mean(cell_array) / np.mean(x_ctrl[index])
                if p < alpha and fold_change >= fold_change_min:
                    ax.annotate("*", (lidx[index], 0.5*ax.get_ylim()[1]), ha="center", c="crimson", fontsize=fontsize_star)
                    
                # Save foldchange for violin plot body transparency and heatmap
                fold_changes.append(fold_change)
                    
            if np.mean(cell_array) <= np.mean(x_ctrl[index]):
                fold_change = np.mean(x_ctrl[index]) / np.mean(cell_array)
                if p < alpha and fold_change >= fold_change_min:
                    ax.annotate("*", (lidx[index], 0.5*ax.get_ylim()[1]), ha="center", c="blue", fontsize=fontsize_star)
                    
                # Save foldchange for violin plot body transparency and heatmap
                fold_changes.append(fold_change)
                    
        ## Set color and transparency of the violin plot bodies
        # Set transparency based on fold change (FC) and p-value
        # All violin plots showing an FC >= fold_change_min and p < alpha will be 100% opaque; 
        # for FCs < fold_change_min and p > alpha will be 10% opaque
        # Violin plots showing experiment data:
        for pcidx, pc in enumerate(v1["bodies"]):
            pc.set_facecolor("crimson")  
            pc.set_edgecolor("black")
            if fold_changes[pcidx] >= fold_change_min and p_values[pcidx] < alpha:
                pc.set_alpha(1)
            else:
                pc.set_alpha(0.1)
        # Violin plots showing control data:
        for pcidx2, pc2 in enumerate(v2["bodies"]):
            pc2.set_facecolor("blue")
            pc2.set_edgecolor("black")
            if fold_changes[pcidx2] >= fold_change_min and p_values[pcidx2] < alpha:
                pc2.set_alpha(1)
            else:
                pc2.set_alpha(0.1)

        ## Set up x- and y- tick labels, and distinct top and bottom axes  
        # Get total number of cells per celltype cluster
        cellcounts_exp=[]
        for array in x_exp:
            cellcounts_exp.append(len(array)) 
        cellcounts_ctrl=[]
        for array in x_ctrl:
            cellcounts_ctrl.append(len(array)) 

        xticklabels=[]    
        for i2, (celltype, cellcount) in enumerate(zip(celltypes, cellcounts_exp)):
            xticklabels.append("{} \n(Control: {}; TetX: {})".format(celltype, cellcounts_ctrl[i2], cellcount))

        if cidx==0:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim()) # DO NOT DELETE THIS
            ax_top.set_ylim(ax.get_ylim()[0], 2.5*ax.get_ylim()[1]) # Slightly increase space on top of first row of plots
            ax_top.set_xticks(lidx)
            ax_top.set_xticklabels(xticklabels, rotation=45, ha="left")
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["left"].set_visible(False)
            ax_top.spines["bottom"].set_visible(False)
            ax_top.xaxis.grid(False) 

        if cidx == len(genes)-1:
            ax_bot = ax.twiny()
            ax_bot.set_xticks([])
            ax_bot.set_xticklabels([])
            ax_bot.spines["top"].set_visible(False)
            ax_bot.spines["left"].set_visible(False)
            ax_bot.spines["bottom"].set_visible(False)

        ax.yaxis.tick_right()
        ax.set_ylabel(labels[cidx], color="black",rotation="horizontal", ha="right", va="center")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_axisbelow(True)
        ax.xaxis.grid(False) 
        
        # Set y axis on log scale including 0
        ax.set_yscale('symlog')

        ax.tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    plt.savefig("figures/vplot_{}.png".format(fig_name), bbox_inches="tight", dpi=300, transparent=True)
    plt.show()
