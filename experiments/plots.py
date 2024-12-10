
# Program to plot 2-D Heat map 
# using matplotlib.pyplot.imshow() method 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from  matplotlib import colors as mcolors
from matplotlib.cm import get_cmap
from FedRLS.ex_fuzzy_manager import parse_rule_base
  

def heatmap(data, row_labels, col_labels, color_labels, ax=None,
            cbar_kw=None, cbarlabel="", location="right", checks=False, ax_cb=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    
    # Create colorbar
    if cbarlabel == "outside":
        pass
    else:
        im_ratio = data.shape[0]/data.shape[1] 
        # Create colorbar
        cbar = ax_cb.figure.colorbar(im, ax=ax_cb, location=location)
        cbar.set_ticks([i+0.5 for i in range(len(color_labels))])
        cbar.ax.tick_params(size=0)
        if location == 'bottom':
            cbar.ax.set_xticklabels(color_labels) 
            cbar.ax.set_xlabel(cbarlabel, rotation=0, va="bottom")
        else:
            cbar.ax.set_yticklabels(color_labels) 
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


    # Remove row's yticks if not labels are present
    if len(row_labels) == 0:
        ax.set_yticks([])
    
    if checks:
        # Make a chek sign for specified (==1) cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] == 1:  # Customize this condition for placing the checkmarks
                    ax.text(j, i, '✔', ha='center', va='center', color='black', fontsize=14)    

    return im


def plot_rules_and_clients(rules, clients_activations,title_rules, title_clients,title,fileName=None,attr=None,n_attr=3,n_classes=3):
    # Information about the clients and which rules are activated on each
    clients_labels = [f"C{i+1}" for i in range(clients_activations.shape[1])]  # Clients labels C1, C2, ...
    clients_colors = ['lightgray','lightgreen']
    clients_colors = [mcolors.CSS4_COLORS[c] for c in clients_colors]
    clients_cmap = mcolors.ListedColormap(clients_colors)
    clients_color_labels = ['Inactive','Active'] 
    clients_bounds = [i for i in range(len(clients_color_labels)+1)]  # These are the edges of your colors, add one extra for the last boundary
    clients_norm = mcolors.BoundaryNorm(clients_bounds, clients_cmap.N)  # cmap.N is the number of discrete colors in the colormap


    # Information about the rules
    # Classes
    classes = rules[:,-1]
    class_labels = [f"C{i+1}" for i in range(n_classes)]
    classes_cmap = get_cmap('Set3',lut=n_classes)   # FIXME: What if more classes are present
    classes_bounds = [i for i in range(len(class_labels)+1)]  # These are the edges of your colors, add one extra for the last boundary
    classes_norm = mcolors.BoundaryNorm(classes_bounds, n_classes)  # cmap.N is the number of discrete colors in the colormap

    # Attributes
    rules = rules[:,:-1]
    if attr is None:
        attr = [f"Att{i+1}" for i in range(rules.shape[1])]
    attr_labels = attr  # Names of the attributes for each rule antecedents

    if n_attr == 3:
        rules_color_labels = ['N/A','Low','Medium','High']  
        ncolors = ['lightgray','lightblue','orange','orangered']
    else:  # FIXME: What happens if not 3 or 5
        rules_color_labels = ['N/A','Very Low','Low','Medium','High','Very High']
        ncolors = ['lightgray','lightblue','lightgreen','lightyellow','orange','orangered'] 

    rules_labels = [f"R{i+1}" for i in range(rules.shape[0])]
    css4colors = [mcolors.CSS4_COLORS[c] for c in ncolors]
    attributes_cmap = mcolors.ListedColormap(css4colors)
    attributes_bounds = [i for i in range(len(rules_color_labels)+1)]  # These are the edges of your colors, add one extra for the last boundary
    attributes_norm = mcolors.BoundaryNorm(attributes_bounds, attributes_cmap.N)  # cmap.N is the number of discrete colors in the colormap


    n_rules = len(rules)
    n_atts = len(attr_labels)
    n_clients = len(clients_labels)

    fig=plt.figure(constrained_layout=True)
    from matplotlib.gridspec import GridSpec
    gs=GridSpec(1, 6, width_ratios=[1,n_atts, 1, 1, n_clients,1])
    gs=GridSpec(2, 6, width_ratios=[1,n_atts, 1, 1, n_clients,1], height_ratios=[12,1])

    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[0,1]) # First row, first column
    ax2=fig.add_subplot(gs[0,2]) # First row, second column
    ax4=fig.add_subplot(gs[0,4]) # First row, third column
    ax5=fig.add_subplot(gs[0,5])
    axh=fig.add_subplot(gs[1,1])
    ax0.set_axis_off() # This axis are for drawing the colorbar
    ax5.set_axis_off()



    # Plot Rules and attributes
    im1= heatmap(
        rules,
        rules_labels,
        attr_labels,
        rules_color_labels,
        ax=ax1,
        cmap=attributes_cmap,
        norm=attributes_norm,
        location="left",
        ax_cb=ax0 # draw the colorbar on the ax0 axis
        ) 

    # Plot Rules's class
    im2 = heatmap(
        classes[:, np.newaxis],
        [],
        ["Class"],
        class_labels,
        ax=ax2,
        cmap=classes_cmap,
        cbarlabel="outside" # draw the colorbar outside (see below)
        )
    
    # Plot Clients activations
    im3 = heatmap(
        clients_activations,
        [],
        clients_labels,
        clients_color_labels,
        ax=ax4,
        cmap=clients_cmap,
        norm=clients_norm,
        location="right",
        checks=True,
        ax_cb=ax5 # draw the colorbar on the ax5 axis
        )  # ,cbarlabel=Linguistic Labels
    
    
    # Draw the classes colorbar/legend below the rules and classes plots
    # axh = fig.add_axes([0.19, new_bottom, 0.35, 0.06])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=axh, orientation='horizontal')
    max_ticks = cbar.get_ticks()[-1]
    pad = max_ticks/(len(class_labels)+1)  
    # Sometimes the labels are not placed correctly
    # print(len(np.unique(classes)))
    # print(classes)
    # print(max_ticks, pad)
    # print([(i*pad)+pad for i in range(n_classes)])
    cbar.set_ticks([(i*pad)+pad for i in range(len(class_labels))])
    cbar.ax.set_xticklabels(class_labels) 
    cbar.set_label('Classes')

    ax1_pos = ax1.get_position()  # Get the [left, bottom, width, height] of the main axis
    ax2_pos = ax2.get_position()  # Get the [left, bottom, width, height] of the main axis
    new_pos = [ax1_pos.x0,ax1_pos.y0 - 0.04,ax2_pos.x1-ax1_pos.x0,0.02]
    axh.set_position(new_pos)

    # Set titles, the title for ax1 on the rigth so it seems centered between ax1 and ax2 (attrs and classes)
    ax1.set_title(title_rules, x=ax1_pos.x0 + (ax2_pos.x1-ax1_pos.x0)/2)
    ax4.set_title(title_clients)

    # FIXME: It overlaps with the rest
    # fig.suptitle(title)

    if fileName:
        plt.savefig(fileName)
        plt.close()    
    else:
        plt.show() 


def plot_rules(rules, title_rules, title,fileName=None,attr=None,n_attr=3,n_classes=3):

    # Information about the rules
    # Classes
    classes = rules[:,-1]
    class_labels = [f"C{i+1}" for i in range(n_classes)]
    classes_cmap = get_cmap('Set3',lut=n_classes)   # FIXME: What if more classes are present
    classes_bounds = [i for i in range(len(class_labels)+1)]  # These are the edges of your colors, add one extra for the last boundary
    classes_norm = mcolors.BoundaryNorm(classes_bounds, n_classes)  # cmap.N is the number of discrete colors in the colormap

    # Attributes
    rules = rules[:,:-1]
    if attr is None:
        attr = [f"Att{i+1}" for i in range(rules.shape[1])]
    attr_labels = attr  # Names of the attributes for each rule antecedents

    if n_attr == 3:
        rules_color_labels = ['N/A','Low','Medium','High']  
        ncolors = ['lightgray','lightblue','orange','orangered']
    else:  # FIXME: What happens if not 3 or 5
        rules_color_labels = ['N/A','Very Low','Low','Medium','High','Very High']
        ncolors = ['lightgray','lightblue','lightgreen','lightyellow','orange','orangered'] 

    rules_labels = [f"R{i+1}" for i in range(rules.shape[0])]
    css4colors = [mcolors.CSS4_COLORS[c] for c in ncolors]
    attributes_cmap = mcolors.ListedColormap(css4colors)
    attributes_bounds = [i for i in range(len(rules_color_labels)+1)]  # These are the edges of your colors, add one extra for the last boundary
    attributes_norm = mcolors.BoundaryNorm(attributes_bounds, attributes_cmap.N)  # cmap.N is the number of discrete colors in the colormap


    n_rules = len(rules)
    n_atts = len(attr_labels)

    fig=plt.figure()  # figsize=((n_atts+4)/2, (n_rules+1)/2)
    from matplotlib.gridspec import GridSpec
    gs=GridSpec(2, 3, width_ratios=[1,n_atts, 1], height_ratios=[12,1])

    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[0,1]) # First row, first column
    ax2=fig.add_subplot(gs[0,2]) # First row, second column
    axh=fig.add_subplot(gs[1,1])
    ax0.set_axis_off() # This axis are for drawing the colorbar



    # Plot Rules and attributes
    im1= heatmap(
        rules,
        rules_labels,
        attr_labels,
        rules_color_labels,
        ax=ax1,
        cmap=attributes_cmap,
        norm=attributes_norm,
        location="left",
        ax_cb=ax0 # draw the colorbar on the ax0 axis
        ) 

    # Plot Rules's class
    im2 = heatmap(
        classes[:, np.newaxis],
        [],
        ["Class"],
        class_labels,
        ax=ax2,
        cmap=classes_cmap,
        cbarlabel="outside" # draw the colorbar outside (see below)
        )
        
    

    # Draw the classes colorbar/legend below the rules and classes plots
    # axh = fig.add_axes([0.19, new_bottom, 0.35, 0.06])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=axh, orientation='horizontal')
    max_ticks = cbar.get_ticks()[-1]
    pad = max_ticks/(len(class_labels)+1)  
    cbar.set_ticks([(i*pad)+pad for i in range(len(class_labels))])
    cbar.ax.set_xticklabels(class_labels) 
    cbar.set_label('Classes')

    ax1_pos = ax1.get_position()  # Get the [left, bottom, width, height] of the main axis
    ax2_pos = ax2.get_position()  # Get the [left, bottom, width, height] of the main axis
    new_pos = [ax1_pos.x0,ax1_pos.y0 - 0.04,ax2_pos.x1-ax1_pos.x0,0.02]
    axh.set_position(new_pos)

    # Set titles, the title for ax1 on the rigth so it seems centered between ax1 and ax2 (attrs and classes)
    ax1.set_title(title_rules, x=ax1_pos.x0 + (ax2_pos.x1-ax1_pos.x0)/2)

    # FIXME: It overlaps with the rest
    # fig.suptitle(title)

    if fileName:
        plt.savefig(fileName)
        plt.close()    
    else:
        plt.show() 


def plot_rules_cent(rules, fileName,n_classes):
    parsed_rules = parse_rule_base(rules)[0]
    rules_mat = []
    classes = []
    for rule in parsed_rules:
        rules_mat.append(rule["var_idx"])
        classes.append(rule["class"])
    rules_mat = np.array(rules_mat)+1 # For the plot


    rules_mat = np.column_stack((rules_mat,classes))
    np.savetxt("final_rules_central.csv", rules_mat, delimiter=";")

    plot_rules(rules_mat,
                title_rules="Rules",
                title="Generated rules and activations per clients",
                fileName=fileName,
                n_classes=n_classes)


def plot_rules_clients(rules, fileName,n_classes):
    parsed_rules = parse_rule_base(rules[0])[0]
    rules_mat = []
    classes = []
    for rule in parsed_rules:
        rules_mat.append(rule["var_idx"])
        classes.append(rule["class"])
    rules_mat = np.array(rules_mat)+1 # For the plot

    scores_mat = []
    for rules_client in rules:
        parsed_rules = parse_rule_base(rules_client)[0]
        c_scores = []
        for rule in parsed_rules:
            c_scores.append(0 if rule["score"]==0 else 1)
        scores_mat.append(c_scores)


    rules_mat = np.column_stack((rules_mat,classes))
    scores_mat = np.array(scores_mat).T

    data = np.column_stack((rules_mat, scores_mat))
    np.savetxt("final_rules_and_activations.csv", data, delimiter=";")

    plot_rules_and_clients(rules_mat, scores_mat,
                    title_rules="Rules",
                    title_clients="Activations",
                    title="Generated rules and activations per clients",
                    fileName=fileName,
                    n_classes=n_classes)

def plot_metrics(df, metric, methods_to_plot=None, plot_title=None,file_name=None):
    if methods_to_plot is None:
        methods_to_plot = df.index.get_level_values('Method').unique()

    plt.figure(figsize=(10, 6))
    for method in methods_to_plot:
        method_data = df.xs(method, level='Method')
        plt.bar(
            method,
            method_data[('mean',metric)],
            yerr=method_data[('std',metric)],
            capsize=5, label=method, alpha=0.8
        )

    if plot_title:
        plt.title(plot_title)
    else:
        plt.title(f'Mean {metric} for Each Method')

    plt.ylabel(f'{metric}')
    plt.xlabel('Methods')
    plt.legend(title='Methods')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if file_name:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

def plot_metrics_per_client(df, metric, methods_to_plot=None, plot_title=None,file_name=None):
    if methods_to_plot is None:
        methods_to_plot = df.index.get_level_values('Method').unique()

    plt.figure(figsize=(10, 6))
    plt.show()



if __name__ == "__main__":
    n_r = 4
    clients = np.random.randint(0,2,( n_r , 3 )) #3 clients    
    rules = np.random.randint(0,3,( n_r , 4 ))
    plot_rules_and_clients(rules, clients,
                    title_rules="Rules",
                    title_clients="Activations",
                    title="Generated rules and activations per clients\n",
                    n_attr=3) 
    plot_rules(rules,
                    title_rules="Rules",
                    title="Generated rules and activations per clients\n",
                    n_attr=3)
