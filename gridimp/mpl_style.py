from cycler import cycler

# see http://matplotlib.org/users/customizing.html for all options

style1 = {
    # Line styles
    'lines.linewidth': 2.5,
    'lines.antialiased': True,

    # Font
    'font.size': 20.0,
    'font.serif' : 'Times New Roman',

    # Axes
    'axes.linewidth': 2.5,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.prop_cycle': cycler('color', [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # magenta
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow
        '#17becf',  # cyan
    ]),

    # Ticks
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'xtick.major.width': 2.5,
    'xtick.minor.width': 2.5,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.labelsize': 'medium',
    'xtick.direction': 'out',


    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'ytick.major.width': 2.5,
    'ytick.minor.width': 2.5,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.labelsize': 'medium',
    'ytick.direction': 'out',

    # Legend
    'legend.fancybox': True,
    'legend.fontsize': 'large',
    'legend.scatterpoints': 5,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [8, 6],
    'figure.titlesize': 'large',
    ### Subplot adjust:
    'figure.subplot.left'    : 0.154, # the left side of the subplots of the figure
    'figure.subplot.right'   : 0.954, # the right side of the subplots of the figure
    'figure.subplot.bottom'  : 0.136, # the bottom of the subplots of the figure
    'figure.subplot.top'     : 0.936, # the top of the subplots of the figure
    'figure.subplot.wspace'  : 0.200, # the amount of width reserved for blank space between subplots,
                                      # expressed as a fraction of the average axis width
    'figure.subplot.hspace'  : 0.200, # the amount of height reserved for white space between subplots,
                                      # expressed as a fraction of the average axis height

    # Images
    'image.cmap': 'magma',
    'image.origin': 'lower',

    # Error Bars
    'errorbar.capsize': 5,

    # Saving
    'savefig.bbox': 'tight',
    'savefig.format': 'png',

}
