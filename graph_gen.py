# Import libraries
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as mp
import pickle
import os
import re
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter

HDWR = 'atom'
# HDWR = 'quera'

params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': r'\usepackage{fontspec,physics}',
}
#Need to install LATEX text on local system for this to work
mpl.rcParams.update(params)

INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] #for algos titles/labels

TITLES = ['adder_9',
            'advantage_9',
            'gcm_h6_13',
            'heisenberg_16',
            'hlf_10',
            'knn_n25',
            'multiplier_10',
            'qaoa_10',
            'qec9xz_n17',
            'qft_10',
            'qugan_n39',
            'qv_32',
            'sat_11',
            'seca_n11',
            'sqrt_18',
            'tfim_128',
            'vqe_uccsd_n28',
            'wstate_27'
]

LABELS = {'adder_9'      : 'ADD'     ,
          'advantage_9'  : 'ADV' ,
          'gcm_h6_13'       : 'GCM'       ,
          'heisenberg_16': 'HSB',
          'hlf_10'       : 'HLF'       ,
          'knn_n25'     : 'KNN'     ,
          'multiplier_10': 'MLT',
          'qaoa_10'      : 'QAOA'      ,
          'qec9xz_n17'       : 'QEC'       ,
          'qft_10'       : 'QFT'       ,
          'qugan_n39'       : 'QGAN'       ,
          'qv_32'       : 'QV'       ,
          'sat_11'       : 'SAT'       ,
          'seca_n11'       : 'SECA'       ,
          'sqrt_18'      : 'SQRT'      ,
          'tfim_128'     : 'TFIM'      ,
          'vqe_uccsd_n28'     : 'VQE'      ,
          'wstate_27'    : 'WST'   }
METHOD = ['Graphine',
          'ELDI',
          'Parallax'
         ]

NUM_U3 = {title: {method: 0 for method in METHOD} for title in TITLES}
NUM_CZ = {title: {method: 0 for method in METHOD} for title in TITLES}
DEPTHS = {title: {method: 0 for method in METHOD} for title in TITLES}

hrdwr = ''
for algo in TITLES:
    with open('./parallax_results/'+HDWR+'/'+algo+'_res.pkl', 'rb') as f:
        par_data = pickle.load(f)
    with open('./disc_graphine_results/'+HDWR+'/'+algo+'_res.pkl', 'rb') as f:
        graphine_data = pickle.load(f)
    try:
        with open('./disc_eldi_results/'+HDWR+'/'+algo+'_res.pkl', 'rb') as f:
            eldi_data = pickle.load(f)
    except:
        pass
    hrdwr = par_data[9]
    if hrdwr != '':
        hrdwr = '/'+par_data[9]+'/'
    NUM_U3[algo]['Parallax'] = par_data[4]
    NUM_CZ[algo]['Parallax'] = par_data[3]
    DEPTHS[algo]['Parallax'] = round(par_data[7])
    
    #Graphine format: algo_name, [all_layers, swap_counts, cz_count, u_count, runtime, compile_time]
    NUM_U3[algo]['Graphine'] = graphine_data[3]
    NUM_CZ[algo]['Graphine'] = graphine_data[2]
    DEPTHS[algo]['Graphine'] = round(graphine_data[4])
    
    #ELDI format: algo_name, [all_layers, swap_counts, cz_count, u_count, runtime, compile_time])
    NUM_U3[algo]['ELDI'] = eldi_data[3]
    NUM_CZ[algo]['ELDI'] = eldi_data[2]
    DEPTHS[algo]['ELDI'] = round(eldi_data[4])

NUM_CZ['vqe_uccsd_n28']['ELDI'] = 0
DEPTHS['vqe_uccsd_n28']['ELDI'] = 0
OFFSET = {'Graphine'  : -0.267,
          'ELDI': 0.0 ,
          'Parallax': 0.267 }

COLORS = {'Graphine'  : '#ffea62',
          'ELDI': '#ffaa3a'   ,
          'Parallax': '#831187'         } # CHANGE NAME, CHANGE COLORS!!!!!


def error_results():
    CZ_ERR = 0.0048
    RAMAN_ERR = 0.000127
    T1 = 4.0 * 1000000 #T1 decoherence time in us
    T2 = 1.49 * 1000000 #T2 decoherence time in us
    fig = mp.figure(figsize=(13.0, 2.3))
    fig.subplots_adjust(left=0.073, top=0.92, right=0.998, bottom=0.12)

    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
    
    bar_width = 0.8 / len(METHOD)
    for i in INDICES:
        # find the maximum value in the current group
        group_max_value = max(
            ((1 - RAMAN_ERR)**NUM_U3[TITLES[i]][m] * 
             (1 - CZ_ERR)**NUM_CZ[TITLES[i]][m] * 
             (1 - (1 - (math.exp(-DEPTHS[TITLES[i]][m] / T1) * math.exp(-DEPTHS[TITLES[i]][m] / T2)))) 
             for m in METHOD)
        )

        for m in METHOD:
            if m == 'ELDI':
                label = r'\textsc{Eldi}'
            elif m == 'Graphine':
                label = r'\textsc{Graphine}'
            elif m == 'Parallax':
                label = r'\textsc{Parallax}'
            if m == 'ELDI' and i == 16:
                continue
            # Calculate success rates as the complement of error rates
            success_rate = (
                (1 - RAMAN_ERR)**NUM_U3[TITLES[i]][m] * 
                (1 - CZ_ERR)**NUM_CZ[TITLES[i]][m] * 
                (1 - (1 - (math.exp(-DEPTHS[TITLES[i]][m] / T1) * math.exp(-DEPTHS[TITLES[i]][m] / T2))))
            )

            # Normalize the bar values to percentages of the group's maximum
            bar_value = 100 * success_rate / group_max_value if group_max_value != 0 else 0

            bar = ax.bar(i + OFFSET[m],
                         bar_value,
                         color=COLORS[m],
                         label=label,
                         linewidth=1.0,
                         edgecolor='black',
                         width=bar_width)

            text_color = 'black' if bar_value <= 30 else 'white'
            text_color = 'black' if COLORS[m] == '#ffea62' or COLORS[m] == '#ffaa3a' else text_color
            va = 'bottom'
            y_text = 2 if bar_value > 30 else max(bar_value + 1,2)
            text_x_position = i + OFFSET[m] + bar_width / 2 - 0.11
            ax.text(text_x_position, 
                    y_text,
                    f"{success_rate:.1e}",
                    color=text_color,
                    ha='center',
                    va=va,
                    fontsize=12,
                    rotation=90)

    ax.set_xlim(-0.6, len(INDICES)-0.4)
    ax.set_yscale('linear')
    ax.set_ylim(0, 100)

    ax.set_xticks(range(len(INDICES)))
    ax.set_xticklabels([LABELS[TITLES[i]] for i in INDICES])

    mp.setp(ax.get_xticklabels(), fontsize=14)
    mp.setp(ax.get_yticklabels(), fontsize=14)

    ax.set_ylabel('Probability of Success \n(\% of Best Case)', fontsize=14)  # Updated ylabel

    mp.savefig('./figures' + hrdwr + 'success.pdf')
    print('./figures' + hrdwr + 'success.pdf')
    mp.close()

def main_results_cz(metric, ylabel):
    fig = mp.figure(figsize=(13.0, 2.3))
    if HDWR == 'quera':
        fig.subplots_adjust(left=0.07, top=0.84, right=0.998, bottom=0.115)
    else:
        fig.subplots_adjust(left=0.07, top=0.957, right=0.998, bottom=0.115)

    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
    
    ax.set_yscale('linear')
    desired_y_ticks = [0, 20, 40, 60, 80, 100]
    bar_width = 0.8 / len(METHOD)
    for i in INDICES:
        group_max_value = max(globals()[metric][TITLES[i]][m] for m in METHOD)

        for m in METHOD:
            if m == 'ELDI':
                label = r'\textsc{Eldi}'
            elif m == 'Graphine':
                label = r'\textsc{Graphine}'
            elif m == 'Parallax':
                label = r'\textsc{Parallax}'

            bar_value = 100 * globals()[metric][TITLES[i]][m] / group_max_value if group_max_value != 0 else 0

            bar = ax.bar(i + OFFSET[m],
                         bar_value,
                         color=COLORS[m],
                         label=label,
                         linewidth=1.0,
                         edgecolor='black',
                         width=bar_width)

            text_color = 'black' if bar_value <= 30 else 'white'
            text_color = 'black' if COLORS[m] == '#ffea62' or COLORS[m] == '#ffaa3a' else text_color
            va = 'bottom'
            y_text = 2 if bar_value > 30 else bar_value + 1
            text_x_position = i + OFFSET[m] + bar_width / 2 - 0.11
            ax.text(text_x_position, # + 0.021
                    y_text,
                    globals()[metric][TITLES[i]][m] if globals()[metric][TITLES[i]][m] != 0 else '',
                    color=text_color,
                    ha='center',
                    va=va,
                    fontsize=12,
                    rotation=90)

    ax.set_xlim(-0.6, len(INDICES)-0.4)
    ax.set_xticks(range(len(INDICES)))
    ax.set_xticklabels([LABELS[TITLES[i]] for i in INDICES])

    mp.setp(ax.get_xticklabels(), fontsize=14)
    mp.setp(ax.get_yticklabels(), fontsize=14)

    ax.set_ylabel(ylabel, fontsize=14)

    ax.set_ylim(0, 100)
    ax.set_yticks(desired_y_ticks)
    ax.set_yticklabels([str(y) for y in desired_y_ticks])

    legend_handles = []
    for m in METHOD:
        if m == 'ELDI':
            label = r'\textsc{Eldi}'
        elif m == 'Graphine':
            label = r'\textsc{Graphine}'
        elif m == 'Parallax':
            label = r'\textsc{Parallax}'
        legend_handles.append(mpatches.Patch(facecolor=COLORS[m], edgecolor='black', label=label))

#This stretches legend across graph
#     if HDWR == 'quera':
#         ax.legend(handles=legend_handles,
#                   ncol=3,  
#                   loc='upper left', 
#                   bbox_to_anchor=(-0.01, 1.07, 1.0175, 0.2),
#                   mode='expand',
#                   fontsize=14, 
#                   handletextpad=0.1,
#                   edgecolor='black')
        
    if HDWR == 'quera':
        ax.legend(handles=legend_handles,
                  ncol=3,
                  loc='upper right', 
                  bbox_to_anchor=(-0.01, 1.07, 1.0175, 0.2),
#                   mode='expand',
                  fontsize=14, 
                  handletextpad=0.3,
                  edgecolor='black')

    mp.savefig('./figures/' + hrdwr + metric.lower() + '.pdf')
    print('./figures/' + hrdwr + metric.lower() + '.pdf')
    mp.close()

main_results_cz('NUM_CZ', 'Num. CZ Gates \n(\% of Worst Case)')
error_results()

"""
Generate Parallelized Circuit Graph
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter

TITLES = ['advantage_9', 'knn_n25', 'qv_32', 'seca_n11', 'sqrt_18', 'wstate_27']

base_dir = 'parallax_par_res'
algo_names = TITLES

def rename_files(subdir, algo_name):
    for filename in os.listdir(subdir):
        if filename.endswith('.pkl') and filename.startswith(algo_name + '_'):
            new_filename = filename.replace(algo_name + '_', '', 1)
            os.rename(os.path.join(subdir, filename), os.path.join(subdir, new_filename))
            print(f"Renamed {filename} to {new_filename}")

for algo_name in algo_names:
    subdir = os.path.join(base_dir, algo_name)
    if os.path.isdir(subdir):
        rename_files(subdir, algo_name)
    else:
        print(f"Directory does not exist: {subdir}")

def decimal_formatter(x, pos):
    if x >= 100:
        return f'{x:.0f}'
    elif x > 10:
        return f'{x:.1f}'
    else:
        return f'{x:.2f}'

def load_new_data(directory):
    all_data = {}
    for algo in TITLES:
        algo_data = []
        algo_directory = f"{directory}/{algo}"
        if not os.path.exists(algo_directory):
            print(f"Directory not found: {algo_directory}")
            continue

        for filename in os.listdir(algo_directory):
            if filename.endswith(".pkl"):
                num_parallel = int(filename.split('_')[0])
                
                with open(os.path.join(algo_directory, filename), 'rb') as file:
                    data = pickle.load(file)
                
                runtime_with_parallelism = data[-2] / num_parallel
                algo_data.append((num_parallel, runtime_with_parallelism))

        algo_data.sort(key=lambda x: x[0])
        all_data[algo] = algo_data

    return all_data

    
data_graphine = load_new_data('./par_graphine_res')
data_eldi = load_new_data('./par_eldi_res')
data_par = load_new_data('./parallax_par_res')

NUM_SHOTS = 8000
fig, axs = plt.subplots(1, 6, figsize=(18, 3))
fig.subplots_adjust(left=0.03, right=0.9, bottom=0.15, top=0.80, wspace=0.25, hspace=0.2)
axs = axs.flatten()

for i, title in enumerate(TITLES):
    num_parallels_graphine, runtimes_graphine = zip(*data_graphine[title])
    num_circs = num_parallels_graphine
    y_labels_graphine = [((NUM_SHOTS / np) * (rt*np)) / 1000000 for np, rt in zip(num_circs, runtimes_graphine)]
    axs[i].plot(num_parallels_graphine, y_labels_graphine, marker='o', linestyle='-', color=COLORS['Graphine'])

    num_parallels_eldi, runtimes_eldi = zip(*data_eldi[title])
    y_labels_eldi = [((NUM_SHOTS / np) * (rt*np)) / 1000000 for np, rt in zip(num_circs, runtimes_eldi)]
    axs[i].plot(num_parallels_eldi, y_labels_eldi, marker='s', linestyle='-', color=COLORS['ELDI'])

    num_parallels_par, runtimes_par = zip(*data_par[title])
    y_labels_par = [((NUM_SHOTS / np) * (rt*np)) / 1000000 for np, rt in zip(num_circs, runtimes_par)]
    axs[i].plot(num_parallels_par, y_labels_par, marker='x', linestyle='-', color=COLORS['Parallax'])

    axs[i].set_title(LABELS[title], fontsize=14)
    axs[i].set_yscale('log')
    axs[i].yaxis.set_minor_locator(ticker.NullLocator())
    axs[i].yaxis.set_major_formatter(NullFormatter())
    axs[i].yaxis.set_minor_formatter(NullFormatter())

    all_y_values = y_labels_graphine + y_labels_eldi + y_labels_par
    min_y, max_y = min(all_y_values), max(all_y_values)
    
    y_ticks = np.logspace(np.log10(min_y), np.log10(max_y), 4)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].set_yticks(y_ticks)
    axs[i].set_yticklabels([f"{y:.2e}" for y in y_ticks])  
    axs[i].yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    axs[i].grid(which='major', axis='y', linestyle='--', linewidth=0.5, color='gray')
    
    if title == "advantage_9":
        axs[i].set_xticks([1,25,49,81,121])
    elif title == 'seca_n11':
        axs[i].set_xticks([1,9,16,25,36,49,64])
    else:
        axs[i].set_xticks(num_circs)

    axs[i].grid(which='major', axis='x', linestyle='--', color='grey', alpha=0.7)
    
graphine_legend = mlines.Line2D([], [], color=COLORS['Graphine'], marker='o', linestyle='-', label=r'\textsc{Graphine}')
eldi_legend = mlines.Line2D([], [], color=COLORS['ELDI'], marker='s', linestyle='-', label=r'\textsc{Eldi}')
parallax_legend = mlines.Line2D([], [], color=COLORS['Parallax'], marker='x', linestyle='-', label=r'\textsc{Parallax}')


fig.text(0.5, -0.02, 'Num. Logical Shots per Physical Shot (Parallelization Factor)', ha='center', fontsize=16)

fig.text(-0.008, 0.45, 'Total Execution Time (s)', va='center', rotation='vertical', fontsize=16)

fig.legend(handles=[graphine_legend, eldi_legend, parallax_legend], loc='upper left', bbox_to_anchor=(0.804, 0.83), ncol=1, fontsize=14)

plt.savefig('./figures/shot_reduction.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

