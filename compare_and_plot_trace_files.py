import os
import pandas as pd
import numpy as np
from matplotlib  import pyplot as plt
from PIL import Image

#repertoires et chemins
base_path = r'D:\bacteria_tracker_yoloV8'
result_path = os.path.join(base_path, "plots")

if not os.path.exists(result_path):
    os.mkdir(result_path)

analyse_directories =[
    'result_best_reduced_dataset_yolov8x',
    'result_best_full_dataset_256_yolov8n_confiance_02'
]

#recherche de tous les fichiers trace de chaque repertoires
trace_files = []

for dir in analyse_directories:
    path_dir = os.path.join(base_path, dir)
    trace_files.extend([f for f in os.listdir(path_dir) if f.split('.')[-1] == 'csv'])

#on garde qu'une occurance de chaque nom de fichier
trace_files = list(set(trace_files))

for t_file in trace_files:
    #pour chaque fichier
    data = pd.DataFrame({})
    #on essaye d'ouvrir le nom de fichier dans chque repertoire
    for i, dir in enumerate(analyse_directories):
        path_file = os.path.join(base_path, dir, t_file)
        try:
            #lecture du fichier
            data_csv = pd.read_csv(path_file, delimiter=',')
            data_csv.columns = ['frame_id', 'filename', 'bact_id', 'status', 'x','y','w','h', 'majorL', 'minorL', 'orientation', 'ellipticity', 'area']
            #analyse et ajout dans nouveau fichier
            bact_evolution = []
            for id in range(data_csv['frame_id'].max()):
                #pour chaque frame on cherche le nombre de bact
                bact_evolution.append(data_csv[data_csv['frame_id'] == id]['bact_id'].shape[0])

            #ajout de la nouvelle colonne dans dataframe
            data[dir] = bact_evolution

        except FileNotFoundError:
            pass

        except Exception as e:
            print("error ",e)
    
    plt.clf()
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    # Ajouter une légende
    plt.legend()

    # Ajouter des étiquettes aux axes et un titre au graphe
    plt.xlabel('frame number')
    plt.ylabel('number of bacteria')
    plt.title(t_file)

    # plot = data.plot()
    plot_filename = t_file.split('.')[0] + ".png"
    plot_path = os.path.join(result_path, plot_filename)
    plt.savefig(plot_path)





