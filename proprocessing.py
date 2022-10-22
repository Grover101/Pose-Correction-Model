import os


route_root = os.getcwd()
route_dataset = os.path.join(route_root, 'dataset')
# Procesamiento de dataos para train en un archivo .csv
images_in_folder = os.path.join(route_dataset, 'train')
print(images_in_folder)
csvs_out_path = 'train_data.csv'
