import tensorflow as tf
import os
import csv
import tqdm


class Preprocessor(object):
    def __init__(self, images_in_folder,
                 csvs_out_path):
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._csvs_out_folder_per_class = 'data_csv'
        self._message = []

        if (self._csvs_out_folder_per_class not in os.listdir()):
            os.makedirs(self._csvs_out_folder_per_class)

        # obtner lista de clases de modelo
        self._pose_class_names = sorted(
            [n for n in os.listdir(images_in_folder)]
        )

    def process(self, detection_threshold=0.1):
        # Procesamiento de imagenes en una ruta
        for pose_class_name in self._pose_class_names:
            # ruta de clases por pose
            images_in_folder = os.path.join(
                self._images_in_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        pose_class_name + '.csv'
                                        )
            # Deteccion de punto por imagen guardando en un csv
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file,
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL
                                            )
                # Lista de imagenes
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder)]
                )
                valid_image_count = 0
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._message.append(
                            'Skipped' + image_path + ' Invalid image')
                        continue

                    if image.shape[2] != 3:
                        self._message.append(
                            'Skipped' + image_path + ' Image is not in RGB')
                        continue
                    #  procesamiento

        print(self._message)

    def class_names(self):
        return self.pose_class_names


route_root = os.getcwd()
route_dataset = os.path.join(route_root, 'dataset')
# Procesamiento de dataos para train en un archivo .csv
images_in_folder = os.path.join(route_dataset, 'train')
print(images_in_folder)
csvs_out_path = 'train_data.csv'
train_preprocessor = Preprocessor(
    images_in_folder,
    csvs_out_path
)
train_preprocessor.process()
