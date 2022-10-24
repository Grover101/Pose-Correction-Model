# Pose Correction

<!-- [![wakatime](https://wakatime.com/badge/github/Grover101/Pose-Correction.svg?style=flat-square)](https://wakatime.com/badge/github/Grover101/Pose-Correction) -->

# Herramientas y Tecnologias

# Procesos del Proyecto

- [x] Obtener Dataset
- [x] Creacion de Dataset
- [x] Data Augmentation
- [x] Normalizacion y Procesamiento de Datos
- [ ] Creacion de Modelo
- [ ] Resultados de Entrenamiento
- [ ] Uso de Modelo para Javascript
- [ ] Creacion de Servicio Web
- [ ] Uso de Modelo en Web
- [ ] Resultado Final

## 1. Obtener Dataset

[![Dataset Pose de Yoga](https://img.shields.io/badge/Download-Dataset-blue?style=flat-square&logo=docusign)](https://drive.google.com/drive/folders/1A5BjyqNvs_q7EfUhTdcNEiesZ8IVxMF2?usp=sharing)

```
   img/
    |__ no_pose/
    |    |__ (1).jpg
    |    |__ ...
    |__ shoudler_stand/
    |    |__ (1).jpg
    |    |__ ...
    |__ traingle/
    |    |__ (1).jpg
    |    |__ ...
    |__ tree/
    |    |__ (1).jpg
    |    |__ ...
    |__ warrior/
         |__ (1).jpg
         |__ ...
```

## 2. Creacion de Dataset

Para crear el dataset a partir de las imagenes ejecutar:

```
python dataset.py
```

```
   dataset/
   |__ train/
       |__ warrior/
       |   |______ (1).jpg
       |   |______ ...
       |__ ...
   |__ test/
       |__ warrior/
       |   |______ (25).jpg
       |   |______ ...
       |__ ...
```

<!-- [text](https://github.com/amalaj7/Pose-Estimation-TFLite) -->

## 3. Data Augmentation

Para crear el dataset augmentation a partir de las imagenes ejecutar:

```
python augmentation.py
```

Para volver a tener el dataset principal con el dataset_augmentation ejecutar:

```
python dataset.py 'augmentation'
```

### Resultados de 5 imagenes generadas a partir de una

| Imagen Original                                     | 1                                                         | 2                                                         | 3                                                         | 4                                                         | 5                                                         |
| --------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| ![tree original](<resources/augmentation/(76).png>) | ![tree modificada 1](resources/augmentation/1471-aug.png) | ![tree modificada 2](resources/augmentation/1472-aug.png) | ![tree modificada 3](resources/augmentation/1473-aug.png) | ![tree modificada 4](resources/augmentation/1474-aug.png) | ![tree modificada 5](resources/augmentation/1475-aug.png) |

## 4. Normalizacion y Procesamiento de Datos

Para la normalizacion y procesamiento de datos se fue leyendo cada imagen de cada clase, se usa `movenet` para la lectura de puntos del cuerpo:

### Documentacion base

[![Prueba del modelo](https://img.shields.io/badge/MoveNet-Docuemtacion-green?style=flat-square&logo=tensorflow)](https://tfhub.dev/google/movenet/singlepose/lightning/4)

### Puntos a tomar en cuenta

[![Prueba del modelo](https://img.shields.io/badge/MoveNet-Prueba%20Demo-orange?style=flat-square&logo=tensorflow)](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet)

![Punto de MoveNet](https://learnopencv.com/wp-content/uploads/2021/05/fix-overlay-issue.jpg)

El dataset a formarse sera un `*.csv` tanto para **train** y **test** por cada imagen se capturara los puntos con las coordenas `(x, y)` de acuerdo a la imagen de referencia correspondiente, se ira guardando por cada clase de postura de yoga. tambien

## 5. Creacion de Modelo

![MLP](https://www.dotnetlovers.com/images/NeuralNetwork314202013722AM.png)
