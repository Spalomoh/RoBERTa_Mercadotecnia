# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:37:24 2023

@author: Sandra Palomo
"""

#------------------ROBERTA-------------------------  


#pip install torch
import torch
print(torch.__version__)
pip install --upgrade transformers


import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax  

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

import pandas as pd
import re
import seaborn as sns
import numpy as np
    

np.set_printoptions(precision=2, linewidth=80)

dataset = pd.read_csv("Reviews.csv", encoding='latin-1')



modeloRoberta = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(modeloRoberta)
model = AutoModelForSequenceClassification.from_pretrained(modeloRoberta)    


#----------------Ejemplo----------------------------
example = dataset['Comment'][297]
print(example)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


#---------------------------------------------------



# Resultados Polaridades RoBERTa
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
    try:
        text = row['Comment']
        myid = row['id']
        roberta_result = polarity_scores_roberta(text)
        res[myid] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}')

results_dataset = pd.DataFrame(res).T
results_dataset = results_dataset.reset_index().rename(columns={'index': 'id'})
results_dataset = results_dataset.merge(dataset, how='left')

results_dataset.head()


results_dataset.to_csv('resultados_sentimiento.csv', index=False)  



#---------------Subcomentarios-----------------------

def divide_comentarios(texto):
    # Dividir el texto en subcomentarios
    subcomentarios = re.split(r'[.;]', texto)
    # Elimina espacios en blanco  y elementos vacíos
    subcomentarios = [s.strip() for s in subcomentarios if s.strip()]
    return subcomentarios


comentarios = dataset['Comment']  # Reemplaza 'dataset' con tu dataframe
subcomentarios = comentarios.apply(divide_comentarios)

#---------------------Verificar------------------------------#
indice_comentario = 508
comentario = dataset['Comment'][indice_comentario]
#----------------------------------------------------#



def analiza_comment(comment):
    subcomments = comment.split('. ')
    scores = []

    for i, subcomment in enumerate(subcomments, start=1):
        roberta_result = polarity_scores_roberta(subcomment)
        scores.append(roberta_result)
        print(f"Subcomentario {i}: {subcomment}")
        print(f"Puntuación de sentimiento: {roberta_result}")

    return scores

# Se obtienen las puntuaciones de sentimiento para los subcomentarios
subcomment_scores = analyze_comment(comentario)



#----------Ubicaciones. Division en subcomentarios de un comentario seleccionado-------------------------------


# Carga el archivo de Reviews
reviews_df = pd.read_csv("Reviews.csv", encoding='ISO-8859-1')

ubicaciones_clave = ['beach', 'sea', 'cortez', 'santa maria', 'divorce beach', 'lovers', "land's end", 'neptune fingers']

def buscar_ubicaciones(texto):
    ubicaciones_encontradas = [ubicacion for ubicacion in ubicaciones_clave if ubicacion in texto.lower()]
    return ubicaciones_encontradas

# Crea listas para almacenar los resultados
comentarios_originales = []
ubicaciones_encontradas = []
subcomentarios = []

# Itera a través de los comentarios del DataFrame
for comentario in reviews_df['Comment']:
    ubicaciones = buscar_ubicaciones(comentario)
    if ubicaciones:
        comentarios_originales.append(comentario)
        ubicaciones_encontradas.append(', '.join(ubicaciones))

# Crea un nuevo DataFrame con los resultados
resultados_df = pd.DataFrame({'Comentario Original': comentarios_originales,
                              'Ubicacion Encontrada': ubicaciones_encontradas})

# Guarda los resultados en un nuevo archivo 
resultados_df.to_csv("Resultados.csv", index=False)


#------Verificar------

fila_a_imprimir = 4 
comentario_original = resultados_df.at[fila_a_imprimir, 'Comentario Original']
subcomentarios = divide_comentarios(comentario_original)

#----------------------------


def divide_comentarios(texto):
    subcomentarios = re.split(r'[.;]', texto)
    subcomentarios = [s.strip() for s in subcomentarios if s.strip()]
    return subcomentarios

def find_subcomments_by_location(comentario, ubicaciones):
    subcomentarios = divide_comentarios(comentario)

    # Lista para almacenar los subcomentarios que contienen ubicaciones
    subcomentarios_filtrados = []

    for subcomentario in subcomentarios:
        subcomentario_lower = subcomentario.lower()

        # Verificamos ubicaciones en subcomentarios
        if any(ubicacion in subcomentario_lower for ubicacion in ubicaciones):
            subcomentarios_filtrados.append(subcomentario)

    return subcomentarios_filtrados

# Carga el conjunto de datos
dataset = pd.read_csv("Resultados.csv")

# Obtiene el comentario original que se quiere dividir en subcomentarios
indice = 24  # Indice del comentario
comentario_original = dataset["Comentario Original"][indice]
ubicacion_encontrada = dataset["Ubicacion Encontrada"][indice].split(', ')

# Encuentra los subcomentarios que contienen las ubicaciones encontradas
subcomentarios_filtrados = find_subcomments_by_location(comentario_original, ubicacion_encontrada)

# Imprime la información 
print(f"Comentario Original: {comentario_original}")
print(f"Ubicaciones Encontradas: {', '.join(ubicacion_encontrada)}")
print("Subcomentarios:")
for subcomentario in subcomentarios_filtrados:
    print(f"  {subcomentario}")



#------ubicaciones más mencionadas------
import matplotlib.pyplot as plt
from collections import Counter

# Carga el archivo de Resultados
resultados_df = pd.read_csv("Resultados.csv")

# Se divide y se cuentan las ubicaciones
ubicaciones_mencionadas = resultados_df['Ubicacion Encontrada'].str.split(', ').explode()
ubicaciones_contadas = ubicaciones_mencionadas.value_counts()

#  gráfico de barras
plt.figure(figsize=(10, 6))
ubicaciones_contadas.plot(kind='bar')
plt.xlabel('Ubicación')
plt.ylabel('Número de Menciones')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

#########
#--------------Frecuencia de Menciones-------------

# Archivo Resultados
resultados_df = pd.read_csv("Resultados.csv")

# Número total de comentarios
total_comentarios = len(resultados_df)

# CLista con todas las ubicaciones mencionadas
todas_ubicaciones = ','.join(resultados_df['Ubicacion Encontrada']).split(',')

# Se eliminan espacios adicionales y se convierten a minúsculas
todas_ubicaciones = [ubicacion.strip().lower() for ubicacion in todas_ubicaciones]

# Contar las frecuencias
contador_frecuencias = Counter(todas_ubicaciones)

# Se convierte el contador en un DataFrame
frecuencia_df = pd.DataFrame.from_dict(contador_frecuencias, orient='index', columns=['Frecuencia'])
frecuencia_df.reset_index(inplace=True)
frecuencia_df.rename(columns={'index': 'Ubicacion'}, inplace=True)

# Frecuencia relativa
frecuencia_df['Frecuencia Relativa'] = frecuencia_df['Frecuencia'] / total_comentarios

# Descendente
frecuencia_df = frecuencia_df.sort_values(by='Frecuencia Relativa', ascending=False)

# Grafica
plt.figure(figsize=(10, 6))
plt.bar(frecuencia_df['Ubicacion'], frecuencia_df['Frecuencia Relativa'], color='skyblue')
plt.xlabel('Ubicación')
plt.ylabel('Frecuencia Relativa de Menciones')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



#----------------------------Resultados de Análisis de sentimeintos--------------------------------------------------------------

dataset = pd.read_csv("Resultados.csv", encoding='latin-1')
comentarios = dataset['Comentario Original']
subcomentarios = comentarios.apply(divide_comentarios)


# DataFrame
resultados_subcomentarios = pd.DataFrame(columns=['Subcomentario', 'roberta_neg', 'roberta_neu', 'roberta_pos'])

# Itera sobre los subcomentarios y calcula los resultados
for i, subcomentarios in tqdm(enumerate(subcomentarios), total=len(subcomentarios)):
    for subcomentario in subcomentarios:
        roberta_result = polarity_scores_roberta(subcomentario)
        
        resultados_subcomentarios = resultados_subcomentarios.append({
            'Subcomentario': subcomentario,
            'roberta_neg': roberta_result['roberta_neg'],
            'roberta_neu': roberta_result['roberta_neu'],
            'roberta_pos': roberta_result['roberta_pos'],
        }, ignore_index=True)

# Nuevo archivo CSV
resultados_subcomentarios.to_csv('resultados_subcomentarios.csv', index=False)

#-#-#

# Cargar el archivo 
resultados_subcomentarios = pd.read_csv('resultados_subcomentarios.csv')

# Gráfica de barras
plt.figure(figsize=(10, 6))
plt.hist(resultados_subcomentarios['roberta_neg'], bins=20, color='red', alpha=0.5, label='Negativo')
plt.hist(resultados_subcomentarios['roberta_neu'], bins=20, color='gray', alpha=0.5, label='Neutro')
plt.hist(resultados_subcomentarios['roberta_pos'], bins=20, color='green', alpha=0.5, label='Positivo')

plt.xlabel('Polaridad del Sentimiento')
plt.ylabel('Frecuencia')
plt.legend(loc='upper right')
plt.show()

#---------------Analisis de actividades--------------------------------------------#

palabras_clave = ['snorkel', 'swimming', 'paddle', 'diving', 'watching', 'scuba', 'experience', 'luxury', 'boat', 'vacation', 'excursion', 'view']

resultados_subcomentarios['Palabra_clave'] = ''

# Itera sobre las palabras clave y asigna la palabra clave a su subcomentario
for palabra in palabras_clave:
    resultados_subcomentarios.loc[resultados_subcomentarios['Subcomentario'].str.contains(palabra, case=False), 'Palabra_clave'] = palabra


resultados_subcomentarios.to_csv('resultados_subcomentarios_con_palabra_clave.csv', index=False)

# Lee el DataFrame 
resultados_subcomentarios = pd.read_csv('resultados_subcomentarios_con_palabra_clave.csv')

# Filtra subcomentarios con palabra clave no vacía
subcomentarios_con_palabra = resultados_subcomentarios[resultados_subcomentarios['Palabra_clave'].notnull()]

# Se configura el estilo
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Gráfico de barras apiladas 
sentimientos_palabras_melted = pd.melt(subcomentarios_con_palabra, id_vars='Palabra_clave', value_vars=['roberta_neg', 'roberta_neu', 'roberta_pos'], var_name='Sentimiento')
sns.barplot(data=sentimientos_palabras_melted, x='Palabra_clave', y='value', hue='Sentimiento')

plt.xlabel('Palabra Clave')
plt.ylabel('Promedio de Sentimiento')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentimiento', loc='upper right')

plt.tight_layout()
plt.show()

#---------------------Analisis con animales--------------------------

resultados_subcomentarios_con_palabra_clave = pd.read_csv("resultados_subcomentarios_con_palabra_clave.csv", encoding='latin-1')

nombres_animales_lista = ["turtle", "whale", "shark", "starfish", "fish", "dolphin", "octopus", "jellyfish", "seahorse"]

resultados_subcomentarios_con_palabra_clave['Mencion_animales'] = resultados_subcomentarios_con_palabra_clave['Subcomentario'].apply(lambda subcomentario: [nombre_animal for nombre_animal in nombres_animales_lista if nombre_animal in subcomentario.lower()])

resultados_expandidos = resultados_subcomentarios_con_palabra_clave.explode('Mencion_animales')

resultados_expandidos.to_csv('resultados_subcomentarios_con_palabra_clave_y_animales.csv', index=False)


subcomentarios_con_mencion_animales = resultados_expandidos[resultados_expandidos['Mencion_animales'].notnull()]

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Gráfico de barras apiladas 
sentimientos_animales_melted = pd.melt(subcomentarios_con_mencion_animales, id_vars='Mencion_animales', value_vars=['roberta_neg', 'roberta_neu', 'roberta_pos'], var_name='Sentimiento')
sns.barplot(data=sentimientos_animales_melted, x='Mencion_animales', y='value', hue='Sentimiento')

plt.xlabel('Mención de Animales Marinos')
plt.ylabel('Promedio de Sentimiento')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentimiento', loc='upper right')


plt.tight_layout()
plt.show()






























    