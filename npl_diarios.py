import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Modelo entrenado con tagged.csv
modelo = load_model("modelo_octubre.h5")

# Tokenizer con un vocabulario máximo de 6000, tomado de tagged.csv
with open('tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)

def crear_secuencia(texto, tokenizer=tokenizer, largo_max=350):
  """
  Convierte un str en un vector numerico para poder realizar predicciones
  """
  secuencia = tokenizer.texts_to_sequences([texto])
  secuencia = pad_sequences(secuencia, padding="post", maxlen=largo_max)
  return(secuencia)

def crear_prediccion(secuencia, modelo=modelo, precision=5):
  """
  A partir de un vector numérico devuelve un diccionario con el sentimiento 
  predicho y los valores para cada tipo de sentimiento en este orden:
  [Positivo, Negativo, Ambivalente]
  """
  raw_predict = modelo.predict(secuencia)
  predict = list(raw_predict[0])
  clase = predict.index(max(predict))
  if clase == 0:
    sentimiento = "Positivo"
  elif clase == 1:
    sentimiento = "Negativo"
  elif clase == 2:
    sentimiento = "Ambivalente"
  predict = [round(i, 5) for i in predict]
  return({"sentimiento":sentimiento, "values":predict})
