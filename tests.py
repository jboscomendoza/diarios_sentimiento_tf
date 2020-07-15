import npl_diarios
import pandas as pd


mi_diario = "Hoy estoy feliz. Todo me sale bien. Es perfecta mi vida. Todo es alegria, todo es felicidad."
sec_diario  = npl_diarios.crear_secuencia(mi_diario)
pred_diario = npl_diarios.crear_prediccion(sec_diario)
# El resultado debe ser:
# {'sentimiento': 'Positivo', 'values': [0.58438, 0.0, 9e-05]}


mult_diarios = list(pd.read_csv("diarios_muestra.csv")["Diario"])
sec_mult_diarios  = [npl_diarios.crear_secuencia(i) for i in mult_diarios]
pred_mult_diarios = [npl_diarios.crear_prediccion(i) for i in sec_mult_diarios]
[print(i) for i in pred_mult_diarios]
# El resultado debe ser:
# {'sentimiento': 'Positivo', 'values': [0.01469, 0.0, 0.00019]}
# {'sentimiento': 'Negativo', 'values': [0.0, 0.50532, 0.0025]}
# {'sentimiento': 'Positivo', 'values': [0.48918, 0.0, 6e-05]}
# {'sentimiento': 'Negativo', 'values': [0.0002, 0.00561, 4e-05]}
# {'sentimiento': 'Ambivalente', 'values': [0.0, 0.00309, 0.02923]}
# {'sentimiento': 'Positivo', 'values': [0.23412, 0.0, 2e-05]}
# {'sentimiento': 'Negativo', 'values': [0.00011, 0.03156, 1e-05]}
# {'sentimiento': 'Negativo', 'values': [1e-05, 0.11378, 0.0]}
# {'sentimiento': 'Negativo', 'values': [0.0, 0.65722, 4e-05]}
# {'sentimiento': 'Negativo', 'values': [0.0, 0.01194, 0.00036]}
# {'sentimiento': 'Negativo', 'values': [0.00024, 0.00178, 2e-05]}
# {'sentimiento': 'Positivo', 'values': [0.0848, 0.0, 0.00011]}
# {'sentimiento': 'Positivo', 'values': [0.7395, 0.0, 2e-05]}
# {'sentimiento': 'Ambivalente', 'values': [0.00017, 0.00019, 0.00071]}
