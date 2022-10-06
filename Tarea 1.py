#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
import scipy
from scipy import stats
from scipy.stats import norm

data = pd.read_csv('speed_dating.csv')


# In[2]:


data.head(10)


# # 1. Población y Muestra. Para empezar, las siguientes preguntas no requieren nada de programación sino únicamente que empieces por establecer algunas definiciones importantes que utilizaras en tu reporte para los ejecutivos de Instagram.
# 
# # Basado en la descripción de los datos que tienes disponibles, explica claramente en menos de 300 palabras: ¿cuál es la población relevante para tu estudio? Para guiar esta respuesta considera: ¿crees que los resultados son representativos de todos los usuarios de Instagram o solo de un subconjunto? ¿Crees que dada la restricción de tu muestra disponible, el análisis carece de relevancia? Justifica de forma clara y concisa tu respuesta.

# Dado que la muestra está integrada principalmente por estudiantes de posgrado, considero que puede tener un sesgo en la importancia que las personas encuestadas le dan a ciertos aspectos para la elección de una pareja, por ejemplo, la importancia de la inteligencia de la persona con la que se tuvo la cita, creo que el peso que los integrantes de esta muestra le pueden dar a aspectos como éste puede ser mayor respecto a algún otro subconjunto de usuarios de instagram. También considero que la edad es un factor que influye en las respuestas de las personas que integran esta muestra, dado que representa un intervalo acotado de entre todos los usuarios de la red social, creo que una mejor muestra puede estar integreada por personas cuyo rango de edad sea más amplio.

# # ¿Es tu muestra i.i.d.? En menos de 150 palabras describe por qué sí o por qué no. [Nota: podrás asumir que sí es i.i.d. en todo el resto de la tarea, pero aquí debes hacer esta reflexión.]

# Considero que no es es idénticamente distribuida por que la muestra no se ha elegido de una forma aleatoria pues se acotó a las personas que cursan un posgrado en cierta universidad.
# Además, creo que no es independiente pues una persona pudo haber llenado varias veces el cuestionario y tener las mismas respuestas en la ponderación de la importancia que le da a ciertas características.

# # Imagina que tu equipo te indica que tienes la oportunidad de recolectar más información que la descrita en la Tabla 1, 5 variablas más como máximo. Pero tu justificación para estas variables debe convencer a los ejecutivos de IG que van a aprobar el pago de esto. ¿Qué información adicional considerarías importante recolectar? y ¿por qué? Justifica tu respuesta (máximo 200 palabras).

# Las variables que incluiría son: 
# 
# 1.- common_friends: una variable que mide cuántos amigos en común tienen. Debido a que puede ser un factor importante para la socialización, en particular para el proceso de elegir una pareja.
# 
# 2.- like_attitude: una variable dummie que considera si le gusto o no la actitud de la persona con la que se tuvo la cita. Debido a que la percepción de la actitud de una persona es un factor importante para decidir salir con alguien.
# 
# 3.- comfortable: una variable dummie que considera si la persona que llena la encuesta se sintió cómoda durante la cita.
# 
# 4.- shared_tastes: una variable que mide qué tan importante es para el individuo que su pareja tenga gustos similares.

# # El equipo de Data Core te ofrece una nueva base de datos que agrupa un experimento similar que realizó Facebook App (plataforma que, al igual que Instagram, pertenece al conglomerado de Meta) cuando implementó Facebook Dating. Esta otra base de datos contiene más variables de las que se midieron en el estudio que tu equipo realizó (i.e. la base descrita en la Tabla 1) y el número de participantes es casi el triple. Por otra parte, el estudio de Facebook App se realizó en 2018 y participaron individuos que utilizaban Facebook e Instagram. Ante esto, debes decidir si quieres: (i) utilizar solamente la base de datos que generó tu equipo para realizar tu análisis; (ii) utilizar solamente la base de datos que generó Facebook App para Facebook Dating; o (iii) unir ambas bases y generar una sola fuente de datos. Explica la decisión de tu respuesta.
# 

# Usaría la unión de ambas bases de datos. Mi decisión se basa en el hecho de que la base que utilizó Facebook puede no tener las limitaciones de las que hablo en el inciso b) de esta pregunta, es decir, es posible que la muestra tomada por facebook considere una muestra más amplia en cuanto a edad y no esté acotada a un solo grupo (como la original que está a acotada a un grupo de estudiantes). Además, el hecho de que haya sido en otro año también enriquecería el estudio y podría considerarse como una variable más de nuestra base de datos

# ## 2. Análisis Descriptivo. El equipo de programadores te pide que revises las características generales de la base de datos.

# ## Los programadores consideran que algunas de las variables más importantes para analizar son female, age, age o, attractive important, sincere important, intelli- gence important, funny important, ambition important, shared interests important, decision y match. Ante esto, se te pide reportar una tabla descriptiva con los valores estimados para la media, mediana, desviación estándar, valor mínimo y valor máximo.

# In[3]:


data[['female', 'age', 'age_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important', 'decision', 'match']].describe()


# ## Lleva a cabo las siguientes gráficas:
# 
# ## (i) histograma de age

# In[4]:


intervalos = range(min(data['age']), max(data['age']), 2)


# In[5]:


plt.hist(x = data.age, bins = intervalos, color = '#F2AB6D', rwidth = 0.85)
plt.title('Histograma de edades de la muestra')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.xticks(intervalos)
plt.xticks(rotation=80)
plt.show()


# ## (ii) scatterplot de attractive important vs intelligence important

# scatterplot de attractive important vs intelligence important

# In[6]:


fig = px.scatter(data, x=data['attractive_important'], y=data['intellicence_important'], 
                 opacity=0.8, color_discrete_sequence=['blue'])

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')
# Set figure title
fig.update_layout(title=dict(text="scatter plot: attractive importante vs intelligence important", 
                             font=dict(color='black')))
# Update marker size
fig.update_traces(marker=dict(size=3))
fig.show()


# ## (iii) misma gráfica que (ii) pero agregando dos líneas de regresión, una para female = 1 y otra para female = 0.

# In[7]:


fig = px.scatter(data, x="attractive_important", y="intellicence_important", facet_col="female", trendline = 'ols')
fig.show()


# ## Describe brevemente (menos de 100 palabras) qué aprendiste de cada gráfica.

# Las gráficas muestran, una linea de regresión de pendiente negativa, es decir que en ambos casos, para hombres y mujeres, a mayor importancia que se le dé a que la pareja sea atractiva, se le da una menor importancia a la inteligencia de la pareja, esta relación entre la importancia de estos dos atributos está aún más marcada en los homrbes (pendiente de la línea de regresión más negativa). 

# ## 3. Pruebas de hipótesis.
# 
# ## Para invertir en el desarrollo de la aplicación, los ejecutivos de Meta necesitan tener evidencia de que el formato de Speed Dating que se ha planteado funciona. La medida de éxito que han establecido es el porcentaje de citas que concluyen en una decisión de aceptar salir con su pareja respectiva (decision = 1). Se espera que ese porcentaje vaya a ser de al menos de 30%. Asumiendo que cuentas con una muestra representativa, contesta las siguientes preguntas:
# 
# ## Plantea la prueba de hipótesis relevante para lo que te piden los ejecutivos.

# H<sub>0</sub> : $\mu \geq  0.30$
# 
# H<sub>a</sub> : $\mu < 0.30$
# 
# Donde $\mu$ es la media poblacional.

# ## Con la información disponible, ¿Es posible observar directamente el elemento que planteas en la hipóteiss nula? Explica tu pensamiento usando menos de 100 palabras.

# Sí es posible, dado que el porcentaje de personas encuestadas con decision=1 entre el número total de personas encuestadas es el mismo que el promedio de la variable decision: dicho promedio es el cociente de la suma de unos y ceros en el numerador, entre el número total de encuestados. El numerador suma exactamente tantos unos como personas con decision=1, mientras que el denominador es el mismo.

# In[8]:


len(data[data['decision'] == 1]) / len (data) == data.decision.mean()


# ## Utilizando los datos evalúa la prueba de hipótesis utilizando una significancia de 10%. Emplea el estadístico t y un valor crítico para llevar a cabo la evaluación (i.e. no uses valor-p o intervalos de confianza).

# El nivel de significancia es $\alpha = 0.1$
# 
# Vamos a utilizar el estadístico $t$:
# 
# $t = \frac{\bar{X}- \mu}{\sqrt{\frac{s^2}{n}}}$

# In[33]:


s2 = np.var(data['decision'])
x_barra = np.mean(data['decision'])
n =len(data)
t = (x_barra - 0.3) / (np.sqrt(s2 / n))
valor_tabla = scipy.stats.t.ppf(0.1, n-1)
print('el valor del estadístico t es: ', t)
print(f'el valor crítico con  nivel de significancia de 0.1 es {valor_tabla}')
x_barra


# No rechazamos H<sub>0</sub> cuando el valor calculado del estadístico t es mayor que el valor crítico, como este es el caso no rechazamos H<sub>0</sub>.

# ## En menos de 200 palabras, redacta una nota ejecutiva para los directores de Meta empleando un lenguaje simple (i.e. no muchos tecnicismos) acerca de la conclusión de tu ejercicio. Considera que los ejecutivos de Meta desconocen por completo el lenguaje estadístico, por lo que deberás pensar en cómo exponer tus resultados y sus implicaciones en un lenguaje claro.

# De acuerdo al análsis estadístico realizado sobre la muestra dada, podemos concluir con un 90% de certeza que el formato de speed dating planteado funciona. Es decir, el porcentaje de personas que han tenido una cita y aceptan salir con la persona con la que tuvieron la cita es de al menos 30%.

# # 4. Bootstrap. El equipo de analítica piensa que la edad de los individuos está correlacionada con la importancia que le dan a que la pareja sea atractiva.

# # (a) Calcula la correlación entre las variables edad y attractive important. Haz un scatterplot entre estas dos variables e incluye en esta gráfica el resultado de una regresión simple.

# In[10]:


# el coeficiente de correlación de Pearson
np.corrcoef(data['age'], data['attractive_important'])[0,1]


# In[11]:


cov_mat = data.cov()
cov_edad_attimp = cov_mat.loc['age', 'attractive_important']
std_edad = data['age'].std()
std_attimp = data['attractive_important'].std()
r = cov_edad_attimp / (std_edad *std_attimp) 
r


# In[12]:


fig = px.scatter(data, x="age", y="attractive_important", trendline = 'ols')
fig.update_layout(title=dict(text="scatter plot: age vs attractive_important", 
                             font=dict(color='black')))
fig.show()


# # Ahora te interesa evaluar si la correlación entre ambas variables es igual a cero. Plantea la hipótesis nula y alternativa.

# H<sub>0</sub> : $\rho = 0$    No existe una correlación 
# 
# H<sub>a</sub> : $\rho \neq 0$    Existe una correlación.
# 
# H<sub>a</sub> es en realidad $\rho > 0$ y $\rho < 0$

# # Empleando Bootstrap, evalúa la prueba de hipótesis para calcular la varianza del estimador. Asume que el estimador se distribuye como una normal. Utiliza submuestras de tamaño 6,760 y 1,000 repeticiones.

# Definimos las siguientes dos funciones:
# calcular_estadistico nos devuelve el coeficiente de correlación, mientras que bootstraping nos devuelve un array que contiene los valores estimados del estimador para cada una de las muestras

# In[13]:


def calcular_estadistico(x):
    '''
    Función para calcular el estadístico de interés.
    
    Parameters
    ----------
    x : numpy array
         valores de la muestra.
         
    Returns
    -------
    estadístico: float
        valor del estadístico.
    '''
    estadistico = np.corrcoef(x['age'], x['attractive_important'])[0,1]
    
    return(estadistico)


# In[14]:


def bootstraping(data, fun_estadistico, n_iteraciones):
    '''
    Función para calcular el valor del estadístico en múltiples muestras generadas
    mediante muestreo repetido con reposición (bootstrapping).
    
    Parameters
    ----------
    x : numpy array
         valores de la muestra.
 
    fun_estadistico : function
        función que recibe como argumento una muestra y devuelve el valor
        del estadístico.
        
    n_iteraciones : int
        número iteraciones (default `9999`).
        
    Returns
    -------
    distribuciones: numpy array
        valor del estadístico en cada muestra de bootstrapping.
    '''
    x = data.index.tolist()
    n = len(x)
    dist_boot = np.full(shape=n_iteraciones, fill_value=np.nan)
    
    for i in tqdm(range(n_iteraciones)):
        resample = data.loc[np.random.choice(x, size=6760, replace=True)]
        dist_boot[i] = fun_estadistico(resample)
        
    return dist_boot


# Obtenemos el array de los valores estimado de cada una de las muestras de tamaño 6760, haciendo 1000 repeticiones, para después calcular la varianza de este array:

# In[15]:


dist_boot = bootstraping(data = data[['age', 'attractive_important']], fun_estadistico = calcular_estadistico, n_iteraciones = 1000)


# In[16]:


print(f'La varianza del estimador, con los coeficietes de correlación obtenidos con el método de bootstrapping es: {np.var(dist_boot)}')


# # Repite el ejercicio anterior usando submuestras de tamaño 3,000. De no haber ajustado la varianza, ¿Cuál es el valor estimado de la varianza que hubieras obtenido?

# In[17]:


def bootstraping_d(data, fun_estadistico, n_iteraciones):
    '''
    Función para calcular el valor del estadístico en múltiples muestras generadas
    mediante muestreo repetido con reposición (bootstrapping).
    
    Parameters
    ----------
    x : numpy array
         valores de la muestra.
 
    fun_estadistico : function
        función que recibe como argumento una muestra y devuelve el valor
        del estadístico.
        
    n_iteraciones : int
        número iteraciones (default `9999`).
        
    Returns
    -------
    distribuciones: numpy array
        valor del estadístico en cada muestra de bootstrapping.
    '''
    x = data.index.tolist()
    n = len(x)
    dist_boot = np.full(shape=n_iteraciones, fill_value=np.nan)
    
    for i in tqdm(range(n_iteraciones)):
        resample = data.loc[np.random.choice(x, size=3000, replace=True)]
        dist_boot[i] = fun_estadistico(resample)
        
    return dist_boot


# In[18]:


dist_boot_d = bootstraping_d(data = data[['age', 'attractive_important']], fun_estadistico = calcular_estadistico, n_iteraciones = 1000)


# In[19]:


print(f'La varianza del índice de correlación usando bootstrapping con submuestras de tamaño 3000 es: {np.var(dist_boot_d)}, mientras que el valor estimado de la varianza del índice de correlación usando bootstrapping con submuestras se tamaño 6760 es: {np.var(dist_boot)} ')


# # Grafica un histograma con las 1,000 repeticiones del inciso (c). ¿Parece formarse una distribución normal?

# In[34]:


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7,3.3))
ax.hist(dist_boot, bins=30, density=True, color='#3182bd', alpha=0.5)
ax.set_title('Distribución de bootstrap')
ax.set_xlabel('Coeficiente de correlación')
ax.set_ylabel('densidad');


# Sí parece una distribución normal

# # Sin suponer que el estimador se distribuye como una normal, calcula el valor-p que hubieras obtenido con las simulaciones del inciso (c) al evaluar la prueba de hipótesis.

# Con los valores obtenidos para la correlación de las 100 muestras del inciso c, vamos a calcular el p-valor
# como el número de coeficientes de correlación mayores (en valor absoluto) a 0 entre el número total de submuestras (1000)

# In[21]:


(sum(abs(dist_boot) > 0)) / 1000


# Dado que el p-valor obtenido nos dio igual a 1 no podemos rechazar la hipótesis nula, lo que sugiere queno existe una correlación entre la edad y la imporancia que se le da al que una persona sea atractiva.

# # Más Pruebas de Hipótesis. Como parte de tu trabajo, el equipo de desarrolladores quiere que des evidencia a favor o en contra de las siguientes afirmaciones.
# 
# # Utiliza los datos que consideres necesarios para: 
# 
# # (i) plantear la prueba de hipótesis pertinente para evaluar cada afirmación; 
# 
# # (ii) obtener el valor-p de cada prueba;  
# 
# # (iii) redacta en un breve enunciado (menos de 100 palabras para los 3 tests) las conclusiones a las que llegas.

# # En promedio, el grupo de estudiantes female = 1 le dan mayor importancia a la inteligencia de la pareja que el grupo female = 0.

# Obteniendo la media, desviación estándar y tamaño de ambas submuestras(female =1 y female=0) 

# In[22]:


data_f = data[data['female'] == 1]
data_m = data[data['female'] == 0]
m_f = data_f['intellicence_important'].mean()
m_m = data_m['intellicence_important'].mean()
var_f = data_f['intellicence_important'].var()
var_m = data_m['intellicence_important'].var()
n_f = len(data_f)
n_m = len(data_m)


# Planteando las hipótesis nula y alternativa:
# 
# H<sub>0</sub> : $\mu_f - \mu_m \leq 0$ 
# 
# H<sub>a</sub> : $\mu_f -\mu_M > 0$ 

# Usaremos el estadístico $t$:
# 
# $t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$

# In[23]:


# Calculando el estadístico t:
d = ((var_f)/(n_f)) + ((var_m)/(n_m)) 
t = (m_f - m_m)/ np.sqrt((d))
t


# Calculamos el valor-p, esto es: $P(t<7.2925)$

# In[24]:


print(f'El p-valor es: {norm.cdf(7.292530, 100, 10)}')
norm.cdf(7.292530, 100, 10) < 0.01


# Al suponer cierta nuestra hipótesis nula, usando un nivel de significancia de $ \alpha = 0.01$ y como: $p-valor < \alpha$, podemos rechazar esta hipótesis, lo que sugiere que, en promedio el grupo con female=1 le da mayor importancia a la inteligencia que el grupo con female = 0

# # En promedio, el grupo de estudiantes female = 1 le da menor importancia al atractivo físico de la pareja que el grupo female = 0.

# Planteando las hipótesis nula y alternativa:
# 
# H<sub>0</sub> : $\mu_f - \mu_m \geq 0$ 
# 
# H<sub>a</sub> : $\mu_f -\mu_M < 0$ 

# In[25]:


data_f = data[data['female'] == 1]
data_m = data[data['female'] == 0]
m_f = data_f['attractive_important'].mean()
m_m = data_m['attractive_important'].mean()
var_f = data_f['attractive_important'].var()
var_m = data_m['attractive_important'].var()
n_f = len(data_f)
n_m = len(data_m)
d = ((var_f)/(n_f)) + ((var_m)/(n_m)) 
t = (m_f - m_m)/ np.sqrt((d))
t


# Calculamos el valor-p, esto es: $P(-30.69967400777379 < t)$

# In[36]:


print(norm.cdf(-30.69967400777379, 100, 10))
norm.cdf(7.292530, 100, 10)  < 0.01


# Al suponer cierta nuestra hipótesis nula, usando un nivel de significancia de $ \alpha = 0.01$ y como: $p-valor < \alpha$, tenemos evidencia para rechazar esta hipótesis, lo que sugiere que, en promedio el grupo con female=1 le da menor importancia a el atractivo que el grupo con female = 0

# # La correlación entre las variables edad y attractive important es menor (en valor absoluto) para el grupo female = 1 que para el grupo female = 0.

# Planteando las hipótesis nula y alternativa:
# 
# H<sub>0</sub> : $|\rho_f| - |\rho_m| \geq 0$ 
# 
# H<sub>a</sub> : $|\rho_f| - |\rho_m| < 0$ 

# In[ ]:





# In[ ]:





# In[ ]:




