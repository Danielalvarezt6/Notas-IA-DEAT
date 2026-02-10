---
layout: default
title: Notas de Inteligencia Artificial - Daniel Alvarez
math: true
---

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Notas Inteligencia Artificial

> **Nota sobre el contenido:** Este material fue sintetizado con el apoyo de **NotebookLM**, tomando como base mis apuntes personales y las presentaciones utilizadas en las sesiones de clase.

## 1. Introducción a la Inteligencia Artificial

### 1.1 Categorías de la IA
Para definir la IA, nos basamos en cuatro enfoques clásicos (según Russell & Norvig):
1.  **Pensar como humanos:** Enfoque de la ciencia cognitiva.
2.  **Actuar como humanos:** La Prueba de Turing.
3.  **Pensar racionalmente:** Leyes del pensamiento (lógica).
4.  **Actuar racionalmente:** El enfoque del **Agente Racional**.

En este curso, nos centramos en la cuarta categoría: la **racionalidad**. Esto implica hacer "lo correcto" para maximizar la utilidad futura, considerando las limitaciones de percepción y cómputo.

---

## 2. Agentes Inteligentes

### 2.1 Definición y Estructura (Modelo PEAS)
Un agente es cualquier entidad que percibe su entorno a través de **sensores** y actúa sobre él mediante **actuadores**. Para diseñar un agente racional, definimos el marco **PEAS**:

*   **P (Performance):** Medida de desempeño (objetivo a maximizar, ej. puntos por basura recogida).
*   **E (Environment):** Entorno donde opera.
*   **A (Actuators):** Mecanismos para actuar (ej. ruedas, motor).
*   **S (Sensors):** Mecanismos para percibir (ej. cámaras, teclado).

![Diagrama Agente-Entorno](https://acumbamail.com/blog/wp-content/uploads/2025/01/agente-ia.png)

La función del agente mapea el historial de percepciones a acciones:
$$f: P^* \rightarrow A$$

### 2.2 Clasificación de Entornos
El diseño del agente depende drásticamente de las propiedades del entorno:

| Propiedad | Definición |
| :--- | :--- |
| **Observable vs. Parcialmente Observable** | ¿Los sensores detectan el estado completo del mundo? |
| **Determinista vs. Estocástico** | ¿El siguiente estado está determinado puramente por el estado actual y la acción? (Si hay azar, es estocástico). |
| **Estático vs. Dinámico** | ¿El entorno cambia mientras el agente está "pensando"? |
| **Discreto vs. Continuo** | ¿Hay un número finito de estados/acciones o son valores continuos? |
| **Episódico vs. Secuencial** | ¿La acción actual afecta decisiones futuras? (En episódico, no). |

### 2.3 Tipos de Agentes
Según su complejidad interna, los agentes se clasifican en:
*   **Agentes Reactivos Simples:** Actúan solo según la percepción actual (reglas *if-then*).
*   **Agentes basados en Modelos:** Mantienen un estado interno para rastrear aspectos del mundo que no pueden ver.
*   **Agentes basados en Metas (Goal-based):** Actúan para alcanzar un estado final deseado.
*   **Agentes basados en Utilidad:** Intentan maximizar una función de "felicidad" o preferencia numérica.

---

## 3. Teoría del Aprendizaje (Learning Theory)

### 3.1 Aprendizaje Supervisado
Es una técnica de *Machine Learning* donde el modelo aprende a partir de un conjunto de datos **etiquetados**. Estos datos proporcionan una **"Verdad Fundamental" (Ground Truth)**, que actúa como un guía o "profesor" para el algoritmo.

* **El objetivo:** Crear una función (hipótesis) $$h$$ que se aproxime a una función desconocida $$f$$ (la realidad), tal que $$h(x) \approx f(x)$$, utilizando un conjunto de entrenamiento $$D$$.
* **Funcionamiento:** El modelo hace predicciones, mide el error respecto a la etiqueta real (usando una **función de pérdida**) y ajusta sus parámetros para minimizar dicha discrepancia.

### 3.2 Error y Generalización
Para saber si el modelo aprende, distinguimos dos tipos de error:
1.  **Error en muestra ($$E_{in}$$):** El error calculado sobre los datos de entrenamiento.
2.  **Error fuera de muestra ($$E_{out}$$):** El error sobre datos nuevos (**generalización**).



### 3.3 Overfitting vs. Underfitting
El éxito del aprendizaje supervisado depende de encontrar el equilibrio entre la complejidad del modelo y la cantidad de datos.

* **Underfitting (Subajuste):** Ocurre cuando el modelo es demasiado simple para capturar la estructura de los datos.
    * *Resultado:* $$E_{in}$$ alto y $$E_{out}$$ alto. El modelo ni siquiera aprende los datos de entrenamiento.
* **Overfitting (Sobreajuste):** Ocurre cuando el modelo es demasiado complejo y empieza a memorizar el **ruido** y detalles irrelevantes del entrenamiento.
    * *Resultado:* $$E_{in}$$ muy bajo, pero $$E_{out}$$ muy alto. El modelo falla al generalizar con datos nuevos.
 
![Diagrama](https://datahacker.rs/wp-content/uploads/2021/11/Picture3-1536x522.jpg)

> **Concepto Crítico (Desigualdad de Hoeffding y Dimensión VC):**
> El aprendizaje es factible si podemos garantizar que $$E_{in} \approx E_{out}$$. Esto depende de la complejidad del modelo, medida por la **Dimensión VC ($$d_{VC}$$)**.
>
> La regla práctica es que necesitamos **10 veces más datos que grados de libertad (parámetros)** ($$N > 10 \cdot d_{VC}$$) para evitar el sobreajuste y garantizar generalización.

---

## 4. Modelos Lineales y Optimización

### 4.1 Regresión Lineal
Buscamos predecir un valor real $$y$$. El modelo es una combinación lineal de los pesos $$w$$ y las características $$x$$:
$$h_w(x) = w_0 + w_1x_1 + \dots + w_n x_n = w^T x$$

Para encontrar los mejores pesos, minimizamos el **Error Cuadrático Medio (MSE)**:
$$J(w) = \frac{1}{N} \sum_{i=1}^{N} (h_w(x^{(i)}) - y^{(i)})^2$$

**Solución Analítica (Ecuación Normal):**
$$w = (X^T X)^{-1} X^T Y$$
*Nota: Si la matriz es muy grande, invertirla es costoso computacionalmente.*

### 4.2 Descenso del Gradiente (Gradient Descent)
Es un algoritmo de optimización iterativo de primer orden utilizado para encontrar los mínimos locales de una función diferenciable. Es el método estándar para entrenar modelos cuando la solución analítica es computacionalmente intratable.

* **Fundamento Matemático:** El algoritmo se basa en la observación de que una función multivariable $$J(w)$$ disminuye más rápidamente si se avanza en la dirección del **gradiente negativo** del punto actual.
* **El Gradiente ($$\nabla J(w)$$):** Es un vector que contiene todas las derivadas parciales de la función de costo. Matemáticamente, este vector apunta hacia la dirección de mayor crecimiento de la función; por lo tanto, nos movemos en la dirección opuesta ($$-\nabla$$) para minimizar el error.
* **Objetivo:** Determinar los parámetros óptimos $$w$$que minimizan la función de costo$$J(w)$$, convergiendo iterativamente hacia un punto donde el gradiente es cero (o muy cercano a cero).

**Algoritmo de Actualización:**
Se repite el siguiente paso hasta satisfacer un criterio de parada (convergencia):

$$w_j \leftarrow w_j - \eta \frac{\partial J(w)}{\partial w_j}$$

Donde:
* $$w_j$$: Es el peso o parámetro a actualizar.
* $$\eta$$ (Eta): Es la **tasa de aprendizaje (learning rate)**, un hiperparámetro escalar que determina la magnitud del paso en cada iteración.
* $$\frac{\partial J(w)}{\partial w_j}$$: Es la derivada parcial (el gradiente) respecto al peso $$w_j$$.

![Descenso de gradiente](https://assets.ibm.com/is/image/ibm/ICLH_Diagram_Batch_03_21-AI-ML-GradientDescent:16x9?fmt=png-alpha&dpr=on%2C1.25&wid=960&hei=540)

*Bloque de código conceptual (basado en notas):*
```python
# Pseudocódigo de Descenso de Gradiente (Batch)
w = inicializar_pesos()
for epoch in range(max_epochs):
    predicciones = dot(X, w)
    error = y - predicciones
    gradiente = - (1/N) * dot(X.T, error) 
    w = w - tasa_aprendizaje * gradiente
    
    if norma(gradiente) < tolerancia:
        break
return w
```

---
# 5. Clasificación y Regularización

### 5.1 Clasificación Lineal vs. Regresión Logística
A diferencia de la regresión lineal, la clasificación predice etiquetas discretas o probabilidades de pertenencia a una clase.

**Tabla Comparativa:**

| Característica | Regresión Lineal | Regresión Logística |
| :--- | :--- | :--- |
| **Variable Objetivo** | Continua (ej. $$24.5^{\circ}C$$) | Categórica / Probabilidad (ej. Spam/No Spam) |
| **Rango de Salida** | $$(-\infty, +\infty)$$|$$[0, 1]$$ |
| **Relación** | Lineal | No lineal (Sigmoide) |
| **Función de Costo** | MSE (Convexa para regresión) | Entropía Cruzada / Log Loss (Convexa para clasificación) |

**El Modelo Logístico (Sigmoide):**
Usamos la función Sigmoide para "aplastar" la salida lineal entre 0 y 1, interpretándola como una probabilidad:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde $$z = x^T w + b$$.

**Función de Costo (Log Loss):**
El MSE no es adecuado aquí porque generaría una función "no convexa" (muchos mínimos locales). Usamos *Maximum Likelihood Estimation*:

$$J(w) = - \frac{1}{M} \sum_{i=1}^{M} [a^{(i)} \log(\hat{a}^{(i)}) + (1 - a^{(i)}) \log(1 - \hat{a}^{(i)})]$$

### 5.2 Regularización (Controlando el Overfitting)
Para evitar que el modelo "memorice" el ruido de los datos de entrenamiento (**Overfitting**), penalizamos los pesos grandes. Esto aplica el principio de la **Navaja de Ockham**: ante dos modelos con error similar, preferimos el más simple.

Nueva función de costo a minimizar:

$$J_{reg}(w) = J_{original}(w) + \lambda \cdot R(w)$$

**Tipos de Regularización:**

1.  **Regularización L2 (Ridge):**
    * Penalización: $$\lambda \sum w_j^2$$
    * **Efecto:** Reduce todos los pesos uniformemente hacia cero (weight decay), pero raramente los hace cero exacto.
    * **Uso:** Cuando todas las variables aportan algo de información.

2.  **Regularización L1 (Lasso):**
    * Penalización: $$\lambda \sum \|w_j\|$$
    * **Efecto:** Puede forzar a que algunos pesos sean **exactamente cero**.
    * **Uso:** Funciona como **selección de características** automática (elimina variables irrelevantes).

> **Hiperparámetro $$\lambda$$:** Controla la fuerza de la penalización.
> * $$\lambda$$muy grande$$\to$$ Underfitting (modelo demasiado simple).
> * $$\lambda = 0$$ $$\to$$ Regresión estándar (riesgo de Overfitting).

***
*Fuente complementaria: [Diferencias entre Regresión Lineal y Logística (AWS)](https://aws.amazon.com/es/compare/the-difference-between-linear-regression-and-logistic-regression/)*

## 6. Árboles de Decisión

Los árboles de decisión parten el espacio de datos mediante reglas secuenciales. Son fáciles de interpretar por humanos.

![Ejemplo arbol de decision](https://lh6.googleusercontent.com/zBZfWd32HV7q2N7KYpaxmfhXvfF4KPjAkAr4BHPO6UqRtdrRaxi7GlGIdIpCaD847Z06R6twakOS2X-JWXxeuKUHkJHziyRY93xrIbi8iW22N3pxBxUB5-f1j2jj56oDr2HAuDI)


### 6.1 Selección de Atributos (Entropía)
Para decidir qué pregunta hacer en cada nodo del árbol (ej. "¿Es mayor a 5?"), usamos medidas de pureza como la **Entropía**:

$$H(Y) = - \sum p_i \log_2(p_i)$$

El algoritmo (como ID3) busca el atributo que maximice la **Ganancia de Información (Information Gain)**, es decir, el atributo que más reduzca la entropía (incertidumbre) de los datos resultantes.

> **Problema:** Los árboles tienden a sobreajustarse mucho (aprenden el ruido).
> **Solución:** Podar el árbol (pruning) o usar bosques aleatorios (Random Forests).

---

## Conceptos Clave (Glosario)

*   **Agente Racional:** Sistema que percibe y actúa maximizando su medida de desempeño esperada.
*   **Dimensión VC ($$d_{VC}$$):** Medida teórica de la capacidad (complejidad) de un modelo para aprender. A mayor dimensión VC, más datos se necesitan.
*   **Sobreajuste (Overfitting):** Cuando un modelo aprende el "ruido" de los datos de entrenamiento y falla al predecir nuevos datos ($$E_{in}$$ bajo, $$E_{out}$$ alto).
*   **Subajuste (Underfitting):** Cuando el modelo es demasiado simple (baja complejidad) para capturar la estructura subyacente de los datos, resultando en un mal desempeño general ($$E_{in}$$ alto, $$E_{out}$$ alto).
*   **Regularización:** Técnica matemática (como añadir $$\lambda \|w\|^2$$) para prevenir el sobreajuste penalizando modelos complejos.
*   **Descenso del Gradiente:** Algoritmo de optimización que ajusta iterativamente los parámetros moviéndose en la dirección opuesta a la pendiente del error.
*   **Entropía:** En teoría de la información, mide el nivel de desorden o incertidumbre en un conjunto de datos. Usado para construir árboles de decisión.
*   **Matriz de Diseño ($$X$$):** Matriz que contiene todos los datos de entrenamiento, donde cada fila es un ejemplo y cada columna una característica (feature).
