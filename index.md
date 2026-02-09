---
layout: default
title: Notas de Inteligencia Artificial - Daniel Alvarez
math: true
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
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

### 3.1 El Problema del Aprendizaje Supervisado
El objetivo es encontrar una función (hipótesis) $$h$$ que se aproxime a una función desconocida $$f$$ (la realidad), tal que $$h(x) \approx f(x)$$, utilizando un conjunto de datos de entrenamiento $$D$$.

*   **Input:** $$X$$ (vector de características).
*   **Output:** $$Y$$ (etiqueta o valor).
*   **Hipótesis:** $$h \in \mathcal{H}$$ (espacio de posibles funciones del modelo).

### 3.2 Error y Generalización
Para saber si el modelo aprende, distinguimos dos tipos de error:
1.  **Error en muestra ($$E_{in}$$):** El error calculado sobre los datos de entrenamiento.
2.  **Error fuera de muestra ($$E_{out}$$):** El error sobre datos nuevos (generalización).

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
Es un algoritmo iterativo para minimizar el error cuando no podemos usar la solución analítica.

**Algoritmo:**
Repetir hasta convergencia:
$$w_j \leftarrow w_j - \eta \frac{\partial J(w)}{\partial w_j}$$
Donde $\eta$ es la **tasa de aprendizaje (learning rate)**.

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

## 5. Clasificación y Regularización

### 5.1 Clasificación Lineal
Si la salida $$Y$$ es discreta (ej. $$\{-1, +1\}$$), usamos un umbral.
*   **Perceptrón:** Usa la función `signo(w^Tx)`.
*   **Regresión Logística:** Usa la función **Sigmoide** para estimar probabilidades entre 0 y 1:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    *Función de costo:* Entropía cruzada (Log Loss), ya que el MSE no es convexo para clasificación.

### 5.2 Regularización (L2 / Weight Decay)
Para evitar que el modelo "memorice" los datos (overfitting), penalizamos los pesos grandes. Esto se llama **Navaja de Ockham**: preferir modelos más simples.

Nueva función de costo a minimizar:
$$J_{reg}(w) = E_{in}(w) + \lambda \sum w_j^2$$
*   $\lambda$ (lambda): Controla cuánto penalizamos la complejidad. Si $\lambda$ es muy grande, el modelo será demasiado simple (underfitting); si es 0, es regresión normal.

---

## 6. Árboles de Decisión

Los árboles de decisión parten el espacio de datos mediante reglas secuenciales. Son fáciles de interpretar por humanos.

### 6.1 Selección de Atributos (Entropía)
Para decidir qué pregunta hacer en cada nodo del árbol (ej. "¿Es mayor a 5?"), usamos medidas de pureza como la **Entropía**:

$$H(Y) = - \sum p_i \log_2(p_i)$$

El algoritmo (como ID3 o C4.5) busca el atributo que maximice la **Ganancia de Información (Information Gain)**, es decir, el atributo que más reduzca la entropía (incertidumbre) de los datos resultantes.

> **Problema:** Los árboles tienden a sobreajustarse mucho (aprenden el ruido).
> **Solución:** Podar el árbol (pruning) o usar bosques aleatorios (Random Forests).

---

## Conceptos Clave (Glosario)

*   **Agente Racional:** Sistema que percibe y actúa maximizando su medida de desempeño esperada.
*   **Dimensión VC ($$d_{VC}$$):** Medida teórica de la capacidad (complejidad) de un modelo para aprender. A mayor dimensión VC, más datos se necesitan.
*   **Sobreajuste (Overfitting):** Cuando un modelo aprende el "ruido" de los datos de entrenamiento y falla al predecir nuevos datos ($$E_{in}$$ bajo, $$E_{out}$$ alto).
*   **Regularización:** Técnica matemática (como añadir $$\lambda ||w||^2$$) para prevenir el sobreajuste penalizando modelos complejos.
*   **Gradiente Descendente:** Algoritmo de optimización que ajusta iterativamente los parámetros moviéndose en la dirección opuesta a la pendiente del error.
*   **Entropía:** En teoría de la información, mide el nivel de desorden o incertidumbre en un conjunto de datos. Usado para construir árboles de decisión.
*   **Matriz de Diseño ($$X$$):** Matriz que contiene todos los datos de entrenamiento, donde cada fila es un ejemplo y cada columna una característica (feature).
