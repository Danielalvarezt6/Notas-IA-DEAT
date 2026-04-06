---
layout: default
title: Notas de Inteligencia Artificial - Daniel Alvarez
math: true
---

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Notas Inteligencia Artificial

> **Nota sobre el contenido:** Este material fue sintetizado con el apoyo de **NotebookLM**, tomando como base mis apuntes personales y las presentaciones utilizadas en las sesiones de clase. Las secciones de búsqueda, juegos adversarios y MDP siguen el desarrollo visto en el cuatrimestre (notas desde enero–marzo de 2026).

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

Un **agente reflexivo** puede implementarse como $$a_t = f(p_t)$$ o, con estado interno, $$a_t = f(p_t, s_t)$$, actualizando $$s_t$$ antes de decidir (por ejemplo, recordando la percepción anterior para comparar).

### 2.4 Estado, representación y espacio de búsqueda
El **estado** suele codificarse como un vector $$s = (s_1, \ldots, s_n)$$ cuyas componentes son **variables de estado** (posición, inventario, etc.). Las propiedades del entorno (estático/dinámico, discreto/continuo, observable o no, determinista/estocástico, episódico/secuencial) determinan cómo se actualiza el estado: en lo determinista, $$s_{k+1} = f(s_k, a_k)$$; con incertidumbre se usan modelos probabilísticos sobre transiciones.

El **cardinal** del espacio de estados crece muy rápido; en problemas combinatorios puede ser astronómicamente grande, lo que motiva algoritmos de búsqueda e inteligentes en lugar de enumeración exhaustiva.

### 2.5 Mapa temático del curso (visión global)
Según las notas, los temas se articulan de lo más abstracto y estructurado hacia aplicaciones con incertidumbre y conocimiento:

* **Aprendizaje automático** y **búsqueda** en espacios de estados finitos.
* **Juegos** (deterministas) y **MDP / juegos estocásticos** cuando hay azar.
* **Problemas de satisfacción de restricciones (CSP)** y **optimización**.
* **Sistemas basados en conocimiento** (representación e inferencia).

---

### 2.6 Sistemas basados en conocimiento
Además del enfoque numérico $$f(x)$$ típico del aprendizaje supervisado, se trabaja el paradigma **declarativo**: una **base de conocimiento** (reglas y relaciones) y una **base de hechos** sobre el mundo actual, consultadas por un **motor de inferencia** que deduce respuestas. El entorno puede modelarse de forma **orientada a objetos** (propiedades como datos, métodos que dependen del estado) para organizar el código del agente.

---

## 3. Teoría del Aprendizaje (Learning Theory)

### 3.1 Aprendizaje Supervisado
Es una técnica de *Machine Learning* donde el modelo aprende a partir de un conjunto de datos **etiquetados**. Estos datos proporcionan una **"Verdad Fundamental" (Ground Truth)**, que actúa como un guía o "profesor" para el algoritmo.

* **El objetivo:** Crear una función (hipótesis) $$h$$ que se aproxime a una función desconocida $$f$$ (la realidad), tal que $$h(x) \approx f(x)$$, utilizando un conjunto de entrenamiento $$D$$.
* **Funcionamiento:** El modelo hace predicciones, mide el error respecto a la etiqueta real (usando una **función de pérdida**) y ajusta sus parámetros para minimizar dicha discrepancia.

Los datos de entrenamiento pueden reflejar **sesgos sociales** (por ejemplo, en un modelo de aprobación de préstamos si las etiquetas históricas discriminan grupos). Eso no se corrige solo con más datos: hace falta criterio ético, auditoría de variables y, a veces, replantear la medida de desempeño.

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
$$J(w) = \frac{1}{M} \sum_{i=1}^{M} (h_w(x^{(i)}) - y^{(i)})^2$$

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
> * $$\lambda$$ muy grande $$\to$$ Underfitting (modelo demasiado simple).
> * $$\lambda = 0$$ $$\to$$ Regresión estándar (riesgo de Overfitting).

### 5.3 Clasificador lineal, pérdida *hinge* y regresión logística
Un clasificador binario puede usar $$h(x) = \text{sign}(w^\top x + b)$$. La pérdida **hinge** (estilo SVM) penaliza ejemplos mal clasificados o demasiado cerca del margen: $$\mathcal{L}(y,\hat{y}) = \max(0, 1 - y \cdot \hat{y})$$. Este enfoque **solo funciona bien si los datos son aproximadamente linealmente separables**; si no lo son, hay que cambiar de representación (por ejemplo **ingeniería de características**, expansión polinomial) o usar modelos no lineales / regresión logística con umbral probabilístico.

La **regresión logística** asigna $$a = \sigma(w^\top x + b)$$ y entrena con entropía cruzada; el gradiente aprovecha $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$ para actualizar $$w$$ y $$b$$ de forma estable.

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

En la práctica, variantes como **CART** pueden entrenar más lento que un solo árbol pequeño, pero la predicción en un árbol equilibrado suele ser del orden $$O(\log n)$$ respecto al tamaño del conjunto.

---

## 7. Búsqueda y planeación en espacios de estados

Un **problema de búsqueda** (espacio finito) se define con: conjunto de **estados** $$S$$, **acciones** legales $$A(s)$$, **modelo de transición** $$\text{Succ}(s,a)$$, **costo local** $$c(s,a)$$, **estado inicial** y **estados meta** (o condición $$\text{Terminal}(s)$$). Un **plan** es una secuencia $$(s_0,a_0,s_1,\ldots)$$ tal que $$s_{i+1} = \text{Succ}(s_i, a_i)$$ y el **costo total** es la suma de costos locales; se busca un plan de **costo mínimo**.

La **búsqueda genérica** mantiene una **frontera** de nodos (cada nodo guarda estado, padre, acción y costo acumulado), expande sucesores y detiene al alcanzar un estado terminal. Si el grafo de estados tiene **ciclos** o estados repetidos, conviene tratarlo como **búsqueda en grafo** y usar un conjunto de estados ya visitados (en Python, un `set`) para no explotar el tiempo.

* **Factor de ramificación** $$b$$ y profundidad afectan la **complejidad temporal y espacial** de los algoritmos (crecimiento típico en $$O(b^d)$$ en el peor caso para búsqueda en profundidad).
* Ejemplos vistos en clase: **Torres de Hanoi**, **puzzle deslizable** (8-puzzle), representación del estado del cubo de Rubik.

### 7.1 Búsqueda informada y $$A^*$$
Una **heurística** $$h(n)$$ estima el costo desde el estado del nodo $$n$$ hasta la meta. Si $$h$$ nunca **sobrestima** el costo real, es **admisible**; entonces $$A^*$$ con $$f(n) = g(n) + h(n)$$ ($$g$$ = costo desde el inicio) encuentra solución **óptima** cuando existe. Con información extra útil en el problema, $$A^*$$ (o variantes) suele ser la mejor opción frente a búsqueda ciega.

---

## 8. Juegos de suma cero y adversarios

En **juegos de dos jugadores y suma cero**, un jugador maximiza una utilidad y el otro la minimiza; el entorno es **dinámico** y **determinista** (en la versión básica), con turnos alternados.

* **Minimax** explora el árbol de juego hasta estados terminales (o hasta profundidad máxima) y propaga valores hacia arriba. Complejidad en el peor caso del orden $$O(b^{d_{\max}})$$.
* **Poda $$\alpha$$–$$\beta$$** elimina ramas que no pueden mejorar el resultado ya garantizado por otra jugada; con **ordenamiento de jugadas** favorable se hacen más cortes.
* **Negamax** reescribe min/max usando $$\min(x,y) = -\max(-x,-y)$$, unificando el código para ambos lados.

Técnicas adicionales en juegos complejos (ajedrez): **tablas de transposición** (cacheo de estados ya evaluados), **búsqueda quiescente** (seguir explorando mientras el material o la posición cambian mucho), **libros de apertura** y **bases de datos de finales**. Las **heurísticas** suelen combinar material, control del centro, seguridad del rey, etc.

---

## 9. Cadenas de Markov y procesos de decisión markovianos (MDP)

Una **cadena de Markov** (por ejemplo de primer orden) satisface que el futuro depende del presente, no de toda la historia: $$P(X_{t+1} \mid X_0,\ldots,X_t) = P(X_{t+1} \mid X_t)$$. Sirve como base para **series de tiempo**, filtrado y pronóstico (*forecasting*).

Un **MDP** extiende el modelo con **acciones** y **recompensas** $$R(s,a,s')$$; suele usarse un **factor de descuento** $$\gamma \in (0,1]$$ para valorar utilidades futuras. La **función valor** $$V^\pi(s)$$ mide el retorno esperado siguiendo la política $$\pi$$; existe una **política óptima** $$\pi^*$$ que maximiza el valor en cada estado (bajo condiciones habituales).

Las **ecuaciones de optimalidad de Bellman** relacionan $$V^*$$ y la función **$$Q^*(s,a)$$** (valor de tomar $$a$$ en $$s$$ y actuar óptimo después). Algoritmos como **iteración de valor** o **iteración de política** (programación dinámica) aproximan $$\pi^*$$ y $$Q$$ cuando el modelo es conocido; la idea codificada en clase actualiza $$Q(s,a)$$ con expectativas sobre transiciones y máximo sobre acciones en el siguiente estado.

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
*   **Heurística admisible:** Estimación del costo a la meta que no sobrestima el costo real; condición clave para optimalidad de $$A^*$$ con costos no negativos.
*   **MDP:** Modelo con estados, acciones, transiciones (posiblemente estocásticas) y recompensas; la política óptima maximiza el retorno esperado descontado.
*   **Poda $$\alpha$$–$$\beta$$:** Técnica que reduce nodos explorados en minimax sin cambiar el resultado en juegos de suma cero con utilidad exacta en hojas.
