---
layout: default
title: Notas de Inteligencia Artificial - Daniel Alvarez
math: true
---

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Notas Inteligencia Artificial

> **Nota sobre el contenido:** Este material fue sintetizado con el apoyo de **NotebookLM**, tomando como base mis apuntes personales y las presentaciones utilizadas en las sesiones de clase.(notas desde enero–marzo de 2026).

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

### 4.3 Características (*features*), plantillas y no linealidad
En la práctica el modelo no actúa sobre $$x$$ crudo sino sobre un vector de características $$\phi(x)$$. La **extracción de características** usa conocimiento del dominio; el **aprendizaje** elige pesos $$w$$ dentro de una familia acotada. Se busca que el conjunto de hipótesis $$\mathcal{F} = \{ f_w : f_w(x) = \text{sign}(w \cdot \phi(x)) \}$$ contenga buenos predictores sin ser demasiado grande.

* **Plantillas de características (*feature templates*):** Agrupan muchas características generadas con la misma regla (por ejemplo, “los tres últimos caracteres del correo son *aaa*, *aab*, …, *com*”). Se definen **tipos** de patrón, no un patrón aislado; el vector resultante suele ser **disperso**: conviene representarlo como diccionario `{"endsWith com": 1}` en lugar de un arreglo denso enorme.
* **Características no lineales en $$x$$, lineales en $$w$$:** Con $$\phi(x) = [1, x, x^2]$$ la predicción $$w \cdot \phi(x)$$ es un polinomio en $$x$$, pero sigue siendo lineal en $$w$$ (misma maquinaria de optimización). Otros ejemplos: **funciones constantes por tramos** (indicadores de intervalos), términos **periódicos** como $$\cos(\omega x)$$, o en clasificación $$\phi(x) = [x_1, x_2, x_1^2 + x_2^2]$$ para que el límite de decisión sea un **círculo** en el plano original pero un **hiperplano** en el espacio de características.

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

Un **problema de búsqueda** se especifica con: estado inicial $$s_{\text{start}}$$, conjunto de **acciones** $$\text{Actions}(s)$$, **sucesor** $$\text{Succ}(s,a)$$, **costo** $$\text{Cost}(s,a)$$ (o costo de la transición), y prueba de fin $$\text{IsEnd}(s)$$. Una **solución** es una **secuencia de acciones** (un plan) que lleva al objetivo; a diferencia de un clasificador “reflejo” $$x \mapsto y$$, aquí hay que valorar **consecuencias futuras** de cada acción.

A cada problema le corresponde un **grafo de espacio de estados** (arcos = transiciones); suele ser enorme o infinito, así que **no** se materializa completo: se explora **bajo demanda**. En el **árbol de búsqueda**, cada **nodo** representa un **camino** (secuencia de acciones) hasta un estado; el mismo estado puede aparecer en varios nodos. La **frontera** (*frontier* / *fringe*) almacena nodos aún no expandidos; la **estrategia** decide cuál expandir.

### 7.1 Tipos de problema (visión rápida)
Según observabilidad e incertidumbre: determinista y totalmente observable (un solo estado de creencia), **sin sensores** (*sensorless* / *conformant*), **parcialmente observable o no determinista** (planes contingentes, a veces intercalar búsqueda y ejecución), o **espacio desconocido** (exploración primero). Las aplicaciones van de rutas y planificación de movimiento hasta traducción automática modelada como secuencias de acciones.

### 7.2 Búsqueda en profundidad, amplitud y en grafo
* **DFS:** frontera tipo **pila** (LIFO); explora primero lo más profundo.
* **BFS:** frontera tipo **cola** (FIFO); encuentra el camino con **menos pasos** si todas las acciones cuestan lo mismo, pero **no** minimiza costo general con acciones de costo distinto.
* **Grafo vs árbol:** sin detectar **estados repetidos**, el trabajo puede crecer exponencialmente (“quien no recuerda el pasado…”). La **búsqueda en grafo** mantiene un conjunto de estados ya **explorados** (`set` / diccionario) y evita reexpandir el mismo estado.

### 7.3 Búsqueda de costo uniforme (UCS)
**Uniform Cost Search** expande siempre el nodo de **menor costo acumulado** $$g(n)$$; la frontera es una **cola de prioridad**. Con costos no negativos es **completa** y **óptima**, pero explora en **contornos de costo** crecientes “en todas direcciones” si no hay información del objetivo.

### 7.4 Heurísticas, *greedy* y $$A^*$$
Una **heurística** $$h(n)$$ estima el costo restante hasta una meta (p. ej. distancia Manhattan o euclidiana en un mapa). La **búsqueda *greedy* best-first*** expande el nodo con menor $$h$$; puede **fallar** y no ser óptima.

**$$A^*$$** combina el costo de llegar al nodo y la estimación al frente:

$$f(n) = g(n) + h(n)$$

* $$h$$ es **admisible** si **nunca sobrestima** el costo real restante hasta la meta (es “optimista”). Con admisibilidad y costos no negativos, $$A^*$$ es óptimo si se declara éxito al **sacar** de la frontera un nodo objetivo (no basta con solo **encolar** un objetivo).
* **UCS** es $$A^*$$ con $$h \equiv 0$$: expande en círculos de costo; $$A^*$$ se “estira” hacia la meta pero mantiene garantías si $$h$$ es admisible.
* Heurísticas **consistentes** (o monótonas) implican $$h(n) \leq c(n,n') + h(n')$$ para sucesores; en la práctica, muchas heurísticas admisibles útiles son consistentes.
* Heurísticas suelen diseñarse como solución de un **problema relajado** (más acciones permitidas). En el **8-puzzle**, contar **fichas mal colocadas** o la **distancia Manhattan** son ejemplos; la Manhattan **domina** a la de fichas mal puestas (explora menos nodos). El **máximo** de varias heurísticas admisibles sigue siendo admisible.

* **Factor de ramificación** $$b$$ y profundidad $$m$$ escalan coste típico $$O(b^m)$$ en exploración exhaustiva del árbol.
* Ejemplos: **Torres de Hanoi**, **8-puzzle**, **cubo de Rubik** (estado y movimientos como acciones).

---

## 8. Juegos de suma cero y adversarios

Un **juego** es un entorno con **más de un agente**. Los ejes que lo clasifican incluyen: determinista o estocástico, **información perfecta** (totalmente observable) o no, dos o más jugadores, **suma cero** (utilidades opuestas: uno maximiza, el otro minimiza) frente a **suma general** (utilidades independientes: cooperación, indiferencia, competencia) o juegos en **equipos**.

Formalización habitual (determinista, por turnos): conjunto de **estados** con un inicial $$s_0$$, **jugadores** que alternan, **acciones** (posiblemente distintas por jugador/estado), función de **transición** $$S \times A \to S$$, test **terminal**, y **utilidades terminales** $$S \times P \to \mathbb{R}$$. Una **estrategia** (*policy*) recomienda una acción en cada estado donde le toca al jugador.

### 8.1 Minimax y complejidad
En juegos **deterministas de suma cero** (gato, ajedrez, damas), el valor **minimax** de un nodo es la mejor utilidad alcanzable contra un adversario **óptimo**. Se implementa con un DFS sobre el árbol de juego. En el peor caso, **tiempo y espacio** $$O(b^m)$$ si $$b$$ es el factor de ramificación y $$m$$ la profundidad del árbol.

### 8.2 Poda $$\alpha$$–$$\beta$$ y límites de profundidad
La **poda $$\alpha$$–$$\beta$$** no cambia el valor minimax en la **raíz** (la mejor jugada para el jugador raíz), aunque valores intermedios puedan quedar incorrectos. $$\alpha$$ resume la mejor opción garantizada para **MAX** en el camino; $$\beta$$ la de **MIN**. Con **orden de generación** de sucesores ideal, el tiempo puede bajar a del orden de $$O(b^{m/2})$$ en el mejor caso (dobla en la práctica la profundidad alcanzable).

En juegos reales **no** se llega a hojas: se usa **profundidad limitada** y una **función de evaluación** en nodos no terminales (a menudo **combinación lineal ponderada de características** del tablero; también redes neuronales entrenadas por autojuego). Cuanto **más profunda** la búsqueda, menos pesa la imperfección de la evaluación. **Profundización iterativa** da un algoritmo *anytime*. La evaluación puede guiar el **orden** de expansiones para favorecer la poda (análogo a cómo una heurística ayuda a $$A^*$$).

* **Negamax** reescribe $$\min(x,y) = -\max(-x,-y)$$ y unifica el código para ambos bandos.

Técnicas adicionales (ajedrez y similares): **tablas de transposición**, **búsqueda quiescente** ante tácticas violentas, **libros de apertura** y **bases de finales**.

---

## 9. Cadenas de Markov y procesos de decisión markovianos (MDP)

Una **cadena de Markov** (por ejemplo de primer orden) satisface que el futuro depende del presente, no de toda la historia: $$P(X_{t+1} \mid X_0,\ldots,X_t) = P(X_{t+1} \mid X_t)$$. Sirve como base para **series de tiempo**, filtrado y pronóstico (*forecasting*).

### 9.1 De la búsqueda determinista al MDP
En un problema de **búsqueda** clásico, $$\text{Succ}(s,a)$$ da un único siguiente estado y el costo reemplaza a la recompensa. En un **MDP**, la acción produce una **distribución** sobre siguientes estados:

| Búsqueda | MDP |
| :--- | :--- |
| $$\text{Succ}(s,a)$$ determinista | Probabilidades $$T(s,a,s') = P(s' \mid s,a)$$ |
| Costo de acción | Recompensa $$R(s,a,s')$$ (o $$R(s,a)$$ según notación) |

Para cada par $$(s,a)$$, las probabilidades de transición suman $$1$$: $$\sum_{s'} T(s,a,s') = 1$$. Los **sucesores** son los $$s'$$ con $$T(s,a,s') > 0$$.

Un MDP queda definido por: conjunto de **estados**, acciones $$\text{Actions}(s)$$, $$T(s,a,s')$$, recompensa de transición, $$\text{IsEnd}(s)$$ y **factor de descuento** $$\gamma \in [0,1]$$ (típicamente $$\gamma = 1$$ solo si el horizonte es finito y está bien definido; si no, $$\gamma < 1$$ ayuda a la convergencia).

### 9.2 Política, utilidad y valor esperado
Una **política** $$\pi$$ asigna una acción a cada estado (donde corresponda). Siguiendo $$\pi$$ se obtiene un **camino aleatorio**; la **utilidad** del camino es la suma **descontada** de recompensas:

$$u = r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots$$

* $$\gamma \to 1$$ valora mucho el futuro; $$\gamma \to 0$$ “solo el momento presente”.

El **valor de $$\pi$$** en $$s$$ es la **utilidad esperada** desde $$s$$:

$$V^\pi(s) = \mathbb{E}[\text{utilidad} \mid \pi, s]$$

Los valores **$$Q^\pi(s,a)$$** son la utilidad esperada tras tomar $$a$$ en $$s$$ y seguir $$\pi$$. Para estados no finales, valen las recurrencias (esquema):

$$Q^\pi(s,a) = \sum_{s'} T(s,a,s')\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr], \qquad
V^\pi(s) = Q^\pi(s,\pi(s)) \ \ (\text{si no es estado final}).$$

**Evaluación de políticas:** se inicializan valores y se iteran las ecuaciones hasta que el cambio máximo sea menor que un umbral $$\varepsilon$$. Coste típico por iteración del orden de $$O(|S| \cdot |A| \cdot S')$$ con $$S'$$ sucesores no nulos por par $$(s,a)$$.

### 9.3 Valor óptimo e iteración de valor (Bellman)
El **valor óptimo** cumple:

$$Q_{\text{opt}}(s,a) = \sum_{s'} T(s,a,s')\bigl[R(s,a,s') + \gamma V_{\text{opt}}(s')\bigr], \qquad
V_{\text{opt}}(s) = \max_{a \in \text{Actions}(s)} Q_{\text{opt}}(s,a)$$

La **política óptima** (greedy respecto a $$Q_{\text{opt}}$$) es $$\pi_{\text{opt}}(s) \in \arg\max_a Q_{\text{opt}}(s,a)$$.

**Iteración de valor:** inicializar $$V(s)$$ (p. ej. $$0$$) y repetir

$$V^{(t)}(s) \leftarrow \max_{a} \sum_{s'} T(s,a,s')\bigl[R(s,a,s') + \gamma V^{(t-1)}(s')\bigr]$$

hasta convergencia. **Convergencia** habitual si $$\gamma < 1$$ o si el grafo del MDP es **acíclico**; con $$\gamma = 1$$ y recompensas nulas puede no converger en ciclos.

**Idea unificadora (programación dinámica):** la búsqueda con DP calcula costos futuros mínimos en grafos; la **evaluación de políticas** calcula $$V^\pi$$; la **iteración de valor** calcula $$V_{\text{opt}}$$. El patrón es escribir la recurrencia y convertirla en asignaciones iterativas hasta converger.

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
*   **Heurística admisible:** Estimación del costo a la meta que no sobrestima el costo real.
*   **MDP:** Modelo con estados, acciones, transiciones (posiblemente estocásticas) y recompensas; la política óptima maximiza el retorno esperado descontado.
*   **Poda $$\alpha$$–$$\beta$$:** Técnica que reduce nodos explorados en minimax sin cambiar el resultado en juegos de suma cero con utilidad exacta en hojas.
*   **UCS (Uniform Cost Search):** Búsqueda que expande por menor costo acumulado $$g$$; óptima con costos no negativos.
*   **Transición $$T(s,a,s')$$:** En MDPs, probabilidad de llegar a $$s'$$ tras $$a$$ en $$s$$; generaliza al sucesor único de la búsqueda clásica.
*   **Iteración de valor:** Algoritmo de Bellman que actualiza $$V(s)$$ con el máximo sobre acciones de la expectativa de recompensa más $$V$$ descontado en sucesores.
