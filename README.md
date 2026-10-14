# Agotados en Almacenes del Valle

Proyecto de Análisis de Datos 1. Una fila es una semana, una tienda y un producto, del 6 de enero de 2025 al 29 de junio de 2026.

Los datos son **sintéticos**. Tienen la estructura que produciría un proceso nocturno desde el punto de venta, el inventario y el calendario de promociones. No son un extracto de una empresa real.

**Pregunta.** ¿Qué factores operativos explican la tasa de agotado de 24 productos de alta rotación en 12 tiendas, de modo que abastecimiento pueda bajarla de 28,3% a 18% o menos entre octubre de 2026 y marzo de 2027?

## Notebooks

| Archivo | Contenido |
| --- | --- |
| `notebook/analisis_agotados_almacenes_del_valle.ipynb` | Proyecto final: calidad, exploración, relaciones y alerta del lunes |
| `notebook/Analisis_exploratorio.ipynb` | Taller 1: solo el análisis exploratorio |

Los dos notebooks ya tienen las salidas guardadas. Para volver a ejecutarlos, desde la raíz del repositorio:

```powershell
pip install -r requirements.txt
```

Abrir el notebook en Jupyter o VS Code / Cursor y ejecutar todas las celdas. Buscan `datos/agotados_semanal.csv` subiendo desde la carpeta del archivo.

## Datos

`datos/agotados_semanal.csv` — 22.464 filas.  
`datos/muestra_datos.csv` — 12 filas de ejemplo (leche entera en Cali Centro).

`src/analizar.py` es el módulo que usa el notebook del proyecto final para limpiar, resumir y entrenar la alerta.

## Hallazgos en corto

- Tasa de agotado: 28,3%. Venta perdida: unos 213 millones de pesos al mes.
- Con promoción la tasa pasa de 23% a 58%.
- Cárnicos, lácteos y panadería concentran el 72% de la plata perdida.
- Cali Centro, Cali Norte y Cali Oriente concentran el 46%.
- La alerta del lunes (regresión logística) ve el 82% de los agotados en la prueba y señala los casos donde está el 92% de la venta perdida de ese periodo.
