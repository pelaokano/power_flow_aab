# Sistema de Analisis de Flujo de Potencia Electrica

Sistema completo en Python para analisis de flujos de potencia en sistemas electricos, compatible con archivos .raw de PSS/E. Implementa multiples metodos de calculo incluyendo flujos AC, DC y optimo.

## Caracteristicas Principales

### Metodos de Flujo de Potencia Implementados

- DC Power Flow: Analisis linealizado rapido
- AC Power Flow - Newton-Raphson: Convergencia cuadratica (~4-6 iteraciones)
- AC Power Flow - Gauss-Seidel: Metodo iterativo robusto con aceleracion
- Flujo Trifasico Desequilibrado: Analisis fase por fase con cargas desbalanceadas
- DC Optimal Power Flow: Optimizacion de costos usando metodo Simplex

### Capacidades del Sistema

- Parser completo de archivos PSS/E v33 (.raw)
- Visualizacion avanzada con graficas comparativas
- Analisis detallado de voltajes, flujos y perdidas
- Manejo de sistemas grandes (probado con IEEE 118 bus)
- Comparacion de metodos con metricas de desempeno
- Despacho economico con restricciones de red

### Requisitos

Python >= 3.8
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
matplotlib >= 3.4.0
openpyxl >= 3.0.0

# Instalar dependencias
pip install -r requirements.txt

### Crear archivo requirements.txt

numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
openpyxl>=3.0.0

## Uso Rapido

### 1. Flujo de Potencia AC (Newton-Raphson)

from read_raw import RawParser
from power_flow_nr import PowerFlowNR

# Leer archivo .raw
parser = RawParser()
parser.leer_archivo("IEEE_14_bus.raw")
datos = parser.obtener_dataframes()

# Resolver flujo de potencia
solver = PowerFlowNR(datos)
convergencia = solver.resolver(max_iter=20, tolerancia=1e-6)

# Generar reportes
if convergencia:
    df_buses, df_flujos = solver.generar_reporte()

### 2. Flujo de Potencia DC

from power_flow_dc import DCPowerFlow

# Resolver flujo DC
pf_dc = DCPowerFlow(datos)
pf_dc.resolver()
pf_dc.mostrar_resultados()

### 3. DC Optimal Power Flow

from optimal_pf_dc import DCOptimalPowerFlow

# Definir costos de generacion ($/MWh)
costos = {
    1: 35.0,  # Bus 1
    2: 25.0,  # Bus 2
    3: 30.0,  # Bus 3
}

# Resolver OPF
opf = DCOptimalPowerFlow(datos, costos)
opf.resolver_opf()
opf.mostrar_resultados()

### 4. Script Completo con Visualizacion

# Newton-Raphson
python ejecutar_pf.py

# Comparar metodos
python ejecutar_pf_v2.py

# Flujo trifasico desequilibrado
python ejecutar_pf_des.py

## Estructura del Proyecto

power-flow-analysis/
|
├── read_raw.py                 # Parser de archivos .raw PSS/E
├── power_flow_nr.py           # Flujo AC Newton-Raphson
├── power_flow_gs.py           # Flujo AC Gauss-Seidel
├── power_flow_dc.py           # Flujo DC linealizado
├── power_flow_des.py          # Flujo trifasico desequilibrado
├── optimal_pf_dc.py           # DC Optimal Power Flow
|
├── ejecutar_pf.py             # Script basico de ejecucion
├── ejecutar_pf_v2.py          # Comparacion de metodos
├── ejecutar_pf_des.py         # Analisis trifasico
|
├── IEEE 14 bus.raw            # Sistema de prueba IEEE 14 bus
├── IEEE 118 Bus v2.raw        # Sistema de prueba IEEE 118 bus
|
├── requirements.txt           # Dependencias Python
└── README.md                  # Esta documentacion

## Metodos Implementados

### Newton-Raphson (NR)

Ventajas:
- Convergencia cuadratica muy rapida
- Tipicamente 4-6 iteraciones
- Ideal para sistemas bien condicionados

Cuando usar: Analisis estandar de flujos de potencia

solver = PowerFlowNR(datos)
solver.resolver(max_iter=20, tolerancia=1e-6)

### Gauss-Seidel (GS)

Ventajas:
- Mas robusto que Newton-Raphson
- Menor uso de memoria
- Factor de aceleracion ajustable

Cuando usar: Sistemas mal condicionados o como metodo de respaldo

solver = PowerFlowGS(datos)
solver.resolver(max_iter=100, tolerancia=1e-6, factor_aceleracion=1.6)

### DC Power Flow

Ventajas:
- Extremadamente rapido (solucion directa)
- Lineal - siempre converge
- Ideal para estudios rapidos

Limitaciones:
- Solo potencia activa
- Asume voltajes = 1.0 pu
- Ignora resistencias

pf = DCPowerFlow(datos)
pf.resolver()

### DC Optimal Power Flow

Caracteristicas:
- Minimiza costos de generacion
- Respeta limites de flujo en lineas
- Usa metodo Simplex (scipy.linprog)

opf = DCOptimalPowerFlow(datos, costos_generacion)
opf.resolver_opf()

## Ejemplos de Salida

### Resultados de Voltajes

RESULTADOS POR BUS:
--------------------------------------------------------------------------------
numero  nombre           tipo    theta_deg  P_injection_MW
--------------------------------------------------------------------------------
1       Bus 1            Slack   0.0000     232.39
2       Bus 2            PV      -4.9826    18.30
3       Bus 3            PV      -12.7250   -94.20
4       Bus 4            PQ      -10.3128   -47.80
...

### Despacho Optimo

DESPACHO OPTIMO DE GENERADORES:
--------------------------------------------------------------------------------
bus_numero  P_optimo_MW  costo_$/MWh  costo_total_$
--------------------------------------------------------------------------------
1           177.32       35.00        6,206.20
2           48.65        25.00        1,216.25
3           21.59        30.00        647.70
--------------------------------------------------------------------------------
Costo Total de Operacion: $8,070.15

## Visualizaciones

El sistema genera automaticamente:

- Perfiles de voltaje por bus
- Angulos de fase
- Balance de generacion vs carga
- Flujos en lineas mas cargadas
- Diagramas fasoriales
- Graficas comparativas entre metodos

## Configuracion Avanzada

### Analisis Trifasico con Desequilibrio

# Crear configuracion con cargas desequilibradas
config_3ph = {
    'distribucion_default': [0.40, 0.35, 0.25],  # Fase A, B, C
    'carga_0': {
        'distribucion': [0.50, 0.30, 0.20]  # Desequilibrio especifico
    }
}

solver = PowerFlowTrifasico(datos, config_3ph)
solver.resolver()

### Comparacion de Metodos

from ejecutar_pf_v2 import ejecucion_completa

# Ejecuta y compara NR vs GS
resultado = ejecucion_completa(
    "IEEE_14_bus.raw", 
    metodo='AMBOS',
    factor_aceleracion=1.6
)

## Casos de Prueba Incluidos

| Sistema | Buses | Lineas | Generadores | Descripcion |
|---------|-------|--------|-------------|-------------|
| IEEE 14 | 14 | 20 | 5 | Sistema pequeno de prueba |
| IEEE 118 | 118 | 186 | 54 | Sistema mediano realista |

## Solucion de Problemas

### El flujo no converge

# Intentar con Gauss-Seidel mas iteraciones
solver = PowerFlowGS(datos)
solver.resolver(max_iter=200, factor_aceleracion=1.4)

### Matriz singular en Newton-Raphson

# Verificar que exista un bus slack
buses_slack = datos['buses'][datos['buses']['tipo'] == 3]
print(f"Buses slack encontrados: {len(buses_slack)}")

### Error al leer archivo .raw

# Verificar version del archivo
parser = RawParser()
parser.leer_archivo("archivo.raw")
print(f"Version PSS/E detectada: {parser.version}")

## Referencias

- IEEE Test Cases: https://electricgrids.engr.tamu.edu/
- PSS/E Format: Siemens PTI PSS/E Program Operation Manual
- Power Systems Analysis: Grainger & Stevenson, "Power System Analysis"
- Optimal Power Flow: Wood & Wollenberg, "Power Generation, Operation and Control"

## Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (git checkout -b feature/NuevaCaracteristica)
3. Commit tus cambios (git commit -m 'Anadir nueva caracteristica')
4. Push a la rama (git push origin feature/NuevaCaracteristica)
5. Abre un Pull Request

## Licencia

Este proyecto esta bajo la Licencia MIT - ver el archivo LICENSE para mas detalles.

## Autor

Pelaokano

## Agradecimientos

- IEEE por los sistemas de prueba
- Comunidad de Python cientifico (NumPy, SciPy, Pandas)
- Profesores y estudiantes que contribuyeron al desarrollo

---

Ultima actualizacion: Octubre 2025
