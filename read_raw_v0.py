"""
PARSER COMPLETO DE ARCHIVOS .RAW DE PSS/E
Sin dependencias externas (solo pandas y numpy)
Soporta versiones 29-35
"""

import pandas as pd
import numpy as np
import re


class RawParser:
    """
    Parser completo para archivos .raw de PSS/E
    """
    
    def __init__(self):
        self.version = None
        self.base_mva = 100.0
        self.frecuencia = 60.0
        
        # Datos extraídos
        self.buses = []
        self.cargas = []
        self.generadores = []
        self.lineas = []
        self.transformadores = []
        self.shunts = []
        
        # Mapeo de buses por número
        self.bus_map = {}
    
    def leer_archivo(self, archivo_raw):
        """
        Lee y parsea un archivo .raw de PSS/E
        """
        print(f"Leyendo archivo: {archivo_raw}")
        print("=" * 80)
        
        with open(archivo_raw, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
        
        # Parsear encabezado
        self._parsear_encabezado(lineas[0:3])
        
        # Determinar secciones
        seccion_actual = None
        i = 3  # Empezar después del encabezado
        
        while i < len(lineas):
            linea = lineas[i].strip()
            
            # Saltar líneas vacías y comentarios
            if not linea or linea.startswith('@'):
                i += 1
                continue
            
            # Detectar cambio de sección (línea que empieza con 0 o Q)
            if linea.startswith('0 /') or linea.startswith('Q') or linea.startswith('0,'):
                seccion_actual = self._siguiente_seccion(seccion_actual)
                i += 1
                continue
            
            # Parsear según la sección actual
            if seccion_actual == 'BUS':
                self._parsear_bus(linea)
            elif seccion_actual == 'LOAD':
                self._parsear_carga(linea)
            elif seccion_actual == 'GENERATOR':
                self._parsear_generador(linea)
            elif seccion_actual == 'BRANCH':
                self._parsear_linea(linea)
            elif seccion_actual == 'TRANSFORMER':
                # Transformadores tienen múltiples líneas
                i = self._parsear_transformador(lineas, i)
                i += 1
                continue
            elif seccion_actual == 'SHUNT':
                self._parsear_shunt(linea)
            
            i += 1
        
        print(f"\nArchivo parseado exitosamente")
        print(f"  Base MVA: {self.base_mva}")
        print(f"  Versión: {self.version}")
        print(f"  Frecuencia: {self.frecuencia} Hz")
        
        return self
    
    def _parsear_encabezado(self, lineas_encabezado):
        """Parsea las primeras líneas del encabezado"""
        # Eliminar comentarios (todo después de '/')
        primera_linea_limpia = lineas_encabezado[0].split('/')[0]
        primera_linea = primera_linea_limpia.split(',')
        
        # IC, SBASE, REV, XFRRAT, NXFRAT, BASFRQ
        if len(primera_linea) >= 2:
            try:
                self.base_mva = float(primera_linea[1].strip())
            except ValueError:
                self.base_mva = 100.0
        
        if len(primera_linea) >= 3:
            try:
                self.version = int(float(primera_linea[2].strip()))
            except ValueError:
                self.version = 33
        
        if len(primera_linea) >= 6:
            try:
                self.frecuencia = float(primera_linea[5].strip())
            except ValueError:
                self.frecuencia = 60.0
    
    def _siguiente_seccion(self, seccion_actual):
        """Determina cuál es la siguiente sección"""
        orden_secciones = [
            None, 'BUS', 'LOAD', 'SHUNT', 'GENERATOR', 'BRANCH', 
            'TRANSFORMER', 'AREA', 'ZONE', 'OWNER', 'FACTS', 'SWSHUNT'
        ]
        
        if seccion_actual is None:
            return 'BUS'
        
        try:
            idx = orden_secciones.index(seccion_actual)
            if idx + 1 < len(orden_secciones):
                return orden_secciones[idx + 1]
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _parsear_linea_csv(self, linea):
        """
        Parsea una línea CSV del archivo raw, manejando strings entre comillas
        """
        # Eliminar comentarios (todo después de '/')
        if '/' in linea:
            linea = linea.split('/')[0]
        
        elementos = []
        actual = ''
        dentro_comillas = False
        
        for char in linea:
            if char == "'" or char == '"':
                dentro_comillas = not dentro_comillas
            elif char == ',' and not dentro_comillas:
                elementos.append(actual.strip())
                actual = ''
            else:
                actual += char
        
        # Agregar último elemento
        if actual.strip():
            elementos.append(actual.strip())
        
        return elementos
    
    def _convertir_tipo(self, valor, tipo_esperado='float'):
        """Convierte un valor a su tipo apropiado"""
        if not valor or valor in ['', ' ']:
            return 0.0 if tipo_esperado == 'float' else 0 if tipo_esperado == 'int' else ''
        
        valor = valor.strip().strip("'").strip('"')
        
        try:
            if tipo_esperado == 'int':
                return int(float(valor))
            elif tipo_esperado == 'float':
                return float(valor)
            else:
                return valor
        except (ValueError, TypeError):
            return 0.0 if tipo_esperado == 'float' else 0 if tipo_esperado == 'int' else valor
    
    def _parsear_bus(self, linea):
        """
        Parsea datos de bus
        Formato PSS/E: I,NAME,BASKV,IDE,AREA,ZONE,OWNER,VM,VA,NVHI,NVLO,EVHI,EVLO
        """
        partes = self._parsear_linea_csv(linea)
        
        if len(partes) < 4:
            return
        
        bus_num = self._convertir_tipo(partes[0], 'int')
        
        # Determinar si es slack
        tipo = self._convertir_tipo(partes[3], 'int') if len(partes) > 3 else 1
        is_slack = (tipo == 3)
        
        bus_data = {
            'numero': bus_num,
            'nombre': self._convertir_tipo(partes[1], 'str') if len(partes) > 1 else f'Bus_{bus_num}',
            'V_base_kV': self._convertir_tipo(partes[2], 'float') if len(partes) > 2 else 138.0,
            'tipo': tipo,
            'is_slack': is_slack,
            'area': self._convertir_tipo(partes[4], 'int') if len(partes) > 4 else 1,
            'zona': self._convertir_tipo(partes[5], 'int') if len(partes) > 5 else 1,
            'V_mag_pu': self._convertir_tipo(partes[7], 'float') if len(partes) > 7 else 1.0,
            'V_ang_deg': self._convertir_tipo(partes[8], 'float') if len(partes) > 8 else 0.0,
            'V_max_pu': self._convertir_tipo(partes[9], 'float') if len(partes) > 9 else 1.1,
            'V_min_pu': self._convertir_tipo(partes[10], 'float') if len(partes) > 10 else 0.9,
        }
        
        self.buses.append(bus_data)
        self.bus_map[bus_num] = len(self.buses) - 1
    
    def _parsear_carga(self, linea):
        """
        Parsea datos de carga
        Formato: I,ID,STATUS,AREA,ZONE,PL,QL,IP,IQ,YP,YQ,OWNER,SCALE,INTRPT
        """
        partes = self._parsear_linea_csv(linea)
        
        if len(partes) < 6:
            return
        
        bus_num = self._convertir_tipo(partes[0], 'int')
        
        carga_data = {
            'bus_numero': bus_num,
            'bus_index': self.bus_map.get(bus_num, -1),
            'id': self._convertir_tipo(partes[1], 'str') if len(partes) > 1 else '1',
            'status': self._convertir_tipo(partes[2], 'int') if len(partes) > 2 else 1,
            'P_MW': self._convertir_tipo(partes[5], 'float') if len(partes) > 5 else 0.0,
            'Q_MVAR': self._convertir_tipo(partes[6], 'float') if len(partes) > 6 else 0.0,
            'P_pu': 0.0,
            'Q_pu': 0.0
        }
        
        # Convertir a pu
        carga_data['P_pu'] = carga_data['P_MW'] / self.base_mva
        carga_data['Q_pu'] = carga_data['Q_MVAR'] / self.base_mva
        
        self.cargas.append(carga_data)
    
    def _parsear_generador(self, linea):
        """
        Parsea datos de generador
        Formato: I,ID,PG,QG,QT,QB,VS,IREG,MBASE,ZR,ZX,RT,XT,GTAP,STAT,RMPCT,PT,PB,O1,F1,...
        """
        partes = self._parsear_linea_csv(linea)
        
        if len(partes) < 4:
            return
        
        bus_num = self._convertir_tipo(partes[0], 'int')
        bus_idx = self.bus_map.get(bus_num, -1)
        
        # Verificar si el bus es slack
        is_slack = False
        if bus_idx >= 0 and bus_idx < len(self.buses):
            is_slack = self.buses[bus_idx]['is_slack']
        
        gen_data = {
            'bus_numero': bus_num,
            'bus_index': bus_idx,
            'id': self._convertir_tipo(partes[1], 'str') if len(partes) > 1 else '1',
            'P_MW': self._convertir_tipo(partes[2], 'float') if len(partes) > 2 else 0.0,
            'Q_MVAR': self._convertir_tipo(partes[3], 'float') if len(partes) > 3 else 0.0,
            'Q_max_MVAR': self._convertir_tipo(partes[4], 'float') if len(partes) > 4 else 100.0,
            'Q_min_MVAR': self._convertir_tipo(partes[5], 'float') if len(partes) > 5 else -100.0,
            'V_set_pu': self._convertir_tipo(partes[6], 'float') if len(partes) > 6 else 1.0,
            'MVA_base': self._convertir_tipo(partes[8], 'float') if len(partes) > 8 else self.base_mva,
            'status': self._convertir_tipo(partes[14], 'int') if len(partes) > 14 else 1,
            'P_max_MW': self._convertir_tipo(partes[16], 'float') if len(partes) > 16 else 100.0,
            'P_min_MW': self._convertir_tipo(partes[17], 'float') if len(partes) > 17 else 0.0,
            'is_slack': is_slack,
            'P_pu': 0.0,
            'Q_pu': 0.0
        }
        
        # Convertir a pu
        gen_data['P_pu'] = gen_data['P_MW'] / self.base_mva
        gen_data['Q_pu'] = gen_data['Q_MVAR'] / self.base_mva
        
        self.generadores.append(gen_data)
    
    def _parsear_linea(self, linea):
        """
        Parsea datos de línea (branch)
        Formato: I,J,CKT,R,X,B,RATEA,RATEB,RATEC,GI,BI,GJ,BJ,ST,MET,LEN,O1,F1,...
        """
        partes = self._parsear_linea_csv(linea)
        
        if len(partes) < 6:
            return
        
        from_bus = self._convertir_tipo(partes[0], 'int')
        to_bus = self._convertir_tipo(partes[1], 'int')
        
        linea_data = {
            'from_bus_numero': from_bus,
            'to_bus_numero': to_bus,
            'from_bus_index': self.bus_map.get(from_bus, -1),
            'to_bus_index': self.bus_map.get(to_bus, -1),
            'circuit': self._convertir_tipo(partes[2], 'str') if len(partes) > 2 else '1',
            'R_pu': self._convertir_tipo(partes[3], 'float') if len(partes) > 3 else 0.0,
            'X_pu': self._convertir_tipo(partes[4], 'float') if len(partes) > 4 else 0.0,
            'B_pu': self._convertir_tipo(partes[5], 'float') if len(partes) > 5 else 0.0,
            'rate_A_MVA': self._convertir_tipo(partes[6], 'float') if len(partes) > 6 else 0.0,
            'rate_B_MVA': self._convertir_tipo(partes[7], 'float') if len(partes) > 7 else 0.0,
            'rate_C_MVA': self._convertir_tipo(partes[8], 'float') if len(partes) > 8 else 0.0,
            'status': self._convertir_tipo(partes[13], 'int') if len(partes) > 13 else 1,
            'longitud': self._convertir_tipo(partes[15], 'float') if len(partes) > 15 else 0.0,
        }
        
        self.lineas.append(linea_data)
    
    def _parsear_transformador(self, lineas, indice_actual):
        """
        Parsea datos de transformador (múltiples líneas)
        Los transformadores tienen 4 líneas de datos en PSS/E
        """
        # Línea 1: I,J,K,CKT,CW,CZ,CM,MAG1,MAG2,NMETR,'NAME',STAT,O1,F1,...
        linea1 = self._parsear_linea_csv(lineas[indice_actual])
        
        if len(linea1) < 3:
            return indice_actual
        
        from_bus = self._convertir_tipo(linea1[0], 'int')
        to_bus = self._convertir_tipo(linea1[1], 'int')
        
        # Verificar si es transformador de 3 devanados (K != 0)
        k_bus = self._convertir_tipo(linea1[2], 'int')
        is_3w = (k_bus != 0)
        
        # Avanzar a las siguientes líneas del transformador
        indice_actual += 1
        if indice_actual >= len(lineas):
            return indice_actual - 1
        
        # Línea 2: R1-2,X1-2,SBASE1-2
        linea2 = self._parsear_linea_csv(lineas[indice_actual])
        
        indice_actual += 1
        if indice_actual >= len(lineas):
            return indice_actual - 1
        
        # Línea 3: WINDV1,NOMV1,ANG1,RATA1,RATB1,RATC1,COD1,CONT1,RMA1,RMI1,VMA1,VMI1,NTP1,TAB1,CR1,CX1
        linea3 = self._parsear_linea_csv(lineas[indice_actual])
        
        indice_actual += 1
        if indice_actual >= len(lineas):
            return indice_actual - 1
        
        # Línea 4: WINDV2,NOMV2
        linea4 = self._parsear_linea_csv(lineas[indice_actual])
        
        trafo_data = {
            'from_bus_numero': from_bus,
            'to_bus_numero': to_bus,
            'from_bus_index': self.bus_map.get(from_bus, -1),
            'to_bus_index': self.bus_map.get(to_bus, -1),
            'circuit': self._convertir_tipo(linea1[3], 'str') if len(linea1) > 3 else '1',
            'R_pu': self._convertir_tipo(linea2[0], 'float') if len(linea2) > 0 else 0.0,
            'X_pu': self._convertir_tipo(linea2[1], 'float') if len(linea2) > 1 else 0.0,
            'windv1': self._convertir_tipo(linea3[0], 'float') if len(linea3) > 0 else 1.0,
            'nomv1_kV': self._convertir_tipo(linea3[1], 'float') if len(linea3) > 1 else 0.0,
            'ang1_deg': self._convertir_tipo(linea3[2], 'float') if len(linea3) > 2 else 0.0,
            'rate_A_MVA': self._convertir_tipo(linea3[3], 'float') if len(linea3) > 3 else 0.0,
            'rate_B_MVA': self._convertir_tipo(linea3[4], 'float') if len(linea3) > 4 else 0.0,
            'rate_C_MVA': self._convertir_tipo(linea3[5], 'float') if len(linea3) > 5 else 0.0,
            'windv2': self._convertir_tipo(linea4[0], 'float') if len(linea4) > 0 else 1.0,
            'nomv2_kV': self._convertir_tipo(linea4[1], 'float') if len(linea4) > 1 else 0.0,
            'status': self._convertir_tipo(linea1[11], 'int') if len(linea1) > 11 else 1,
            'is_3winding': is_3w
        }
        
        # Calcular tap ratio
        if trafo_data['windv1'] != 0:
            trafo_data['tap_ratio'] = trafo_data['windv2'] / trafo_data['windv1']
        else:
            trafo_data['tap_ratio'] = 1.0
        
        self.transformadores.append(trafo_data)
        
        return indice_actual
    
    def _parsear_shunt(self, linea):
        """
        Parsea datos de shunt fijo
        Formato: I,ID,STATUS,GL,BL
        """
        partes = self._parsear_linea_csv(linea)
        
        if len(partes) < 5:
            return
        
        bus_num = self._convertir_tipo(partes[0], 'int')
        
        shunt_data = {
            'bus_numero': bus_num,
            'bus_index': self.bus_map.get(bus_num, -1),
            'id': self._convertir_tipo(partes[1], 'str') if len(partes) > 1 else '1',
            'status': self._convertir_tipo(partes[2], 'int') if len(partes) > 2 else 1,
            'G_MW': self._convertir_tipo(partes[3], 'float') if len(partes) > 3 else 0.0,
            'B_MVAR': self._convertir_tipo(partes[4], 'float') if len(partes) > 4 else 0.0,
            'G_pu': 0.0,
            'B_pu': 0.0
        }
        
        # Convertir a pu
        shunt_data['G_pu'] = shunt_data['G_MW'] / self.base_mva
        shunt_data['B_pu'] = shunt_data['B_MVAR'] / self.base_mva
        
        self.shunts.append(shunt_data)
    
    def obtener_dataframes(self):
        """
        Convierte los datos parseados a DataFrames de pandas
        """
        # Agregar nombres de buses a otros elementos
        for carga in self.cargas:
            bus_idx = carga['bus_index']
            if 0 <= bus_idx < len(self.buses):
                carga['bus_nombre'] = self.buses[bus_idx]['nombre']
            else:
                carga['bus_nombre'] = f"Bus_{carga['bus_numero']}"
        
        for gen in self.generadores:
            bus_idx = gen['bus_index']
            if 0 <= bus_idx < len(self.buses):
                gen['bus_nombre'] = self.buses[bus_idx]['nombre']
            else:
                gen['bus_nombre'] = f"Bus_{gen['bus_numero']}"
        
        for linea in self.lineas:
            from_idx = linea['from_bus_index']
            to_idx = linea['to_bus_index']
            if 0 <= from_idx < len(self.buses):
                linea['from_bus_nombre'] = self.buses[from_idx]['nombre']
            else:
                linea['from_bus_nombre'] = f"Bus_{linea['from_bus_numero']}"
            
            if 0 <= to_idx < len(self.buses):
                linea['to_bus_nombre'] = self.buses[to_idx]['nombre']
            else:
                linea['to_bus_nombre'] = f"Bus_{linea['to_bus_numero']}"
        
        for trafo in self.transformadores:
            from_idx = trafo['from_bus_index']
            to_idx = trafo['to_bus_index']
            if 0 <= from_idx < len(self.buses):
                trafo['from_bus_nombre'] = self.buses[from_idx]['nombre']
            else:
                trafo['from_bus_nombre'] = f"Bus_{trafo['from_bus_numero']}"
            
            if 0 <= to_idx < len(self.buses):
                trafo['to_bus_nombre'] = self.buses[to_idx]['nombre']
            else:
                trafo['to_bus_nombre'] = f"Bus_{trafo['to_bus_numero']}"
        
        return {
            'base_MVA': self.base_mva,
            'frecuencia_Hz': self.frecuencia,
            'version': self.version,
            'buses': pd.DataFrame(self.buses),
            'cargas': pd.DataFrame(self.cargas),
            'generadores': pd.DataFrame(self.generadores),
            'lineas': pd.DataFrame(self.lineas),
            'transformadores': pd.DataFrame(self.transformadores),
            'shunts': pd.DataFrame(self.shunts)
        }


def leer_raw(archivo_raw):
    """
    Función principal para leer archivos .raw
    """
    parser = RawParser()
    parser.leer_archivo(archivo_raw)
    datos = parser.obtener_dataframes()
    
    print(f"\nBUSES: {len(datos['buses'])}")
    print(datos['buses'].head())
    
    print(f"\nCARGAS: {len(datos['cargas'])}")
    print(datos['cargas'].head())
    
    print(f"\nGENERADORES: {len(datos['generadores'])}")
    print(datos['generadores'].head())
    
    print(f"\nLÍNEAS: {len(datos['lineas'])}")
    print(datos['lineas'].head())
    
    print(f"\nTRANSFORMADORES: {len(datos['transformadores'])}")
    print(datos['transformadores'].head())
    
    if len(datos['shunts']) > 0:
        print(f"\nSHUNTS: {len(datos['shunts'])}")
        print(datos['shunts'].head())
    
    return datos


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def guardar_datos_csv(datos, carpeta_salida='datos_raw'):
    """Guarda todos los DataFrames en archivos CSV"""
    import os
    
    os.makedirs(carpeta_salida, exist_ok=True)
    
    for nombre, valor in datos.items():
        if isinstance(valor, pd.DataFrame):
            archivo = os.path.join(carpeta_salida, f"{nombre}.csv")
            valor.to_csv(archivo, index=False)
            print(f"Guardado: {archivo}")


def guardar_datos_excel(datos, archivo_salida='datos_sistema.xlsx'):
    """Guarda todos los DataFrames en un archivo Excel"""
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        for nombre, valor in datos.items():
            if isinstance(valor, pd.DataFrame):
                valor.to_excel(writer, sheet_name=nombre, index=False)
    
    print(f"Guardado: {archivo_salida}")


def resumen_sistema(datos):
    """Imprime un resumen del sistema"""
    print("\n" + "="*80)
    print(" RESUMEN DEL SISTEMA")
    print("="*80)
    
    print(f"Base MVA: {datos['base_MVA']}")
    print(f"Frecuencia: {datos['frecuencia_Hz']} Hz")
    print(f"Versión PSS/E: {datos['version']}")
    
    print(f"\nNúmero de Buses: {len(datos['buses'])}")
    print(f"Número de Líneas: {len(datos['lineas'])}")
    print(f"Número de Transformadores: {len(datos['transformadores'])}")
    print(f"Número de Cargas: {len(datos['cargas'])}")
    print(f"Número de Generadores: {len(datos['generadores'])}")
    print(f"Número de Shunts: {len(datos['shunts'])}")
    
    # Estadísticas de potencia
    if len(datos['cargas']) > 0:
        P_carga_total = datos['cargas']['P_MW'].sum()
        Q_carga_total = datos['cargas']['Q_MVAR'].sum()
        print(f"\nPotencia de Carga Total: {P_carga_total:.2f} MW, {Q_carga_total:.2f} MVAR")
    
    if len(datos['generadores']) > 0:
        P_gen_total = datos['generadores']['P_MW'].sum()
        print(f"Potencia de Generación Total: {P_gen_total:.2f} MW")
        
        if len(datos['cargas']) > 0:
            print(f"Diferencia Gen-Carga: {P_gen_total - P_carga_total:.2f} MW")
    
    # Buses slack
    buses_slack = datos['buses'][datos['buses']['is_slack'] == True]
    print(f"\nBuses Slack: {len(buses_slack)}")
    if len(buses_slack) > 0:
        print(buses_slack[['numero', 'nombre', 'V_base_kV']])


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    
    # archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 14 bus.raw"  # Reemplaza con tu archivo
    archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 118 Bus v2.raw"  # Reemplaza con tu archivo
    
    print("="*80)
    print(" PARSER PERSONALIZADO DE ARCHIVOS .RAW")
    print("="*80)
    print()
    
    try:
        # Leer archivo
        datos = leer_raw(archivo_raw)
        
        # Mostrar resumen
        resumen_sistema(datos)
        
        # Guardar datos
        print("\n" + "="*80)
        print("Guardando datos...")
        # guardar_datos_csv(datos, carpeta_salida='datos_parseados')
        # guardar_datos_excel(datos, archivo_salida='sistema_raw.xlsx')
        
        # Ejemplos de análisis
        print("\n" + "="*80)
        print(" EJEMPLOS DE ANÁLISIS")
        print("="*80)
        
        # Líneas más cargadas (por impedancia)
        if len(datos['lineas']) > 0:
            print("\nTop 5 líneas con mayor reactancia:")
            top_lineas = datos['lineas'].nlargest(5, 'X_pu')[
                ['from_bus_nombre', 'to_bus_nombre', 'R_pu', 'X_pu', 'rate_A_MVA']
            ]
            print(top_lineas)
        
        # Cargas más grandes
        if len(datos['cargas']) > 0:
            print("\nTop 5 cargas más grandes:")
            top_cargas = datos['cargas'].nlargest(5, 'P_MW')[
                ['bus_nombre', 'P_MW', 'Q_MVAR']
            ]
            print(top_cargas)
        
        # Generadores
        if len(datos['generadores']) > 0:
            print("\nGeneradores:")
            gens = datos['generadores'][
                ['bus_nombre', 'P_MW', 'Q_MVAR', 'V_set_pu', 'is_slack']
            ]
            print(gens)
        
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo '{archivo_raw}'")
        print("\nDescarga casos de prueba desde:")
        print("  - https://github.com/SanPen/VeraGrid/tree/master/Grids_and_profiles/grids")
        print("  - https://electricgrids.engr.tamu.edu/")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()