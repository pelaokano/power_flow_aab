"""
FLUJO DE POTENCIA AC - MÉTODO GAUSS-SEIDEL
Compatible con parser de archivos .RAW de PSS/E
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
import warnings
warnings.filterwarnings('ignore')


class PowerFlowGS:
    """
    Solver de flujo de potencia AC usando Gauss-Seidel
    """
    
    def __init__(self, datos_sistema):
        """
        Inicializa el solver con datos del sistema
        
        Args:
            datos_sistema: diccionario retornado por RawParser.obtener_dataframes()
        """
        self.base_mva = datos_sistema['base_MVA']
        self.frecuencia = datos_sistema['frecuencia_Hz']
        
        # DataFrames
        self.df_buses = datos_sistema['buses']
        self.df_cargas = datos_sistema['cargas']
        self.df_generadores = datos_sistema['generadores']
        self.df_lineas = datos_sistema['lineas']
        self.df_transformadores = datos_sistema['transformadores']
        self.df_shunts = datos_sistema['shunts']
        
        # Variables del sistema
        self.n_buses = len(self.df_buses)
        self.Ybus = None
        
        # Mapeo número de bus -> índice
        self.bus_num_to_idx = {num: idx for idx, num in enumerate(self.df_buses['numero'])}
        self.idx_to_bus_num = {idx: num for num, idx in self.bus_num_to_idx.items()}
        
        # Vectores de voltaje
        self.V_complex = np.ones(self.n_buses, dtype=complex)  # Voltajes complejos
        
        # Clasificación de buses
        self.bus_types = None  # 1=PQ, 2=PV, 3=Slack
        self.pq_buses = []
        self.pv_buses = []
        self.slack_bus = None
        
        # Potencias especificadas
        self.P_spec = np.zeros(self.n_buses)
        self.Q_spec = np.zeros(self.n_buses)
        
        # Para rastrear convergencia
        self.iteraciones_realizadas = 0
        
        print(f"Sistema inicializado: {self.n_buses} buses")
        print(f"Base MVA: {self.base_mva}")
        
    def construir_ybus(self):
        """
        Construye la matriz de admitancias del sistema
        """
        print("\nConstruyendo matriz Ybus...")
        
        # Inicializar matriz dispersa
        Y = lil_matrix((self.n_buses, self.n_buses), dtype=complex)
        
        # Agregar líneas de transmisión
        if self.df_lineas is not None and len(self.df_lineas) > 0:
            for _, linea in self.df_lineas.iterrows():
                if linea['status'] != 1:
                    continue
                    
                i = self.bus_num_to_idx[linea['from_bus_numero']]
                j = self.bus_num_to_idx[linea['to_bus_numero']]
                
                # Admitancia serie
                z = linea['R_pu'] + 1j * linea['X_pu']
                if abs(z) < 1e-10:
                    z = 1e-10
                y_serie = 1.0 / z
                
                # Admitancia shunt
                y_shunt = 1j * linea['B_pu'] / 2.0
                
                # Shunts en nodos
                y_i = linea['GI_pu'] + 1j * linea['BI_pu']
                y_j = linea['GJ_pu'] + 1j * linea['BJ_pu']
                
                # Llenar matriz
                Y[i, i] += y_serie + y_shunt + y_i
                Y[j, j] += y_serie + y_shunt + y_j
                Y[i, j] -= y_serie
                Y[j, i] -= y_serie
        
        # Agregar transformadores
        if self.df_transformadores is not None and len(self.df_transformadores) > 0:
            for _, trafo in self.df_transformadores.iterrows():
                if trafo['STAT'] != 1:
                    continue
                
                i = self.bus_num_to_idx[trafo['from_bus_index']]
                j = self.bus_num_to_idx[trafo['to_bus_index']]
                
                z_base = trafo['R1_2_pu'] + 1j * trafo['X1_2_pu']
                if abs(z_base) < 1e-10:
                    z_base = 1e-10
                
                tap = trafo['WINDV1'] / trafo['WINDV2'] if trafo['WINDV2'] != 0 else 1.0
                ang_shift = np.deg2rad(trafo['ANG1'])
                tap_complex = tap * np.exp(1j * ang_shift)
                
                y_trafo = 1.0 / z_base
                
                Y[i, i] += y_trafo / (abs(tap_complex)**2)
                Y[j, j] += y_trafo
                Y[i, j] -= y_trafo / np.conj(tap_complex)
                Y[j, i] -= y_trafo / tap_complex
        
        # Agregar shunts fijos
        if self.df_shunts is not None and len(self.df_shunts) > 0:
            for _, shunt in self.df_shunts.iterrows():
                if shunt['status'] != 1:
                    continue
                    
                i = self.bus_num_to_idx[shunt['bus_numero']]
                y_shunt = shunt['GL_pu'] + 1j * shunt['BL_pu']
                Y[i, i] += y_shunt
        
        self.Ybus = Y.tocsr()
        print(f"Ybus construida: {self.n_buses}x{self.n_buses} (elementos no-cero: {self.Ybus.nnz})")
        
    def clasificar_buses(self):
        """
        Clasifica los buses en: Slack (3), PV (2), PQ (1)
        """
        print("\nClasificando buses...")
        
        self.bus_types = np.ones(self.n_buses, dtype=int)
        
        buses_con_gen = set()
        if self.df_generadores is not None:
            for _, gen in self.df_generadores.iterrows():
                if gen['status'] == 1:
                    buses_con_gen.add(gen['bus_numero'])
        
        for idx, row in self.df_buses.iterrows():
            bus_num = row['numero']
            bus_idx = self.bus_num_to_idx[bus_num]
            tipo_bus = row['tipo']
            
            if tipo_bus == 3:  # Slack
                self.bus_types[bus_idx] = 3
                self.slack_bus = bus_idx
                self.V_complex[bus_idx] = row['V_mag_pu'] * np.exp(1j * np.deg2rad(row['V_ang_deg']))
                
            elif tipo_bus == 2 or bus_num in buses_con_gen:  # PV
                self.bus_types[bus_idx] = 2
                self.pv_buses.append(bus_idx)
                self.V_complex[bus_idx] = row['V_mag_pu']  # Solo magnitud inicial
            else:  # PQ
                self.pq_buses.append(bus_idx)
                self.V_complex[bus_idx] = row['V_mag_pu']
        
        print(f"  Slack: 1 bus (bus #{self.idx_to_bus_num[self.slack_bus]})")
        print(f"  PV: {len(self.pv_buses)} buses")
        print(f"  PQ: {len(self.pq_buses)} buses")
        
    def calcular_potencias_especificadas(self):
        """
        Calcula las potencias netas especificadas en cada bus
        """
        print("\nCalculando potencias especificadas...")
        
        self.P_spec = np.zeros(self.n_buses)
        self.Q_spec = np.zeros(self.n_buses)
        
        # Agregar generación
        if self.df_generadores is not None:
            for _, gen in self.df_generadores.iterrows():
                if gen['status'] == 1:
                    bus_idx = self.bus_num_to_idx[gen['bus_numero']]
                    self.P_spec[bus_idx] += gen['P_pu']
                    self.Q_spec[bus_idx] += gen['Q_pu']
        
        # Restar cargas
        if self.df_cargas is not None:
            for _, carga in self.df_cargas.iterrows():
                if carga['status'] == 1:
                    bus_idx = self.bus_num_to_idx[carga['bus_numero']]
                    self.P_spec[bus_idx] -= carga['PL_pu']
                    self.Q_spec[bus_idx] -= carga['QL_pu']
        
        print(f"  P total generada: {sum(self.P_spec[self.P_spec > 0]):.4f} pu")
        print(f"  P total carga: {-sum(self.P_spec[self.P_spec < 0]):.4f} pu")
        
    def actualizar_voltaje_bus_pq(self, i, factor_aceleracion=1.0):
        """
        Actualiza el voltaje de un bus PQ usando Gauss-Seidel
        
        V_i^(k+1) = (1/Y_ii) * [(P_i - jQ_i)/V_i^* - sum(Y_ij * V_j)]
        """
        # Potencia especificada
        S_spec = self.P_spec[i] - 1j * self.Q_spec[i]
        
        # Sumatoria de corrientes de buses vecinos
        suma = 0.0
        for j in range(self.n_buses):
            if j != i:
                suma += self.Ybus[i, j] * self.V_complex[j]
        
        # Nueva estimación del voltaje
        Y_ii = self.Ybus[i, i]
        if abs(Y_ii) < 1e-10:
            return self.V_complex[i]
        
        V_nuevo = (1.0 / Y_ii) * (np.conj(S_spec / self.V_complex[i]) - suma)
        
        # Aplicar factor de aceleración
        if factor_aceleracion != 1.0:
            V_nuevo = self.V_complex[i] + factor_aceleracion * (V_nuevo - self.V_complex[i])
        
        return V_nuevo
    
    def actualizar_voltaje_bus_pv(self, i, V_mag_spec, factor_aceleracion=1.0):
        """
        Actualiza el voltaje de un bus PV manteniendo magnitud constante
        
        1. Calcular nuevo ángulo con ecuación de potencia activa
        2. Ajustar magnitud al valor especificado
        """
        # Potencia activa especificada
        P_spec = self.P_spec[i]
        
        # Calcular Q actual (necesaria para la iteración)
        suma = 0.0
        for j in range(self.n_buses):
            if j != i:
                suma += self.Ybus[i, j] * self.V_complex[j]
        
        # Calcular Q inyectada actual
        I_i = self.Ybus[i, i] * self.V_complex[i] + suma
        S_i = self.V_complex[i] * np.conj(I_i)
        Q_calc = S_i.imag
        
        # Usar Q calculada para actualizar voltaje
        S_spec = P_spec - 1j * Q_calc
        
        # Nueva estimación
        Y_ii = self.Ybus[i, i]
        if abs(Y_ii) < 1e-10:
            return self.V_complex[i]
        
        V_nuevo = (1.0 / Y_ii) * (np.conj(S_spec / self.V_complex[i]) - suma)
        
        # Aplicar factor de aceleración al ángulo
        if factor_aceleracion != 1.0:
            ang_nuevo = np.angle(V_nuevo)
            ang_viejo = np.angle(self.V_complex[i])
            ang_acelerado = ang_viejo + factor_aceleracion * (ang_nuevo - ang_viejo)
            V_nuevo = V_mag_spec * np.exp(1j * ang_acelerado)
        else:
            # Mantener magnitud especificada
            V_nuevo = V_mag_spec * np.exp(1j * np.angle(V_nuevo))
        
        return V_nuevo
    
    def resolver(self, max_iter=100, tolerancia=1e-6, factor_aceleracion=1.6):
        """
        Resuelve el flujo de potencia usando Gauss-Seidel
        
        Args:
            max_iter: número máximo de iteraciones
            tolerancia: tolerancia de convergencia
            factor_aceleracion: factor de aceleración (1.0 = sin aceleración, 1.4-1.6 típico)
        """
        print("\n" + "="*80)
        print("RESOLVIENDO FLUJO DE POTENCIA - GAUSS-SEIDEL")
        print("="*80)
        
        # Preparar sistema
        self.construir_ybus()
        self.clasificar_buses()
        self.calcular_potencias_especificadas()
        
        # Obtener magnitudes especificadas para buses PV
        V_mag_pv = {}
        for i in self.pv_buses:
            bus_num = self.idx_to_bus_num[i]
            bus_data = self.df_buses[self.df_buses['numero'] == bus_num].iloc[0]
            V_mag_pv[i] = bus_data['V_mag_pu']
        
        print(f"\nIterando (tolerancia: {tolerancia}, aceleración: {factor_aceleracion})...")
        print(f"{'Iter':<6} {'Max |ΔV|':<12} {'Estado'}")
        print("-" * 40)
        
        for iteracion in range(max_iter):
            V_anterior = self.V_complex.copy()
            
            # Actualizar buses PQ
            for i in self.pq_buses:
                self.V_complex[i] = self.actualizar_voltaje_bus_pq(i, factor_aceleracion)
            
            # Actualizar buses PV
            for i in self.pv_buses:
                self.V_complex[i] = self.actualizar_voltaje_bus_pv(i, V_mag_pv[i], factor_aceleracion)
            
            # Calcular cambio máximo
            delta_V = np.abs(self.V_complex - V_anterior)
            max_delta = np.max(delta_V)
            
            estado = "Convergió ✓" if max_delta < tolerancia else ""
            print(f"{iteracion+1:<6} {max_delta:<12.2e} {estado}")
            
            # Verificar convergencia
            if max_delta < tolerancia:
                print("\n¡Convergencia alcanzada!")
                self.iteraciones_realizadas = iteracion + 1
                return True
        
        print("\n⚠ No convergió en el número máximo de iteraciones")
        self.iteraciones_realizadas = max_iter
        return False
    
    def calcular_potencias_inyectadas(self):
        """
        Calcula las potencias inyectadas con voltajes actuales
        """
        I = self.Ybus.dot(self.V_complex)
        S = self.V_complex * np.conj(I)
        return S.real, S.imag
    
    def calcular_flujos(self):
        """
        Calcula flujos de potencia en líneas y transformadores
        """
        print("\nCalculando flujos en líneas...")
        
        flujos = []
        
        # Flujos en líneas
        if self.df_lineas is not None:
            for _, linea in self.df_lineas.iterrows():
                if linea['status'] != 1:
                    continue
                    
                i = self.bus_num_to_idx[linea['from_bus_numero']]
                j = self.bus_num_to_idx[linea['to_bus_numero']]
                
                # Admitancia
                z = linea['R_pu'] + 1j * linea['X_pu']
                y = 1.0 / z if abs(z) > 1e-10 else 0
                y_shunt = 1j * linea['B_pu'] / 2.0
                
                # Corriente de i a j
                I_ij = y * (self.V_complex[i] - self.V_complex[j]) + y_shunt * self.V_complex[i]
                S_ij = self.V_complex[i] * np.conj(I_ij) * self.base_mva
                
                # Corriente de j a i
                I_ji = y * (self.V_complex[j] - self.V_complex[i]) + y_shunt * self.V_complex[j]
                S_ji = self.V_complex[j] * np.conj(I_ji) * self.base_mva
                
                flujos.append({
                    'tipo': 'Línea',
                    'from_bus': linea['from_bus_numero'],
                    'to_bus': linea['to_bus_numero'],
                    'P_from_MW': S_ij.real,
                    'Q_from_MVAR': S_ij.imag,
                    'P_to_MW': S_ji.real,
                    'Q_to_MVAR': S_ji.imag,
                    'Perdidas_MW': S_ij.real + S_ji.real,
                    'Perdidas_MVAR': S_ij.imag + S_ji.imag,
                })
        
        return pd.DataFrame(flujos)
    
    def generar_reporte(self):
        """
        Genera un reporte completo de resultados
        """
        print("\n" + "="*80)
        print("RESULTADOS DEL FLUJO DE POTENCIA")
        print("="*80)
        
        # Extraer magnitud y ángulo
        V_mag = np.abs(self.V_complex)
        V_ang = np.angle(self.V_complex)
        
        # Voltajes en buses
        print("\nVOLTAJES EN BUSES:")
        print("-" * 80)
        resultados_buses = []
        
        for idx in range(self.n_buses):
            bus_num = self.idx_to_bus_num[idx]
            tipo_str = {1: 'PQ', 2: 'PV', 3: 'Slack'}[self.bus_types[idx]]
            
            resultados_buses.append({
                'Bus': bus_num,
                'Tipo': tipo_str,
                'V (pu)': V_mag[idx],
                'Ang (°)': np.rad2deg(V_ang[idx]),
                'P_gen (MW)': 0,
                'Q_gen (MVAR)': 0,
                'P_carga (MW)': 0,
                'Q_carga (MVAR)': 0,
            })
        
        df_resultados = pd.DataFrame(resultados_buses)
        
        # Agregar generación y carga
        if self.df_generadores is not None:
            for _, gen in self.df_generadores.iterrows():
                if gen['status'] == 1:
                    mask = df_resultados['Bus'] == gen['bus_numero']
                    df_resultados.loc[mask, 'P_gen (MW)'] += gen['P_MW']
                    df_resultados.loc[mask, 'Q_gen (MVAR)'] += gen['Q_MVAR']
        
        if self.df_cargas is not None:
            for _, carga in self.df_cargas.iterrows():
                if carga['status'] == 1:
                    mask = df_resultados['Bus'] == carga['bus_numero']
                    df_resultados.loc[mask, 'P_carga (MW)'] += carga['PL_MW']
                    df_resultados.loc[mask, 'Q_carga (MVAR)'] += carga['QL_MVAR']
        
        print(df_resultados.to_string(index=False))
        
        # Flujos en líneas
        df_flujos = self.calcular_flujos()
        if not df_flujos.empty:
            print("\n\nFLUJOS EN LÍNEAS:")
            print("-" * 80)
            print(df_flujos.to_string(index=False))
            
            perdidas_totales = df_flujos['Perdidas_MW'].sum()
            print(f"\nPérdidas totales: {perdidas_totales:.4f} MW")
            print(f"Iteraciones realizadas: {self.iteraciones_realizadas}")
        
        return df_resultados, df_flujos


# EJEMPLO DE USO
if __name__ == "__main__":
    print("Módulo de Flujo de Potencia AC - Gauss-Seidel")
    print("Usar en conjunto con RawParser para resolver sistemas de potencia")