"""
FLUJO DE POTENCIA AC - MÉTODO NEWTON-RAPHSON
Compatible con parser de archivos .RAW de PSS/E
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
import warnings
warnings.filterwarnings('ignore')


class PowerFlowNR:
    """
    Solver de flujo de potencia AC usando Newton-Raphson
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
        self.V_mag = np.ones(self.n_buses)  # Magnitudes en pu
        self.V_ang = np.zeros(self.n_buses)  # Ángulos en radianes
        
        # Clasificación de buses
        self.bus_types = None  # 1=PQ, 2=PV, 3=Slack
        self.pq_buses = []
        self.pv_buses = []
        self.slack_bus = None
        
        # Potencias especificadas
        self.P_spec = np.zeros(self.n_buses)
        self.Q_spec = np.zeros(self.n_buses)
        
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
                
                # Admitancia shunt (susceptancia capacitiva)
                y_shunt = 1j * linea['B_pu'] / 2.0
                
                # Shunts en nodos i y j (conductancias y susceptancias)
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
                
                # Impedancia del transformador
                z_base = trafo['R1_2_pu'] + 1j * trafo['X1_2_pu']
                if abs(z_base) < 1e-10:
                    z_base = 1e-10
                
                # Relación de transformación
                tap = trafo['WINDV1'] / trafo['WINDV2'] if trafo['WINDV2'] != 0 else 1.0
                ang_shift = np.deg2rad(trafo['ANG1'])
                tap_complex = tap * np.exp(1j * ang_shift)
                
                # Admitancia
                y_trafo = 1.0 / z_base
                
                # Modelo del transformador
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
        
        self.bus_types = np.ones(self.n_buses, dtype=int)  # Por defecto PQ
        
        # Identificar buses con generadores
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
                self.V_mag[bus_idx] = row['V_mag_pu']
                self.V_ang[bus_idx] = np.deg2rad(row['V_ang_deg'])
                
            elif tipo_bus == 2 or bus_num in buses_con_gen:  # PV
                self.bus_types[bus_idx] = 2
                self.pv_buses.append(bus_idx)
                self.V_mag[bus_idx] = row['V_mag_pu']
            else:  # PQ
                self.pq_buses.append(bus_idx)
                self.V_mag[bus_idx] = row['V_mag_pu']
        
        print(f"  Slack: 1 bus (bus #{self.idx_to_bus_num[self.slack_bus]})")
        print(f"  PV: {len(self.pv_buses)} buses")
        print(f"  PQ: {len(self.pq_buses)} buses")
        
    def calcular_potencias_especificadas(self):
        """
        Calcula las potencias netas especificadas en cada bus
        P_spec = P_gen - P_carga
        Q_spec = Q_gen - Q_carga
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
        
    def calcular_potencias_inyectadas(self):
        """
        Calcula las potencias inyectadas con voltajes actuales
        S_i = V_i * conj(sum(Y_ij * V_j))
        """
        V = self.V_mag * np.exp(1j * self.V_ang)
        I = self.Ybus.dot(V)
        S = V * np.conj(I)
        
        return S.real, S.imag
    
    def calcular_jacobiano(self):
        """
        Construye la matriz Jacobiana para Newton-Raphson
        J = [dP/dtheta  dP/dV]
            [dQ/dtheta  dQ/dV]
        """
        n = self.n_buses
        V = self.V_mag
        theta = self.V_ang
        
        # Inicializar submatrices
        n_pq = len(self.pq_buses)
        n_pv = len(self.pv_buses)
        n_eq = n_pq + n_pv  # Número de ecuaciones de potencia activa
        
        J = np.zeros((n_eq + n_pq, n_eq + n_pq))
        
        # Calcular corrientes inyectadas
        V_complex = V * np.exp(1j * theta)
        I = self.Ybus.dot(V_complex)
        
        # Índices de variables
        buses_P = self.pv_buses + self.pq_buses  # Buses con ecuación de P
        buses_Q = self.pq_buses  # Buses con ecuación de Q
        
        # J1: dP/dtheta
        for i_eq, i in enumerate(buses_P):
            for j_eq, j in enumerate(buses_P):
                if i == j:
                    J[i_eq, j_eq] = -self.calcular_Qi(i) - V[i]**2 * self.Ybus[i, i].imag
                else:
                    Y_ij = self.Ybus[i, j]
                    theta_ij = theta[i] - theta[j]
                    J[i_eq, j_eq] = V[i] * V[j] * (
                        Y_ij.real * np.sin(theta_ij) - Y_ij.imag * np.cos(theta_ij)
                    )
        
        # J2: dP/dV
        for i_eq, i in enumerate(buses_P):
            for j_eq, j in enumerate(buses_Q):
                if i == j:
                    J[i_eq, n_eq + j_eq] = self.calcular_Pi(i) / V[i] + V[i] * self.Ybus[i, i].real
                else:
                    Y_ij = self.Ybus[i, j]
                    theta_ij = theta[i] - theta[j]
                    J[i_eq, n_eq + j_eq] = V[i] * (
                        Y_ij.real * np.cos(theta_ij) + Y_ij.imag * np.sin(theta_ij)
                    )
        
        # J3: dQ/dtheta
        for i_eq, i in enumerate(buses_Q):
            for j_eq, j in enumerate(buses_P):
                if i == j:
                    J[n_eq + i_eq, j_eq] = self.calcular_Pi(i) - V[i]**2 * self.Ybus[i, i].real
                else:
                    Y_ij = self.Ybus[i, j]
                    theta_ij = theta[i] - theta[j]
                    J[n_eq + i_eq, j_eq] = -V[i] * V[j] * (
                        Y_ij.real * np.cos(theta_ij) + Y_ij.imag * np.sin(theta_ij)
                    )
        
        # J4: dQ/dV
        for i_eq, i in enumerate(buses_Q):
            for j_eq, j in enumerate(buses_Q):
                if i == j:
                    J[n_eq + i_eq, n_eq + j_eq] = self.calcular_Qi(i) / V[i] - V[i] * self.Ybus[i, i].imag
                else:
                    Y_ij = self.Ybus[i, j]
                    theta_ij = theta[i] - theta[j]
                    J[n_eq + i_eq, n_eq + j_eq] = V[i] * (
                        Y_ij.real * np.sin(theta_ij) - Y_ij.imag * np.cos(theta_ij)
                    )
        
        return J
    
    def calcular_Pi(self, i):
        """Calcula P inyectada en bus i"""
        V = self.V_mag
        theta = self.V_ang
        P = 0
        for j in range(self.n_buses):
            Y_ij = self.Ybus[i, j]
            theta_ij = theta[i] - theta[j]
            P += V[j] * (Y_ij.real * np.cos(theta_ij) + Y_ij.imag * np.sin(theta_ij))
        return V[i] * P
    
    def calcular_Qi(self, i):
        """Calcula Q inyectada en bus i"""
        V = self.V_mag
        theta = self.V_ang
        Q = 0
        for j in range(self.n_buses):
            Y_ij = self.Ybus[i, j]
            theta_ij = theta[i] - theta[j]
            Q += V[j] * (Y_ij.real * np.sin(theta_ij) - Y_ij.imag * np.cos(theta_ij))
        return V[i] * Q
    
    def resolver(self, max_iter=20, tolerancia=1e-6):
        """
        Resuelve el flujo de potencia usando Newton-Raphson
        """
        print("\n" + "="*80)
        print("RESOLVIENDO FLUJO DE POTENCIA - NEWTON RAPHSON")
        print("="*80)
        
        # Preparar sistema
        self.construir_ybus()
        self.clasificar_buses()
        self.calcular_potencias_especificadas()
        
        buses_P = self.pv_buses + self.pq_buses
        buses_Q = self.pq_buses
        
        print(f"\nIterando (tolerancia: {tolerancia})...")
        print(f"{'Iter':<6} {'Max |ΔP|':<12} {'Max |ΔQ|':<12} {'Estado'}")
        print("-" * 50)
        
        for iteracion in range(max_iter):
            # Calcular desbalances
            P_calc, Q_calc = self.calcular_potencias_inyectadas()
            
            delta_P = self.P_spec[buses_P] - P_calc[buses_P]
            delta_Q = self.Q_spec[buses_Q] - Q_calc[buses_Q]
            
            # Vector de desbalances
            delta_f = np.concatenate([delta_P, delta_Q])
            
            # Verificar convergencia
            max_dP = np.max(np.abs(delta_P))
            max_dQ = np.max(np.abs(delta_Q)) if len(delta_Q) > 0 else 0
            
            estado = "Convergió" if max(max_dP, max_dQ) < tolerancia else ""
            print(f"{iteracion+1:<6} {max_dP:<12.2e} {max_dQ:<12.2e} {estado}")
            
            if max(max_dP, max_dQ) < tolerancia:
                print("\n¡Convergencia alcanzada!")
                return True
            
            # Calcular Jacobiano
            J = self.calcular_jacobiano()
            
            # Resolver sistema lineal
            try:
                delta_x = np.linalg.solve(J, delta_f)
            except np.linalg.LinAlgError:
                print("\nError: Matriz Jacobiana singular")
                return False
            
            # Actualizar variables
            n_eq = len(buses_P)
            self.V_ang[buses_P] += delta_x[:n_eq]
            self.V_mag[buses_Q] += delta_x[n_eq:]
        
        print("\n⚠ No convergió en el número máximo de iteraciones")
        return False
    
    def calcular_flujos(self):
        """
        Calcula flujos de potencia en líneas y transformadores
        """
        print("\nCalculando flujos en líneas...")
        
        V_complex = self.V_mag * np.exp(1j * self.V_ang)
        
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
                I_ij = y * (V_complex[i] - V_complex[j]) + y_shunt * V_complex[i]
                S_ij = V_complex[i] * np.conj(I_ij) * self.base_mva
                
                # Corriente de j a i
                I_ji = y * (V_complex[j] - V_complex[i]) + y_shunt * V_complex[j]
                S_ji = V_complex[j] * np.conj(I_ji) * self.base_mva
                
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
                'V (pu)': self.V_mag[idx],
                'Ang (°)': np.rad2deg(self.V_ang[idx]),
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
        
        return df_resultados, df_flujos


# EJEMPLO DE USO
if __name__ == "__main__":
    # Este código requiere que primero uses el RawParser
    # Ejemplo:
    #
    # from raw_parser import RawParser
    # 
    # parser = RawParser()
    # parser.leer_archivo("IEEE_14_bus.raw")
    # datos_sistema = parser.obtener_dataframes()
    # 
    # solver = PowerFlowNR(datos_sistema)
    # if solver.resolver():
    #     df_buses, df_flujos = solver.generar_reporte()
    
    print("Módulo de Flujo de Potencia AC - Newton Raphson")
    print("Usar en conjunto con RawParser para resolver sistemas de potencia")