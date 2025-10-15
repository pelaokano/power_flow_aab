"""
FLUJO DE POTENCIA TRIFÁSICO DESEQUILIBRADO
Método Newton-Raphson en coordenadas de fase (abc)
Soporta cargas y líneas desequilibradas
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, bmat
import warnings
warnings.filterwarnings('ignore')


class PowerFlowTrifasico:
    """
    Solver de flujo de potencia trifásico desequilibrado usando Newton-Raphson
    Trabaja directamente en coordenadas de fase (a, b, c)
    """
    
    def __init__(self, datos_sistema, configuracion_trifasica=None):
        """
        Inicializa el solver trifásico
        
        Args:
            datos_sistema: diccionario del RawParser.obtener_dataframes()
            configuracion_trifasica: dict con configuración de cargas y líneas por fase
        """
        self.base_mva = datos_sistema['base_MVA']
        self.frecuencia = datos_sistema['frecuencia_Hz']
        
        # DataFrames del sistema balanceado
        self.df_buses = datos_sistema['buses']
        self.df_cargas = datos_sistema['cargas']
        self.df_generadores = datos_sistema['generadores']
        self.df_lineas = datos_sistema['lineas']
        self.df_transformadores = datos_sistema['transformadores']
        self.df_shunts = datos_sistema['shunts']
        
        # Variables del sistema
        self.n_buses = len(self.df_buses)
        self.n_nodes = self.n_buses * 3  # 3 nodos por bus (a, b, c)
        
        # Matriz de admitancia trifásica (3N x 3N)
        self.Ybus_3ph = None
        
        # Mapeo número de bus -> índice
        self.bus_num_to_idx = {num: idx for idx, num in enumerate(self.df_buses['numero'])}
        self.idx_to_bus_num = {idx: num for num, idx in self.bus_num_to_idx.items()}
        
        # Voltajes por fase (cada bus tiene Va, Vb, Vc)
        # Inicializar con voltajes balanceados
        self.V_complex = self._inicializar_voltajes()
        
        # Clasificación de buses
        self.bus_types = None
        self.pq_buses = []
        self.pv_buses = []
        self.slack_bus = None
        
        # Potencias especificadas por fase
        self.P_spec = np.zeros(self.n_nodes)  # [Pa1, Pb1, Pc1, Pa2, ...]
        self.Q_spec = np.zeros(self.n_nodes)
        
        # Configuración trifásica
        self.config_3ph = configuracion_trifasica if configuracion_trifasica else {}
        
        print(f"Sistema trifásico inicializado:")
        print(f"  - Buses monofásicos: {self.n_buses}")
        print(f"  - Nodos trifásicos: {self.n_nodes}")
        print(f"  - Base MVA: {self.base_mva}")
        
    def _inicializar_voltajes(self):
        """
        Inicializa voltajes con secuencia balanceada
        Va = V∠0°, Vb = V∠-120°, Vc = V∠120°
        """
        V = np.zeros(self.n_nodes, dtype=complex)
        
        # Ángulos de secuencia balanceada
        ang_a = 0.0
        ang_b = -2 * np.pi / 3  # -120°
        ang_c = 2 * np.pi / 3   # 120°
        
        for i in range(self.n_buses):
            V_mag = self.df_buses.iloc[i]['V_mag_pu']
            
            # Índices para las tres fases del bus i
            idx_a = 3 * i
            idx_b = 3 * i + 1
            idx_c = 3 * i + 2
            
            V[idx_a] = V_mag * np.exp(1j * ang_a)
            V[idx_b] = V_mag * np.exp(1j * ang_b)
            V[idx_c] = V_mag * np.exp(1j * ang_c)
        
        return V
    
    def construir_ybus_trifasica(self):
        """
        Construye la matriz de admitancias trifásica (3N x 3N)
        Cada elemento es una matriz 3x3 que representa acoplamiento entre fases
        """
        print("\nConstruyendo matriz Ybus trifásica...")
        
        # Inicializar matriz dispersa 3N x 3N
        Y = lil_matrix((self.n_nodes, self.n_nodes), dtype=complex)
        
        # Agregar líneas de transmisión
        if self.df_lineas is not None and len(self.df_lineas) > 0:
            for idx_linea, linea in self.df_lineas.iterrows():
                if linea['status'] != 1:
                    continue
                
                i = self.bus_num_to_idx[linea['from_bus_numero']]
                j = self.bus_num_to_idx[linea['to_bus_numero']]
                
                # Obtener matriz de impedancia 3x3 de la línea
                Z_abc = self._calcular_impedancia_linea(linea, idx_linea)
                
                # Invertir para obtener admitancia
                Y_abc = np.linalg.inv(Z_abc)
                
                # Admitancia shunt (capacitancia)
                B_shunt = linea['B_pu'] / 2.0
                Y_shunt_abc = np.diag([1j * B_shunt] * 3)
                
                # Llenar submatrices 3x3 en la matriz grande
                for fi in range(3):  # fase i
                    for fj in range(3):  # fase j
                        idx_i = 3*i + fi
                        idx_j = 3*j + fj
                        
                        # Elementos diagonales (auto-admitancia)
                        Y[idx_i, idx_i] += Y_abc[fi, fi] + Y_shunt_abc[fi, fi]
                        Y[idx_j, idx_j] += Y_abc[fj, fj] + Y_shunt_abc[fj, fj]
                        
                        # Elementos fuera de diagonal (admitancia mutua)
                        Y[idx_i, idx_j] -= Y_abc[fi, fj]
                        Y[idx_j, idx_i] -= Y_abc[fj, fi]
        
        # Agregar transformadores (simplificado como Y-Y balanceado)
        if self.df_transformadores is not None and len(self.df_transformadores) > 0:
            for _, trafo in self.df_transformadores.iterrows():
                if trafo['STAT'] != 1:
                    continue
                
                i = self.bus_num_to_idx[trafo['from_bus_index']]
                j = self.bus_num_to_idx[trafo['to_bus_index']]
                
                # Impedancia base del transformador
                z_base = trafo['R1_2_pu'] + 1j * trafo['X1_2_pu']
                if abs(z_base) < 1e-10:
                    z_base = 1e-10
                
                y_trafo = 1.0 / z_base
                
                # Matriz de admitancia 3x3 (diagonal para Y-Y balanceado)
                Y_trafo_abc = np.diag([y_trafo] * 3)
                
                # Llenar submatrices
                for f in range(3):
                    idx_i = 3*i + f
                    idx_j = 3*j + f
                    
                    Y[idx_i, idx_i] += Y_trafo_abc[f, f]
                    Y[idx_j, idx_j] += Y_trafo_abc[f, f]
                    Y[idx_i, idx_j] -= Y_trafo_abc[f, f]
                    Y[idx_j, idx_i] -= Y_trafo_abc[f, f]
        
        # Agregar shunts (balanceados)
        if self.df_shunts is not None and len(self.df_shunts) > 0:
            for _, shunt in self.df_shunts.iterrows():
                if shunt['status'] != 1:
                    continue
                
                i = self.bus_num_to_idx[shunt['bus_numero']]
                y_shunt = shunt['GL_pu'] + 1j * shunt['BL_pu']
                
                # Aplicar a las tres fases
                for f in range(3):
                    idx = 3*i + f
                    Y[idx, idx] += y_shunt
        
        self.Ybus_3ph = Y.tocsr()
        print(f"Ybus trifásica construida: {self.n_nodes}x{self.n_nodes}")
        print(f"Elementos no-cero: {self.Ybus_3ph.nnz}")
    
    def _calcular_impedancia_linea(self, linea, idx_linea):
        """
        Calcula la matriz de impedancia 3x3 de una línea
        Incluye impedancias propias y mutuas entre fases
        """
        # Obtener configuración específica de la línea si existe
        config_key = f"linea_{idx_linea}"
        
        if config_key in self.config_3ph and 'Z_abc' in self.config_3ph[config_key]:
            # Usar matriz de impedancia especificada
            return self.config_3ph[config_key]['Z_abc']
        else:
            # Usar modelo simplificado: líneas balanceadas
            z_self = linea['R_pu'] + 1j * linea['X_pu']
            
            # Impedancia mutua (aproximación típica: 60% de la impedancia propia)
            z_mutual = 0.6 * z_self
            
            # Matriz de impedancia 3x3
            Z_abc = np.array([
                [z_self, z_mutual, z_mutual],
                [z_mutual, z_self, z_mutual],
                [z_mutual, z_mutual, z_self]
            ], dtype=complex)
            
            return Z_abc
    
    def clasificar_buses(self):
        """
        Clasifica buses (similar al caso balanceado)
        En trifásico, cada bus tiene 3 nodos pero misma clasificación
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
                
                # Establecer voltajes de slack (balanceados)
                V_mag = row['V_mag_pu']
                V_ang = np.deg2rad(row['V_ang_deg'])
                
                self.V_complex[3*bus_idx] = V_mag * np.exp(1j * V_ang)
                self.V_complex[3*bus_idx + 1] = V_mag * np.exp(1j * (V_ang - 2*np.pi/3))
                self.V_complex[3*bus_idx + 2] = V_mag * np.exp(1j * (V_ang + 2*np.pi/3))
                
            elif tipo_bus == 2 or bus_num in buses_con_gen:  # PV
                self.bus_types[bus_idx] = 2
                self.pv_buses.append(bus_idx)
            else:  # PQ
                self.pq_buses.append(bus_idx)
        
        print(f"  Slack: 1 bus")
        print(f"  PV: {len(self.pv_buses)} buses")
        print(f"  PQ: {len(self.pq_buses)} buses")
    
    def calcular_potencias_especificadas(self):
        """
        Calcula potencias especificadas por fase
        Distribuye cargas entre fases según configuración
        """
        print("\nCalculando potencias especificadas por fase...")
        
        self.P_spec = np.zeros(self.n_nodes)
        self.Q_spec = np.zeros(self.n_nodes)
        
        # Agregar generación (balanceada por defecto)
        if self.df_generadores is not None:
            for _, gen in self.df_generadores.iterrows():
                if gen['status'] == 1:
                    bus_idx = self.bus_num_to_idx[gen['bus_numero']]
                    
                    # Distribuir igualmente entre fases
                    P_per_fase = gen['P_pu'] / 3.0
                    Q_per_fase = gen['Q_pu'] / 3.0
                    
                    for f in range(3):
                        idx = 3*bus_idx + f
                        self.P_spec[idx] += P_per_fase
                        self.Q_spec[idx] += Q_per_fase
        
        # Agregar cargas (desequilibradas si se especifica)
        if self.df_cargas is not None:
            for idx_carga, carga in self.df_cargas.iterrows():
                if carga['status'] == 1:
                    bus_idx = self.bus_num_to_idx[carga['bus_numero']]
                    
                    # Verificar si hay configuración específica de desequilibrio
                    config_key = f"carga_{idx_carga}"
                    
                    if config_key in self.config_3ph:
                        # Usar distribución especificada
                        dist = self.config_3ph[config_key].get('distribucion', [1/3, 1/3, 1/3])
                        # print('Usar distribución especificada')
                    else:
                        # Usar distribución por defecto (puede ser desequilibrada)
                        dist = self.config_3ph.get('distribucion_default', [0.4, 0.35, 0.25])
                        # print('Usar distribución por defecto')

                    print(f"carga: {carga['bus_numero']}, P: {carga['PL_pu']} y Q: {carga['QL_pu']}")
                    # Distribuir carga entre fases
                    for f in range(3):
                        idx = 3*bus_idx + f
                        self.P_spec[idx] -= round(carga['PL_pu'] * dist[f], 4)
                        self.Q_spec[idx] -= round(carga['QL_pu'] * dist[f], 4)
                        print(f" fase: {f},  P_spec: {self.P_spec[idx]} y Q_spec: {self.Q_spec[idx]}")

        
        # Calcular totales por fase
        P_fase_a = sum(self.P_spec[i] for i in range(0, self.n_nodes, 3))
        P_fase_b = sum(self.P_spec[i] for i in range(1, self.n_nodes, 3))
        P_fase_c = sum(self.P_spec[i] for i in range(2, self.n_nodes, 3))
        
        print(f"  P fase A: {P_fase_a:.4f} pu ({P_fase_a * self.base_mva:.2f} MW)")
        print(f"  P fase B: {P_fase_b:.4f} pu ({P_fase_b * self.base_mva:.2f} MW)")
        print(f"  P fase C: {P_fase_c:.4f} pu ({P_fase_c * self.base_mva:.2f} MW)")
        print(f"  Total: {(P_fase_a + P_fase_b + P_fase_c):.4f} pu")
        
        # Calcular desequilibrio
        P_total = P_fase_a + P_fase_b + P_fase_c
        P_avg = P_total / 3
        deseq = max(abs(P_fase_a - P_avg), abs(P_fase_b - P_avg), abs(P_fase_c - P_avg))
        deseq_pct = (deseq / abs(P_avg)) * 100 if P_avg != 0 else 0
        
        print(f"  Desequilibrio: {deseq_pct:.2f}%")
    
    def calcular_potencias_inyectadas(self):
        """
        Calcula potencias inyectadas en cada nodo
        """
        I = self.Ybus_3ph.dot(self.V_complex)
        S = self.V_complex * np.conj(I)
        return S.real, S.imag
    
    def construir_jacobiano(self):
        """
        Construye la matriz Jacobiana para el sistema trifásico
        Mucho más grande que en el caso balanceado
        """
        # Identificar nodos PQ y PV
        nodos_P = []  # Nodos con ecuación de P
        nodos_Q = []  # Nodos con ecuación de Q
        
        for bus_idx in self.pv_buses + self.pq_buses:
            for f in range(3):  # Las 3 fases
                nodos_P.append(3*bus_idx + f)
        
        for bus_idx in self.pq_buses:
            for f in range(3):  # Solo PQ tienen ecuación de Q
                nodos_Q.append(3*bus_idx + f)
        
        n_eq_P = len(nodos_P)
        n_eq_Q = len(nodos_Q)
        n_eq = n_eq_P + n_eq_Q
        
        # Inicializar Jacobiano
        J = np.zeros((n_eq, n_eq))
        
        V = self.V_complex
        V_mag = np.abs(V)
        V_ang = np.angle(V)
        
        # Calcular corrientes
        I = self.Ybus_3ph.dot(V)
        
        # J1: dP/dθ
        for i_eq, i in enumerate(nodos_P):
            for j_eq, j in enumerate(nodos_P):
                if i == j:
                    # Elemento diagonal
                    Q_i = (V[i] * np.conj(I[i])).imag
                    Y_ii = self.Ybus_3ph[i, i]
                    J[i_eq, j_eq] = -Q_i - V_mag[i]**2 * Y_ii.imag
                else:
                    # Elemento fuera de diagonal
                    Y_ij = self.Ybus_3ph[i, j]
                    theta_ij = V_ang[i] - V_ang[j]
                    J[i_eq, j_eq] = V_mag[i] * V_mag[j] * (
                        Y_ij.real * np.sin(theta_ij) - Y_ij.imag * np.cos(theta_ij)
                    )
        
        # J2: dP/dV
        for i_eq, i in enumerate(nodos_P):
            for j_eq, j in enumerate(nodos_Q):
                if i == j:
                    # Elemento diagonal
                    P_i = (V[i] * np.conj(I[i])).real
                    Y_ii = self.Ybus_3ph[i, i]
                    J[i_eq, n_eq_P + j_eq] = P_i / V_mag[i] + V_mag[i] * Y_ii.real
                else:
                    # Elemento fuera de diagonal
                    Y_ij = self.Ybus_3ph[i, j]
                    theta_ij = V_ang[i] - V_ang[j]
                    J[i_eq, n_eq_P + j_eq] = V_mag[i] * (
                        Y_ij.real * np.cos(theta_ij) + Y_ij.imag * np.sin(theta_ij)
                    )
        
        # J3: dQ/dθ
        for i_eq, i in enumerate(nodos_Q):
            for j_eq, j in enumerate(nodos_P):
                if i == j:
                    # Elemento diagonal
                    P_i = (V[i] * np.conj(I[i])).real
                    Y_ii = self.Ybus_3ph[i, i]
                    J[n_eq_P + i_eq, j_eq] = P_i - V_mag[i]**2 * Y_ii.real
                else:
                    # Elemento fuera de diagonal
                    Y_ij = self.Ybus_3ph[i, j]
                    theta_ij = V_ang[i] - V_ang[j]
                    J[n_eq_P + i_eq, j_eq] = -V_mag[i] * V_mag[j] * (
                        Y_ij.real * np.cos(theta_ij) + Y_ij.imag * np.sin(theta_ij)
                    )
        
        # J4: dQ/dV
        for i_eq, i in enumerate(nodos_Q):
            for j_eq, j in enumerate(nodos_Q):
                if i == j:
                    # Elemento diagonal
                    Q_i = (V[i] * np.conj(I[i])).imag
                    Y_ii = self.Ybus_3ph[i, i]
                    J[n_eq_P + i_eq, n_eq_P + j_eq] = Q_i / V_mag[i] - V_mag[i] * Y_ii.imag
                else:
                    # Elemento fuera de diagonal
                    Y_ij = self.Ybus_3ph[i, j]
                    theta_ij = V_ang[i] - V_ang[j]
                    J[n_eq_P + i_eq, n_eq_P + j_eq] = V_mag[i] * (
                        Y_ij.real * np.sin(theta_ij) - Y_ij.imag * np.cos(theta_ij)
                    )
        
        return J, nodos_P, nodos_Q
    
    def resolver(self, max_iter=20, tolerancia=1e-6):
        """
        Resuelve el flujo de potencia trifásico usando Newton-Raphson
        """
        print("\n" + "="*80)
        print("RESOLVIENDO FLUJO DE POTENCIA TRIFÁSICO - NEWTON RAPHSON")
        print("="*80)
        
        # Preparar sistema
        self.construir_ybus_trifasica()
        self.clasificar_buses()
        self.calcular_potencias_especificadas()
        
        # Identificar nodos
        nodos_P = []
        nodos_Q = []
        
        for bus_idx in self.pv_buses + self.pq_buses:
            for f in range(3):
                nodos_P.append(3*bus_idx + f)
        
        for bus_idx in self.pq_buses:
            for f in range(3):
                nodos_Q.append(3*bus_idx + f)
        
        print(f"\nIterando (tolerancia: {tolerancia})...")
        print(f"{'Iter':<6} {'Max |ΔP|':<12} {'Max |ΔQ|':<12} {'Estado'}")
        print("-" * 50)
        
        for iteracion in range(max_iter):
            # Calcular desbalances
            P_calc, Q_calc = self.calcular_potencias_inyectadas()
            
            delta_P = self.P_spec[nodos_P] - P_calc[nodos_P]
            delta_Q = self.Q_spec[nodos_Q] - Q_calc[nodos_Q]
            
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
            J, nodos_P_j, nodos_Q_j = self.construir_jacobiano()
            
            # Resolver sistema lineal
            try:
                delta_x = np.linalg.solve(J, delta_f)
            except np.linalg.LinAlgError:
                print("\nError: Matriz Jacobiana singular")
                return False
            
            # Actualizar variables
            V_mag = np.abs(self.V_complex)
            V_ang = np.angle(self.V_complex)
            
            n_eq_P = len(nodos_P)
            V_ang[nodos_P] += delta_x[:n_eq_P]
            V_mag[nodos_Q] += delta_x[n_eq_P:]
            
            # Reconstruir voltajes complejos
            self.V_complex = V_mag * np.exp(1j * V_ang)
        
        print("\nNo convergió en el número máximo de iteraciones")
        return False
    
    def generar_reporte(self):
        """
        Genera reporte detallado por fase
        """
        print("\n" + "="*80)
        print("RESULTADOS DEL FLUJO DE POTENCIA TRIFÁSICO")
        print("="*80)
        
        V_mag = np.abs(self.V_complex)
        V_ang = np.rad2deg(np.angle(self.V_complex))
        
        # Crear DataFrame con resultados por fase
        resultados = []
        
        for i in range(self.n_buses):
            bus_num = self.idx_to_bus_num[i]
            tipo_str = {1: 'PQ', 2: 'PV', 3: 'Slack'}[self.bus_types[i]]
            
            for f, fase in enumerate(['A', 'B', 'C']):
                idx = 3*i + f
                
                resultados.append({
                    'Bus': bus_num,
                    'Fase': fase,
                    'Tipo': tipo_str if f == 0 else '',
                    'V (pu)': V_mag[idx],
                    'Ang (°)': V_ang[idx],
                    'P_spec (pu)': self.P_spec[idx],
                    'Q_spec (pu)': self.Q_spec[idx],
                })
        
        df_resultados = pd.DataFrame(resultados)
        
        print("\nVOLTAJES POR FASE:")
        print("-" * 80)
        print(df_resultados.to_string(index=False))
        
        # Análisis de desequilibrio
        self.analizar_desequilibrio(df_resultados)
        
        return df_resultados
    
    def analizar_desequilibrio(self, df_resultados):
        """
        Analiza el desequilibrio de voltajes por bus
        """
        print("\n\nANÁLISIS DE DESEQUILIBRIO:")
        print("-" * 80)
        
        desequilibrios = []
        
        for i in range(self.n_buses):
            bus_num = self.idx_to_bus_num[i]
            
            # Voltajes de las tres fases
            Va = self.V_complex[3*i]
            Vb = self.V_complex[3*i + 1]
            Vc = self.V_complex[3*i + 2]
            
            # Voltaje promedio
            V_avg = (Va + Vb + Vc) / 3
            
            # Desequilibrio (NEMA MG-1)
            if abs(V_avg) > 0:
                dev_max = max(abs(Va - V_avg), abs(Vb - V_avg), abs(Vc - V_avg))
                deseq_pct = (dev_max / abs(V_avg)) * 100
            else:
                deseq_pct = 0
            
            desequilibrios.append({
                'Bus': bus_num,
                'Desequilibrio (%)': deseq_pct,
                '|Va| (pu)': abs(Va),
                '|Vb| (pu)': abs(Vb),
                '|Vc| (pu)': abs(Vc),
            })
        
        df_deseq = pd.DataFrame(desequilibrios)
        
        # Mostrar buses con mayor desequilibrio
        df_deseq_sorted = df_deseq.sort_values('Desequilibrio (%)', ascending=False)
        print("\nTop 10 buses con mayor desequilibrio:")
        print(df_deseq_sorted.head(10).to_string(index=False))
        
        # Estadísticas generales
        print(f"\nEstadísticas de desequilibrio:")
        print(f"  Promedio: {df_deseq['Desequilibrio (%)'].mean():.2f}%")
        print(f"  Máximo: {df_deseq['Desequilibrio (%)'].max():.2f}%")
        print(f"  Buses con >5%: {len(df_deseq[df_deseq['Desequilibrio (%)'] > 5])}")


if __name__ == "__main__":
    print("Módulo de Flujo de Potencia Trifásico Desequilibrado")
    print("Usar en conjunto con RawParser")