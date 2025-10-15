"""
DC POWER FLOW - Compatible con RawParser
Implementación del flujo de potencia DC linealizado
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class DCPowerFlow:
    """
    Flujo de Potencia DC (Linealizado)
    
    Suposiciones:
    - Voltajes ≈ 1.0 pu
    - Ángulos pequeños (sin(θ) ≈ θ, cos(θ) ≈ 1)
    - Resistencias despreciables (R << X)
    - Solo potencia activa
    """
    
    def __init__(self, raw_data):
        """
        Inicializa con datos del RawParser
        
        Args:
            raw_data: Diccionario con DataFrames del RawParser
        """
        self.base_mva = raw_data['base_MVA']
        self.frequency = raw_data['frecuencia_Hz']
        
        # DataFrames originales
        self.df_buses = raw_data['buses'].copy()
        self.df_lines = raw_data['lineas'].copy() if raw_data['lineas'] is not None else pd.DataFrame()
        self.df_transformers = raw_data['transformadores'].copy() if raw_data['transformadores'] is not None else pd.DataFrame()
        self.df_loads = raw_data['cargas'].copy() if raw_data['cargas'] is not None else pd.DataFrame()
        self.df_generators = raw_data['generadores'].copy() if raw_data['generadores'] is not None else pd.DataFrame()
        
        # Crear mapeo de número de bus a índice
        self._crear_mapeo_buses()
        
        self.num_buses = len(self.df_buses)
        
        # Matrices del sistema
        self.B_matrix = None  # Matriz de susceptancias
        self.P_injection = None  # Vector de inyecciones de potencia
        
        # Resultados
        self.theta = None  # Ángulos de voltaje (radianes)
        self.theta_deg = None  # Ángulos de voltaje (grados)
        self.P_flows = None  # Flujos de potencia en ramas
        self.converged = False
        
        # self._mostrar_info_sistema()
    
    def _crear_mapeo_buses(self):
        """
        Crea mapeo entre número de bus y su índice en el array
        """
        # Resetear índices para que coincidan con la posición
        self.df_buses = self.df_buses.reset_index(drop=True)
        
        # Crear diccionario: numero_bus -> indice
        self.bus_num_to_idx = {bus_num: idx for idx, bus_num in enumerate(self.df_buses['numero'])}
        
        # Agregar columna de índice
        self.df_buses['idx'] = range(len(self.df_buses))
        
        # Identificar bus slack (tipo == 3)
        self.df_buses['is_slack'] = self.df_buses['tipo'] == 3
        
        # Mapear índices en cargas
        if not self.df_loads.empty:
            self.df_loads['bus_idx'] = self.df_loads['bus_numero'].map(self.bus_num_to_idx)
        
        # Mapear índices en generadores
        if not self.df_generators.empty:
            self.df_generators['bus_idx'] = self.df_generators['bus_numero'].map(self.bus_num_to_idx)
        
        # Mapear índices en líneas
        if not self.df_lines.empty:
            self.df_lines['from_idx'] = self.df_lines['from_bus_numero'].map(self.bus_num_to_idx)
            self.df_lines['to_idx'] = self.df_lines['to_bus_numero'].map(self.bus_num_to_idx)
        
        # Mapear índices en transformadores
        if not self.df_transformers.empty:
            self.df_transformers['from_idx'] = self.df_transformers['from_bus_index'].map(self.bus_num_to_idx)
            self.df_transformers['to_idx'] = self.df_transformers['to_bus_index'].map(self.bus_num_to_idx)
    
    def _mostrar_info_sistema(self):
        """Muestra información del sistema"""
        print(f"\n{'='*80}")
        print(f"  SISTEMA ELÉCTRICO INICIALIZADO")
        print(f"{'='*80}")
        print(f"  Base MVA:        {self.base_mva:.1f}")
        print(f"  Frecuencia:      {self.frequency:.1f} Hz")
        print(f"  Buses:           {self.num_buses}")
        print(f"  Líneas:          {len(self.df_lines)}")
        print(f"  Transformadores: {len(self.df_transformers)}")
        print(f"  Cargas:          {len(self.df_loads)}")
        print(f"  Generadores:     {len(self.df_generators)}")
        
        # Verificar bus slack
        slack_buses = self.df_buses[self.df_buses['is_slack']]
        if len(slack_buses) > 0:
            slack_info = slack_buses.iloc[0]
            print(f"  Bus Slack:       {slack_info['numero']} - {slack_info['nombre']}")
        else:
            print(f"ADVERTENCIA: No se encontró bus slack (tipo=3)")
        print(f"{'='*80}\n")
    
    def construir_matriz_susceptancias(self):
        """
        Construye la matriz B de susceptancias del sistema
        B[i,j] = -1/X_ij (elementos fuera de la diagonal)
        B[i,i] = suma de susceptancias conectadas al bus i
        """
        print("Construyendo matriz de susceptancias B...")
        
        # Usar matriz dispersa para eficiencia
        B = lil_matrix((self.num_buses, self.num_buses))
        
        # Procesar líneas
        for idx, line in self.df_lines.iterrows():
            if line['status'] == 0:  # Línea fuera de servicio
                continue
            
            i = line['from_idx']
            j = line['to_idx']
            
            if pd.isna(i) or pd.isna(j):
                continue
            
            i, j = int(i), int(j)
            
            # Susceptancia = 1/X (en flujo DC ignoramos R)
            X = line['X_pu']
            
            if abs(X) < 1e-10:  # Evitar división por cero
                print(f"  ⚠ Línea {line['from_bus_numero']}-{line['to_bus_numero']}: X muy pequeño, ignorando")
                continue
            
            b_ij = 1.0 / X
            
            # Elementos fuera de la diagonal
            B[i, j] -= b_ij
            B[j, i] -= b_ij
            
            # Elementos de la diagonal
            B[i, i] += b_ij
            B[j, j] += b_ij
        
        # Procesar transformadores
        for idx, trafo in self.df_transformers.iterrows():
            if trafo.get('STAT', 1) == 0:  # Transformador fuera de servicio
                continue
            
            i = trafo['from_idx']
            j = trafo['to_idx']
            tertiary = trafo.get('tertiary_bus_index', 0)
            
            if pd.isna(i) or pd.isna(j):
                continue
            
            i, j = int(i), int(j)
            
            # Verificar si es transformador de 2 o 3 devanados
            es_tres_devanados = tertiary > 0
            
            if not es_tres_devanados:
                # TRANSFORMADOR DE 2 DEVANADOS
                # X1_2_pu es la impedancia total (equivalente a X_pu)
                X = trafo['X1_2_pu']
                
                if abs(X) < 1e-10:
                    print(f"  ⚠ Transformador 2D {trafo['from_bus_index']}-{trafo['to_bus_index']}: X muy pequeño, ignorando")
                    continue
                
                b_ij = 1.0 / X
                
                # En flujo DC simplificado, ignoramos taps y ángulos de fase
                B[i, j] -= b_ij
                B[j, i] -= b_ij
                B[i, i] += b_ij
                B[j, j] += b_ij
            
            else:
                # TRANSFORMADOR DE 3 DEVANADOS
                # Modelo estrella: convertir impedancias a modelo PI equivalente
                # Z1, Z2, Z3 desde las impedancias medidas
                X12 = trafo['X1_2_pu']
                X23 = trafo['X2_3_pu']
                X31 = trafo['X3_1_pu']
                
                # Conversión delta a estrella
                X1 = (X12 + X31 - X23) / 2
                X2 = (X12 + X23 - X31) / 2
                X3 = (X23 + X31 - X12) / 2
                
                k = trafo['to_idx']
                if pd.isna(k):
                    continue
                k = int(k)
                
                # Agregar las tres ramas del modelo estrella
                if abs(X1) > 1e-10:
                    b1 = 1.0 / X1
                    B[i, k] -= b1
                    B[k, i] -= b1
                    B[i, i] += b1
                    B[k, k] += b1
                
                if abs(X2) > 1e-10:
                    b2 = 1.0 / X2
                    B[j, k] -= b2
                    B[k, j] -= b2
                    B[j, j] += b2
                    B[k, k] += b2
                
                if abs(X3) > 1e-10:
                    tertiary_idx = self.bus_num_to_idx.get(tertiary, None)
                    if tertiary_idx is not None:
                        t = int(tertiary_idx)
                        b3 = 1.0 / X3
                        B[t, k] -= b3
                        B[k, t] -= b3
                        B[t, t] += b3
                        B[k, k] += b3
        
        self.B_matrix = B.tocsr()  # Convertir a formato CSR para resolver
        
        print(f"Matriz B construida ({self.num_buses}×{self.num_buses})")
        print(f"Elementos no-cero: {self.B_matrix.nnz}")
    
    def calcular_inyecciones_potencia(self):
        """
        Calcula vector de inyección neta de potencia en cada bus
        P_injection[i] = P_gen[i] - P_load[i]
        """
        print("\nCalculando inyecciones de potencia...")
        
        self.P_injection = np.zeros(self.num_buses)
        
        # Agregar generación
        for idx, gen in self.df_generators.iterrows():
            if gen.get('status', 1) == 0:
                continue
            
            bus_idx = gen['bus_idx']
            if pd.isna(bus_idx):
                continue
            
            bus_idx = int(bus_idx)
            if 0 <= bus_idx < self.num_buses:
                self.P_injection[bus_idx] += gen['P_pu']
        
        # Restar cargas
        for idx, load in self.df_loads.iterrows():
            if load.get('status', 1) == 0:
                continue
            
            bus_idx = load['bus_idx']
            if pd.isna(bus_idx):
                continue
            
            bus_idx = int(bus_idx)
            if 0 <= bus_idx < self.num_buses:
                self.P_injection[bus_idx] -= load['PL_pu']
        
        # Estadísticas
        P_gen_total = self.df_generators[self.df_generators.get('status', 1) == 1]['P_pu'].sum()
        P_load_total = self.df_loads[self.df_loads.get('status', 1) == 1]['PL_pu'].sum()
        
        # print(f"Vector de inyecciones calculado")
        # print(f"Generación total: {P_gen_total:.4f} pu ({P_gen_total * self.base_mva:.2f} MW)")
        # print(f"Carga total:      {P_load_total:.4f} pu ({P_load_total * self.base_mva:.2f} MW)")
        # print(f"Balance:          {P_gen_total - P_load_total:.4f} pu ({(P_gen_total - P_load_total)*self.base_mva:.2f} MW)")
    
    def resolver(self):
        """
        Resuelve el flujo de potencia DC
        Sistema: B * θ = P_injection
        """
        print("\n" + "="*80)
        print(" RESOLVIENDO FLUJO DE POTENCIA DC")
        print("="*80)
        
        # Construir sistema
        self.construir_matriz_susceptancias()
        self.calcular_inyecciones_potencia()
        
        # Identificar bus slack
        slack_buses = self.df_buses[self.df_buses['is_slack'] == True]
        
        if len(slack_buses) == 0:
            print("\n✗ ERROR: No se encontró bus slack (tipo=3)")
            return False
        
        slack_idx = slack_buses['idx'].iloc[0]
        slack_info = slack_buses.iloc[0]
        print(f"\nBus de referencia (Slack): {slack_info['numero']} - {slack_info['nombre']}")
        
        # Preparar sistema reducido (sin bus slack)
        active_indices = [i for i in range(self.num_buses) if i != slack_idx]
        
        B_reduced = self.B_matrix[active_indices, :][:, active_indices]
        P_reduced = self.P_injection[active_indices]
        
        print(f"Sistema reducido: {len(active_indices)}×{len(active_indices)}")
        
        try:
            # Resolver sistema lineal: B_reduced * θ_reduced = P_reduced
            theta_reduced = spsolve(B_reduced, P_reduced)
            
            # Reconstruir vector completo con θ_slack = 0
            self.theta = np.zeros(self.num_buses)
            self.theta[active_indices] = theta_reduced
            self.theta[slack_idx] = 0.0  # Referencia angular
            
            # Convertir a grados
            self.theta_deg = np.degrees(self.theta)
            
            self.converged = True
            print(f"\nSistema resuelto exitosamente")
            
            # Calcular flujos en ramas
            self._calcular_flujos_ramas()
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error al resolver sistema: {e}")
            import traceback
            traceback.print_exc()
            self.converged = False
            return False
    
    def _calcular_flujos_ramas(self):
        """
        Calcula flujos de potencia en todas las ramas (líneas y transformadores)
        P_ij = (θ_i - θ_j) / X_ij
        """
        print("\nCalculando flujos en ramas...")
        
        branch_flows = []
        
        # Flujos en líneas
        for idx, line in self.df_lines.iterrows():
            if line['status'] == 0:
                continue
            
            i = line['from_idx']
            j = line['to_idx']
            
            if pd.isna(i) or pd.isna(j):
                continue
            
            i, j = int(i), int(j)
            X = line['X_pu']
            
            if abs(X) < 1e-10:
                continue
            
            # Flujo de potencia
            P_ij = (self.theta[i] - self.theta[j]) / X
            P_ji = -P_ij
            
            # Pérdidas (no hay en DC, pero se calcula para compatibilidad)
            P_loss = 0.0
            
            # Cargabilidad (% de rating)
            rate = line['rate_A_MVA']
            if rate > 0:
                loading = abs(P_ij * self.base_mva) / rate * 100
            else:
                loading = 0.0
            
            bus_from_num = line['from_bus_numero']
            bus_to_num = line['to_bus_numero']
            bus_from_name = self.df_buses[self.df_buses['numero'] == bus_from_num]['nombre'].iloc[0]
            bus_to_name = self.df_buses[self.df_buses['numero'] == bus_to_num]['nombre'].iloc[0]
            
            branch_flows.append({
                'tipo': 'Línea',
                'from_bus_num': bus_from_num,
                'to_bus_num': bus_to_num,
                'from_nombre': bus_from_name,
                'to_nombre': bus_to_name,
                'circuito': line.get('_CKT', '1'),
                'P_ij_pu': P_ij,
                'P_ji_pu': P_ji,
                'P_ij_MW': P_ij * self.base_mva,
                'P_ji_MW': P_ji * self.base_mva,
                'P_loss_MW': P_loss,
                'loading_pct': loading,
                'rate_MVA': rate
            })
        
        # Flujos en transformadores
        for idx, trafo in self.df_transformers.iterrows():
            if trafo.get('STAT', 1) == 0:
                continue
            
            i = trafo['from_idx']
            j = trafo['to_idx']
            tertiary = trafo.get('tertiary_bus_index', 0)
            
            if pd.isna(i) or pd.isna(j):
                continue
            
            i, j = int(i), int(j)
            
            # Verificar si es de 2 o 3 devanados
            es_tres_devanados = tertiary > 0
            
            if not es_tres_devanados:
                # TRANSFORMADOR DE 2 DEVANADOS
                # X1_2_pu es la impedancia total del transformador
                X = trafo['X1_2_pu']
                
                if abs(X) < 1e-10:
                    continue
                
                P_ij = (self.theta[i] - self.theta[j]) / X
                P_ji = -P_ij
                P_loss = 0.0
                
                # Rating para transformadores de 2 devanados
                rate = trafo.get('SBASE1_2_MVA', 100.0)
                if rate > 0:
                    loading = abs(P_ij * self.base_mva) / rate * 100
                else:
                    loading = 0.0
                
                bus_from_num = trafo['from_bus_index']
                bus_to_num = trafo['to_bus_index']
                bus_from_name = self.df_buses[self.df_buses['numero'] == bus_from_num]['nombre'].iloc[0] if not self.df_buses[self.df_buses['numero'] == bus_from_num].empty else f"Bus_{bus_from_num}"
                bus_to_name = self.df_buses[self.df_buses['numero'] == bus_to_num]['nombre'].iloc[0] if not self.df_buses[self.df_buses['numero'] == bus_to_num].empty else f"Bus_{bus_to_num}"
                
                branch_flows.append({
                    'tipo': 'Trafo 2D',
                    'from_bus_num': bus_from_num,
                    'to_bus_num': bus_to_num,
                    'from_nombre': bus_from_name,
                    'to_nombre': bus_to_name,
                    'circuito': trafo.get('ckt_id', '1'),
                    'P_ij_pu': P_ij,
                    'P_ji_pu': P_ji,
                    'P_ij_MW': P_ij * self.base_mva,
                    'P_ji_MW': P_ji * self.base_mva,
                    'P_loss_MW': P_loss,
                    'loading_pct': loading,
                    'rate_MVA': rate
                })
            
            else:
                # TRANSFORMADOR DE 3 DEVANADOS
                # Calcular flujos en cada devanado usando modelo estrella
                X12 = trafo['X1_2_pu']
                X23 = trafo['X2_3_pu']
                X31 = trafo['X3_1_pu']
                
                X1 = (X12 + X31 - X23) / 2
                X2 = (X12 + X23 - X31) / 2
                X3 = (X23 + X31 - X12) / 2
                
                # Nota: Para trafo 3D se requiere modelado más complejo
                # Por simplicidad, solo reportamos flujo entre primario y secundario
                if abs(X1) > 1e-10 and abs(X2) > 1e-10:
                    # Aproximación: flujo primario-secundario
                    X_equiv = X1 + X2
                    P_ij = (self.theta[i] - self.theta[j]) / X_equiv
                    
                    rate = trafo.get('SBASE1_2_MVA', 100.0)
                    loading = abs(P_ij * self.base_mva) / rate * 100 if rate > 0 else 0.0
                    
                    bus_from_num = trafo['from_bus_index']
                    bus_to_num = trafo['to_bus_index']
                    bus_from_name = self.df_buses[self.df_buses['numero'] == bus_from_num]['nombre'].iloc[0] if not self.df_buses[self.df_buses['numero'] == bus_from_num].empty else f"Bus_{bus_from_num}"
                    bus_to_name = self.df_buses[self.df_buses['numero'] == bus_to_num]['nombre'].iloc[0] if not self.df_buses[self.df_buses['numero'] == bus_to_num].empty else f"Bus_{bus_to_num}"
                    
                    branch_flows.append({
                        'tipo': 'Trafo 3D',
                        'from_bus_num': bus_from_num,
                        'to_bus_num': bus_to_num,
                        'from_nombre': bus_from_name,
                        'to_nombre': bus_to_name,
                        'circuito': trafo.get('ckt_id', '1'),
                        'P_ij_pu': P_ij,
                        'P_ji_pu': -P_ij,
                        'P_ij_MW': P_ij * self.base_mva,
                        'P_ji_MW': -P_ij * self.base_mva,
                        'P_loss_MW': 0.0,
                        'loading_pct': loading,
                        'rate_MVA': rate
                    })
        
        self.P_flows = pd.DataFrame(branch_flows)
        # print(f"Flujos calculados para {len(self.P_flows)} elementos")
    
    def obtener_resultados_buses(self):
        """Retorna DataFrame con resultados por bus"""
        if not self.converged:
            print("Error: Debe resolver el flujo de potencia primero")
            return None
        
        results = self.df_buses.copy()
        results['theta_rad'] = self.theta
        results['theta_deg'] = self.theta_deg
        results['V_pu'] = 1.0  # En flujo DC asumimos V = 1
        results['P_injection_pu'] = self.P_injection
        results['P_injection_MW'] = self.P_injection * self.base_mva
        
        return results
    
    def obtener_resultados_flujos(self):
        """Retorna DataFrame con flujos en ramas"""
        if not self.converged:
            print("Error: Debe resolver el flujo de potencia primero")
            return None
        
        return self.P_flows
    
    def mostrar_resultados(self):
        """Muestra resultados del flujo de potencia"""
        if not self.converged:
            print("Error: El sistema no convergió")
            return
        
        print("\n" + "="*80)
        print("RESULTADOS DEL FLUJO DE POTENCIA DC")
        print("="*80)
        
        # Resultados por bus
        print("\nRESULTADOS POR BUS:")
        print("-" * 80)
        
        bus_results = self.obtener_resultados_buses()
        
        # Mostrar buses con información relevante
        display_cols = ['numero', 'nombre', 'tipo', 'theta_deg', 'P_injection_MW']
        pd.options.display.float_format = '{:.4f}'.format
        print(bus_results[display_cols].to_string(index=False, max_rows=30))
        
        # Estadísticas de ángulos
        print(f"\nEstadísticas de ángulos:")
        print(f"  Máximo:  {self.theta_deg.max():8.4f}°")
        print(f"  Mínimo:  {self.theta_deg.min():8.4f}°")
        print(f"  Rango:   {self.theta_deg.max() - self.theta_deg.min():8.4f}°")
        
        # Resultados de flujos
        print("\n" + "="*80)
        print("FLUJOS EN RAMAS:")
        print("-" * 80)
        
        # Ordenar por cargabilidad descendente
        sorted_flows = self.P_flows.sort_values('loading_pct', ascending=False)
        
        display_cols = ['tipo', 'from_nombre', 'to_nombre', 'P_ij_MW', 'loading_pct', 'rate_MVA']
        print(sorted_flows[display_cols].head(20).to_string(index=False))
        
        # Elementos más cargados
        print(f"\nTop 5 elementos más cargados:")
        top5 = sorted_flows.head(5)
        for idx, branch in top5.iterrows():
            print(f"  {branch['tipo']:15s} {branch['from_nombre'][:15]:15s} -> {branch['to_nombre'][:15]:15s}: "
                  f"{branch['loading_pct']:6.2f}% ({branch['P_ij_MW']:8.2f} MW)")
        
        # Verificar sobrecargas
        overloads = sorted_flows[sorted_flows['loading_pct'] > 100]
        if len(overloads) > 0:
            print(f"\nADVERTENCIA: {len(overloads)} elementos sobrecargados (>100%)")
            for idx, branch in overloads.head(10).iterrows():
                print(f"  {branch['tipo']:15s} {branch['from_nombre'][:15]:15s} -> {branch['to_nombre'][:15]:15s}: "
                      f"{branch['loading_pct']:6.2f}%")
        else:
            print(f"\nNo hay elementos sobrecargados")
    
    def exportar_resultados(self, archivo_excel='resultados_dc_power_flow.xlsx'):
        """Exporta todos los resultados a Excel"""
        if not self.converged:
            print("Error: Debe resolver el flujo de potencia primero")
            return
        
        print(f"\nExportando resultados a {archivo_excel}...")
        
        with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
            # Resultados por bus
            bus_results = self.obtener_resultados_buses()
            bus_results.to_excel(writer, sheet_name='Buses', index=False)
            
            # Flujos en ramas
            self.P_flows.to_excel(writer, sheet_name='Flujos', index=False)
            
            # Datos originales
            self.df_loads.to_excel(writer, sheet_name='Cargas', index=False)
            self.df_generators.to_excel(writer, sheet_name='Generadores', index=False)
            self.df_lines.to_excel(writer, sheet_name='Lineas', index=False)
            if not self.df_transformers.empty:
                self.df_transformers.to_excel(writer, sheet_name='Transformadores', index=False)
        
        print(f"Resultados exportados exitosamente")


# =============================================================================
# EJEMPLO DE USO COMPLETO
# =============================================================================

if __name__ == "__main__":
    # Importar el parser
    from read_raw import RawParser  # Ajusta el nombre del archivo
    
    # Ruta del archivo .raw
    # archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 14 bus.raw"
    archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 118 Bus v2.raw"

    
    print("="*80)
    print(" ANÁLISIS COMPLETO DE FLUJO DE POTENCIA DC")
    print("="*80)
    
    try:
        # PASO 1: Leer archivo .raw
        print("\nPASO 1: Leyendo archivo .raw...")
        parser = RawParser()
        parser.leer_archivo(archivo_raw)
        data = parser.obtener_dataframes()
        
        # PASO 2: Crear objeto de flujo de potencia
        print("\nPASO 2: Inicializando flujo de potencia DC...")
        pf = DCPowerFlow(data)
        
        # PASO 3: Resolver
        print("\nPASO 3: Resolviendo sistema...")
        success = pf.resolver()
        
        if success:
            # PASO 4: Mostrar resultados
            pf.mostrar_resultados()
            
            # PASO 5: Exportar
            # pf.exportar_resultados('IEEE14_DC_resultados.xlsx')
        else:
            print("\nEl flujo de potencia no convergió")
    
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo '{archivo_raw}'")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()