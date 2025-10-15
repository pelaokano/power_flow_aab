"""
DC OPTIMAL POWER FLOW usando Método Simplex
Hereda de DCPowerFlow y reutiliza sus métodos
Usa scipy.optimize.linprog (método simplex)
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack


class DCOptimalPowerFlow:
    """
    DC Optimal Power Flow - Minimiza costos de generación
    Reutiliza métodos de DCPowerFlow para construcción de matrices y reportes
    Usa simplex de scipy para optimización
    """
    
    def __init__(self, raw_data, costos_generacion):
        """
        Inicializa DC-OPF
        
        Args:
            raw_data: Diccionario con DataFrames del RawParser
            costos_generacion: Dict {bus_numero: costo_$/MWh} o lista de costos
        """
        # Importar DCPowerFlow para reutilizar métodos
        from power_flow_dc import DCPowerFlow
        
        # Crear instancia temporal para reutilizar métodos de construcción
        self._pf_helper = DCPowerFlow(raw_data)
        
        # Copiar atributos necesarios
        self.base_mva = self._pf_helper.base_mva
        self.frequency = self._pf_helper.frequency
        self.df_buses = self._pf_helper.df_buses.copy()
        self.df_lines = self._pf_helper.df_lines.copy()
        self.df_transformers = self._pf_helper.df_transformers.copy()
        self.df_loads = self._pf_helper.df_loads.copy()
        self.df_generators = self._pf_helper.df_generators.copy()
        self.num_buses = self._pf_helper.num_buses
        self.bus_num_to_idx = self._pf_helper.bus_num_to_idx
        
        # Procesar costos de generación
        self.costos_gen = self._procesar_costos(costos_generacion)
        
        # Matrices del sistema
        self.B_matrix = None
        self.P_injection = None
        
        # Resultados de optimización
        self.theta = None
        self.theta_deg = None
        self.P_flows = None
        self.P_gen_optimo = None
        self.costo_total = None
        self.precios_nodales = None
        self.converged = False
        
        print(f"\n{'='*80}")
        print(f"  DC OPTIMAL POWER FLOW INICIALIZADO")
        print(f"{'='*80}")
        print(f"  Sistema: {self.num_buses} buses, {len(self.df_generators)} generadores")
        print(f"  Método: Simplex (scipy.optimize.linprog)")
        print(f"{'='*80}\n")
    
    def _procesar_costos(self, costos):
        """Procesa costos de generación a formato uniforme"""
        costos_dict = {}
        
        if isinstance(costos, dict):
            costos_dict = costos
        elif isinstance(costos, list):
            for i, (idx, gen) in enumerate(self.df_generators.iterrows()):
                if i < len(costos):
                    costos_dict[gen['bus_numero']] = costos[i]
        
        # Verificar que todos los generadores tengan costo
        for idx, gen in self.df_generators.iterrows():
            bus_num = gen['bus_numero']
            if bus_num not in costos_dict:
                print(f"Generador en bus {bus_num} sin costo, usando 30 $/MWh por defecto")
                costos_dict[bus_num] = 30.0
        
        return costos_dict
    
    def resolver_opf(self):
        """
        Resuelve el DC Optimal Power Flow usando método simplex
        
        Problema:
            Minimizar: sum(costo_i * P_gen_i)
            Sujeto a:
                - Balance de potencia: B @ theta = G @ P_gen - P_load
                - Límites de generación: P_min <= P_gen <= P_max
                - Límites de flujo: |P_ij| <= P_max_ij
                - Referencia angular: theta_slack = 0
        """
        print("\n" + "="*80)
        print(" RESOLVIENDO DC OPTIMAL POWER FLOW CON SIMPLEX")
        print("="*80)
        
        # REUTILIZAR: Construcción de matriz B
        print("\nPaso 1: Construyendo matriz de susceptancias...")
        self._pf_helper.construir_matriz_susceptancias()
        self.B_matrix = self._pf_helper.B_matrix
        
        # Paso 2: Preparar datos
        print("Paso 2: Preparando datos del problema...")
        n_gen = len(self.df_generators)
        n_bus = self.num_buses
        
        # Cargas por bus (fijas)
        P_load = np.zeros(n_bus)
        for idx, load in self.df_loads.iterrows():
            if load.get('status', 1) == 0:
                continue
            bus_idx = int(load['bus_idx'])
            P_load[bus_idx] += load['PL_pu']
        
        # Matriz que mapea generadores a buses (G_matrix)
        G_matrix = lil_matrix((n_bus, n_gen))
        print(self.df_generators)
        for i, (idx, gen) in enumerate(self.df_generators.iterrows()):
            # print(f'{i}: {idx} and {gen}')
            bus_idx = int(gen['bus_idx'])
            G_matrix[bus_idx, i] = 1.0
        G_matrix = G_matrix.tocsr()
        
        # Identificar bus slack
        slack_idx = self.df_buses[self.df_buses['is_slack']]['idx'].iloc[0]
        slack_info = self.df_buses[self.df_buses['is_slack']].iloc[0]
        print(f"Bus Slack: {slack_info['numero']} - {slack_info['nombre']}")
        
        # Paso 3: Formular problema para linprog
        print("\nPaso 3: Formulando problema de optimización...")
        
        # Variables de decisión: x = [P_gen_1, ..., P_gen_n, theta_1, ..., theta_m]
        n_vars = n_gen + n_bus
        
        # ====== FUNCIÓN OBJETIVO ======
        # Minimizar: sum(costo_i * P_gen_i)
        c = np.zeros(n_vars)
        for i, (idx, gen) in enumerate(self.df_generators.iterrows()):
            bus_num = gen['bus_numero']
            c[i] = self.costos_gen[bus_num]
        # Costos para theta son 0
        print(f"Variables de decisión: {n_gen} P_gen + {n_bus} theta = {n_vars}")
        
        # ====== RESTRICCIONES DE IGUALDAD ======
        # Balance de potencia: B @ theta - G @ P_gen = -P_load
        # En forma: [-G | B] @ [P_gen; theta] = -P_load
        
        A_eq_list = []
        b_eq_list = []
        
        # Balance de potencia en cada bus
        A_balance = hstack([-G_matrix, self.B_matrix]).tocsr()
        b_balance = -P_load
        
        A_eq_list.append(A_balance)
        b_eq_list.append(b_balance)
        
        # Referencia angular: theta[slack] = 0
        A_slack = lil_matrix((1, n_vars))
        A_slack[0, n_gen + slack_idx] = 1.0
        A_eq_list.append(A_slack.tocsr())
        b_eq_list.append(np.array([0.0]))
        
        # Combinar restricciones de igualdad
        A_eq = vstack(A_eq_list).tocsr()
        b_eq = np.concatenate(b_eq_list)
        
        print(f"  Restricciones de igualdad: {A_eq.shape[0]}")
        
        # ====== RESTRICCIONES DE DESIGUALDAD ======
        A_ub_list = []
        b_ub_list = []
        
        # Límites de flujo en líneas: -P_max <= P_ij <= P_max
        # P_ij = (theta_i - theta_j) / X_ij
        n_flow_constraints = 0
        
        for idx, line in self.df_lines.iterrows():
            if line['status'] == 0:
                continue
            
            i = int(line['from_idx'])
            j = int(line['to_idx'])
            X = line['X_pu']
            rate = line['rate_A_MVA']
            
            if abs(X) < 1e-10 or rate == 0:
                continue
            
            # P_ij = (theta_i - theta_j) / X
            # Restricción 1: P_ij <= P_max → (theta_i - theta_j)/X <= P_max
            A_line_1 = lil_matrix((1, n_vars))
            A_line_1[0, n_gen + i] = 1.0 / X
            A_line_1[0, n_gen + j] = -1.0 / X
            A_ub_list.append(A_line_1.tocsr())
            b_ub_list.append(np.array([rate / self.base_mva]))
            
            # Restricción 2: -P_ij <= P_max → -(theta_i - theta_j)/X <= P_max
            A_line_2 = lil_matrix((1, n_vars))
            A_line_2[0, n_gen + i] = -1.0 / X
            A_line_2[0, n_gen + j] = 1.0 / X
            A_ub_list.append(A_line_2.tocsr())
            b_ub_list.append(np.array([rate / self.base_mva]))
            
            n_flow_constraints += 2
        
        # Límites de flujo en transformadores
        for idx, trafo in self.df_transformers.iterrows():
            if trafo.get('STAT', 1) == 0:
                continue
            
            i = int(trafo['from_idx'])
            j = int(trafo['to_idx'])
            X = trafo['X1_2_pu']
            rate = trafo.get('SBASE1_2_MVA', 100.0)
            
            if abs(X) < 1e-10 or rate == 0:
                continue
            
            # P_ij <= P_max
            A_trafo_1 = lil_matrix((1, n_vars))
            A_trafo_1[0, n_gen + i] = 1.0 / X
            A_trafo_1[0, n_gen + j] = -1.0 / X
            A_ub_list.append(A_trafo_1.tocsr())
            b_ub_list.append(np.array([rate / self.base_mva]))
            
            # -P_ij <= P_max
            A_trafo_2 = lil_matrix((1, n_vars))
            A_trafo_2[0, n_gen + i] = -1.0 / X
            A_trafo_2[0, n_gen + j] = 1.0 / X
            A_ub_list.append(A_trafo_2.tocsr())
            b_ub_list.append(np.array([rate / self.base_mva]))
            
            n_flow_constraints += 2
        
        # Combinar restricciones de desigualdad
        if A_ub_list:
            A_ub = vstack(A_ub_list).tocsr()
            b_ub = np.concatenate(b_ub_list)
        else:
            A_ub = None
            b_ub = None
        
        print(f"Restricciones de flujo: {n_flow_constraints}")
        
        # ====== LÍMITES DE VARIABLES ======
        bounds = []
        
        # Límites de P_gen
        for idx, gen in self.df_generators.iterrows():
            P_min = gen['P_min_MW'] / self.base_mva
            P_max = gen['P_max_MW'] / self.base_mva
            bounds.append((P_min, P_max))
        
        # Límites de theta (sin límites explícitos)
        for i in range(n_bus):
            bounds.append((None, None))
        
        # Paso 4: Resolver con linprog (método simplex)
        print("\nPaso 4: Resolviendo con método simplex...")
        print("  (Esto puede tardar unos segundos...)")
        
        try:
            resultado = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',  # Usa HiGHS (más rápido y robusto que simplex puro)
                options={'disp': False, 'presolve': True}
            )
            
            if resultado.success:
                print(f"\nOptimización exitosa!")
                print(f"Status: {resultado.message}")
                print(f"Iteraciones: {resultado.nit if hasattr(resultado, 'nit') else 'N/A'}")

                # Extraer resultados
                x_opt = resultado.x
            
                # Generación óptima
                self.P_gen_optimo = x_opt[:n_gen]

                print(f'generacion optima: {self.P_gen_optimo}')
                
                # Ángulos óptimos
                self.theta = x_opt[n_gen:]
                self.theta_deg = np.degrees(self.theta)

                print(self.theta_deg)
                
                # Costo total
                self.costo_total = resultado.fun * self.base_mva
                
                print(self.costo_total)

                # Precios nodales (valores duales de restricciones de balance)
                # if hasattr(resultado, 'ineqlin') and resultado.ineqlin is not None:
                #     # Precios = duales de balance de potencia
                #     duales_balance = resultado.ineqlin.marginals[:n_bus]
                #     self.precios_nodales = {}
                #     for bus_idx in range(n_bus):
                #         bus_num = self.df_buses.iloc[bus_idx]['numero']
                #         self.precios_nodales[bus_num] = -duales_balance[bus_idx] * self.base_mva
                
                # Actualizar DataFrames con resultados
                for i, (idx, gen) in enumerate(self.df_generators.iterrows()):
                    self.df_generators.at[idx, 'P_pu'] = self.P_gen_optimo[i]
                    self.df_generators.at[idx, 'P_MW'] = self.P_gen_optimo[i] * self.base_mva
                
                # Calcular inyecciones resultantes
                self.P_injection = G_matrix @ self.P_gen_optimo - P_load
                
                #REUTILIZAR: Cálculo de flujos en ramas
                self._pf_helper.theta = self.theta
                self._pf_helper.theta_deg = self.theta_deg
                self._pf_helper.P_injection = self.P_injection
                self._pf_helper.converged = True
                self._pf_helper._calcular_flujos_ramas()
                self.P_flows = self._pf_helper.P_flows
                
                self.converged = True
                
                print(f"\nCosto Total de Operación: ${self.costo_total:,.2f}")
                
                return True
            
            else:
                print(f"\nOptimización falló!")
                print(f"  Status: {resultado.status}")
                print(f"  Mensaje: {resultado.message}")
                self.converged = False
                return False
        
        except Exception as e:
            print(f"\nError durante optimización: {e}")
            import traceback
            traceback.print_exc()
            self.converged = False
            return False
    
    def obtener_resultados_buses(self):
        """REUTILIZAR con extensión para OPF"""
        if not self.converged:
            print("Error: Debe resolver el OPF primero")
            return None
        
        results = self.df_buses.copy()
        results['theta_rad'] = self.theta
        results['theta_deg'] = self.theta_deg
        results['V_pu'] = 1.0
        results['P_injection_pu'] = self.P_injection
        results['P_injection_MW'] = self.P_injection * self.base_mva
        
        # Agregar precios nodales si están disponibles
        if self.precios_nodales:
            results['LMP_$/MWh'] = results['numero'].map(self.precios_nodales)
        
        return results
    
    def obtener_resultados_flujos(self):
        """REUTILIZAR directamente"""
        if not self.converged:
            print("Error: Debe resolver el OPF primero")
            return None
        return self.P_flows
    
    def obtener_despacho_optimo(self):
        """Obtiene tabla con despacho óptimo de generadores"""
        if not self.converged:
            print("Error: Debe resolver el OPF primero")
            return None
        
        despacho = []
        for i, (idx, gen) in enumerate(self.df_generators.iterrows()):
            bus_num = gen['bus_numero']
            bus_name = self.df_buses[self.df_buses['numero'] == bus_num]['nombre'].iloc[0]
            costo = self.costos_gen[bus_num]
            
            despacho.append({
                'bus_numero': bus_num,
                'bus_nombre': bus_name,
                'P_optimo_MW': self.P_gen_optimo[i] * self.base_mva,
                'P_min_MW': gen['P_min_MW'],
                'P_max_MW': gen['P_max_MW'],
                'costo_$/MWh': costo,
                'costo_total_$': self.P_gen_optimo[i] * self.base_mva * costo
            })
        
        return pd.DataFrame(despacho)
    
    def mostrar_resultados(self):
        """REUTILIZAR con extensión para mostrar info de OPF"""
        if not self.converged:
            print("Error: El sistema no convergió")
            return
        
        print("\n" + "="*80)
        print(" RESULTADOS DEL DC OPTIMAL POWER FLOW")
        print("="*80)
        
        # ====== RESULTADOS DE OPTIMIZACIÓN ======
        print("\nCOSTO DE OPERACIÓN:")
        print("-" * 80)
        print(f"  Costo Total: ${self.costo_total:,.2f}")
        
        # Despacho óptimo
        print("\nDESPACHO ÓPTIMO DE GENERADORES:")
        print("-" * 80)
        despacho = self.obtener_despacho_optimo()
        pd.options.display.float_format = '{:.2f}'.format
        print(despacho[['bus_numero', 'bus_nombre', 'P_optimo_MW', 'costo_$/MWh', 'costo_total_$']].to_string(index=False))
        
        # Estadísticas de generación
        total_gen = despacho['P_optimo_MW'].sum()
        total_load = self.df_loads['PL_MW'].sum()
        costo_promedio = self.costo_total / total_gen if total_gen > 0 else 0
        
        print(f"\n  Generación total: {total_gen:.2f} MW")
        print(f"  Carga total:      {total_load:.2f} MW")
        print(f"  Balance:          {total_gen - total_load:.2f} MW")
        print(f"  Costo promedio:   ${costo_promedio:.2f}/MWh")
        
        # ====== RESULTADOS POR BUS ======
        print("\n" + "="*80)
        print("RESULTADOS POR BUS:")
        print("-" * 80)
        
        bus_results = self.obtener_resultados_buses()
        display_cols = ['numero', 'nombre', 'tipo', 'theta_deg', 'P_injection_MW']
        if 'LMP_$/MWh' in bus_results.columns:
            display_cols.append('LMP_$/MWh')
        
        pd.options.display.float_format = '{:.4f}'.format
        print(bus_results[display_cols].head(20).to_string(index=False))
        
        # Estadísticas de ángulos
        print(f"\nEstadísticas de ángulos:")
        print(f"Máximo:  {self.theta_deg.max():8.4f}°")
        print(f"Mínimo:  {self.theta_deg.min():8.4f}°")
        print(f"Rango:   {self.theta_deg.max() - self.theta_deg.min():8.4f}°")
        
        # ====== FLUJOS EN RAMAS ======
        print("\n" + "="*80)
        print("FLUJOS EN RAMAS:")
        print("-" * 80)
        
        sorted_flows = self.P_flows.sort_values('loading_pct', ascending=False)
        display_cols = ['tipo', 'from_nombre', 'to_nombre', 'P_ij_MW', 'loading_pct', 'rate_MVA']
        pd.options.display.float_format = '{:.2f}'.format
        print(sorted_flows[display_cols].head(15).to_string(index=False))
        
        # Top 5 más cargados
        print(f"\nTop 5 elementos más cargados:")
        top5 = sorted_flows.head(5)
        for idx, branch in top5.iterrows():
            print(f"  {branch['tipo']:15s} {branch['from_nombre'][:15]:15s} -> {branch['to_nombre'][:15]:15s}: "
                  f"{branch['loading_pct']:6.2f}% ({branch['P_ij_MW']:8.2f} MW)")
        
        # Verificar sobrecargas
        overloads = sorted_flows[sorted_flows['loading_pct'] > 100]
        if len(overloads) > 0:
            print(f"\nADVERTENCIA: {len(overloads)} elementos sobrecargados (>100%)")
            for idx, branch in overloads.head(5).iterrows():
                print(f"  {branch['tipo']:15s} {branch['from_nombre'][:15]:15s} -> {branch['to_nombre'][:15]:15s}: "
                      f"{branch['loading_pct']:6.2f}%")
        else:
            print(f"\nNo hay elementos sobrecargados")
        
        # Precios nodales
        if self.precios_nodales:
            print("\n" + "="*80)
            print("PRECIOS MARGINALES LOCALES (LMP):")
            print("-" * 80)
            precios_sorted = sorted(self.precios_nodales.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Top 10 buses con mayor LMP:")
            for i, (bus_num, precio) in enumerate(precios_sorted[:10], 1):
                bus_name = self.df_buses[self.df_buses['numero']==bus_num]['nombre'].iloc[0]
                print(f"  {i:2d}. Bus {bus_num:3d} {bus_name[:20]:20s}: ${precio:7.2f}/MWh")
    
    def exportar_resultados(self, archivo_excel='resultados_dc_opf.xlsx'):
        """REUTILIZAR con extensión para incluir datos de OPF"""
        if not self.converged:
            print("Error: Debe resolver el OPF primero")
            return
        
        print(f"\nExportando resultados a {archivo_excel}...")
        
        with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
            # Despacho óptimo
            despacho = self.obtener_despacho_optimo()
            despacho.to_excel(writer, sheet_name='Despacho_Optimo', index=False)
            
            # Resultados por bus
            bus_results = self.obtener_resultados_buses()
            bus_results.to_excel(writer, sheet_name='Buses', index=False)
            
            # Flujos en ramas
            self.P_flows.to_excel(writer, sheet_name='Flujos', index=False)
            
            # Precios nodales
            if self.precios_nodales:
                precios_df = pd.DataFrame([
                    {'bus_numero': k, 'LMP_$/MWh': v}
                    for k, v in self.precios_nodales.items()
                ])
                precios_df.to_excel(writer, sheet_name='Precios_Nodales', index=False)
            
            # Datos originales
            self.df_loads.to_excel(writer, sheet_name='Cargas', index=False)
            self.df_generators.to_excel(writer, sheet_name='Generadores', index=False)
            self.df_lines.to_excel(writer, sheet_name='Lineas', index=False)
        
        print(f"Resultados exportados exitosamente")


# =============================================================================
# EJEMPLO DE USO COMPLETO
# =============================================================================

if __name__ == "__main__":
    from read_raw import RawParser
    
    # Ruta del archivo .raw
    # archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 14 bus.raw"
    archivo_raw = r"C:\Users\AdrianAlarconBecerra\Desktop\power_flow_aap\IEEE 118 Bus v2.raw"
    
    print("="*80)
    print(" ANÁLISIS COMPLETO: DC OPTIMAL POWER FLOW")
    print("="*80)
    
    try:
        # PASO 1: Leer archivo .raw
        print("\nPASO 1: Leyendo archivo .raw...")
        parser = RawParser()
        parser.leer_archivo(archivo_raw)
        data = parser.obtener_dataframes()

        fun_gen = lambda x: 100 if x == 0 else x*1.5
        
        data['generadores']['P_min_MW'] = 0
        data['generadores']['P_max_MW'] = data['generadores']['P_MW'].apply(fun_gen)

        data['lineas']['rate_A_MVA'] = 100
        
        print(data['lineas'][list(data['lineas'].columns)[:10]])

        # PASO 2: Definir costos de generación
        print("\nPASO 2: Definiendo costos de generación...")
        
        # Opción A: Costos por buses
        # costos para sistema de 14 bus
        # costos = {
        #     1: 35.0,   # Bus 1: $20/MWh
        #     2: 25.0,   # Bus 2: $25/MWh
        #     3: 30.0,   # Bus 3: $30/MWh
        #     6: 22.0,   # Bus 6: $22/MWh
        #     8: 28.0,   # Bus 8: $28/MWh
        # }
        # costos para sistema de 118 buses
        costos = [16.229570577414048, 18.037983595242423, 10.814876753593817, 16.732407371989552, 19.459959360400788, 17.055015620661965, 18.78328805092262, 15.685646014298072, 18.855622145721178, 13.078526316759365, 19.68505555986564, 19.64893172757651, 17.114034651233663, 12.329198406610207, 12.303917246136013, 16.50807085684537, 17.68474181185183, 14.046241415722768, 14.049569067564219, 18.743839315546904, 15.290856838621156, 14.737977785301847, 12.626733155406253, 14.248636853460658, 15.526228663765943, 13.617125709620865, 15.454810671907515, 11.492408087293686, 10.40927866639684, 18.824222950615834, 13.28119496271231, 13.676033464006503, 14.288418249957608, 14.395900381143537, 12.009906148418212, 19.391158283871967, 10.366023330928769, 14.16227026504729, 13.466441615715592, 17.890105089254522, 14.827177692945444, 14.689829950997193, 14.901608956170424, 14.367368280985021, 15.784617823242773, 16.0322346189108, 12.872887847471052, 17.357698936348292, 17.081017119356698, 13.687902720071383, 12.524053168947264, 17.369008211532424, 14.264161418621933]
        
        # Opción B: Lista de costos (en orden de generadores)
        # costos = [20.0, 25.0, 30.0, 22.0, 28.0]
        
        # PASO 3: Crear objeto DC-OPF
        print("\nPASO 3: Inicializando DC-OPF...")
        opf = DCOptimalPowerFlow(data, costos)
        
        # PASO 4: Resolver optimización
        print("\nPASO 4: Resolviendo optimización...")
        success = opf.resolver_opf()
        
        if success:
            # PASO 5: Mostrar resultados
            opf.mostrar_resultados()
            
            # PASO 6: Exportar resultados
            # opf.exportar_resultados('IEEE14_DC_OPF_resultados.xlsx')
            
            print("\n" + "="*80)
            print(" ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("="*80)
        else:
            print("\nLa optimización no convergió")
    
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo '{archivo_raw}'")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()