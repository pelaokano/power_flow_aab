"""
EJEMPLO COMPLETO CON SELECCIÓN DE MÉTODO
Leer archivo .RAW y resolver flujo de potencia AC
Permite elegir entre Newton-Raphson y Gauss-Seidel
"""

import numpy as np
import matplotlib.pyplot as plt
from read_raw import RawParser
from power_flow_nr import PowerFlowNR
from power_flow_gs import PowerFlowGS


def ejecucion_completa(archivo_raw, metodo='NR', factor_aceleracion=1.6):
    """
    Ejecuta el flujo completo: leer .raw y resolver flujo de potencia
    
    Args:
        archivo_raw: ruta del archivo .raw
        metodo: 'NR' para Newton-Raphson, 'GS' para Gauss-Seidel, 'AMBOS' para comparar
        factor_aceleracion: factor de aceleración para Gauss-Seidel (1.0-1.8)
    
    Returns:
        tuple: (solver, df_buses, df_flujos) o dict si metodo='AMBOS'
    """
    
    print("╔" + "═"*78 + "╗")
    print("║" + " ANÁLISIS DE FLUJO DE POTENCIA AC ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\nArchivo: {archivo_raw}")
    
    # Mostrar método seleccionado
    metodo_str = {
        'NR': 'Newton-Raphson',
        'GS': f'Gauss-Seidel (α={factor_aceleracion})',
        'AMBOS': 'Comparación de ambos métodos'
    }
    print(f"Método: {metodo_str.get(metodo, metodo)}\n")
    
    # ============================================
    # PASO 1: LEER ARCHIVO .RAW
    # ============================================
    print("\n[1/3] LECTURA DE ARCHIVO .RAW")
    print("-" * 80)
    
    try:
        parser = RawParser()
        parser.leer_archivo(archivo_raw)
        datos = parser.obtener_dataframes()
        
        print(f"\nArchivo leído exitosamente")
        print(f"  - Buses: {len(datos['buses'])}")
        print(f"  - Líneas: {len(datos['lineas']) if datos['lineas'] is not None else 0}")
        print(f"  - Generadores: {len(datos['generadores']) if datos['generadores'] is not None else 0}")
        print(f"  - Cargas: {len(datos['cargas']) if datos['cargas'] is not None else 0}")
        print(f"  - Transformadores: {len(datos['transformadores']) if datos['transformadores'] is not None else 0}")
        
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo '{archivo_raw}'")
        return None
    except Exception as e:
        print(f"\nERROR al leer archivo: {e}")
        return None
    
    # ============================================
    # PASO 2: RESOLVER SEGÚN MÉTODO SELECCIONADO
    # ============================================
    if metodo == 'AMBOS':
        return resolver_y_comparar(datos, factor_aceleracion)
    elif metodo == 'NR':
        return resolver_newton_raphson(datos)
    elif metodo == 'GS':
        return resolver_gauss_seidel(datos, factor_aceleracion)
    else:
        print(f"\nERROR: Método '{metodo}' no reconocido. Use 'NR', 'GS' o 'AMBOS'")
        return None


def resolver_newton_raphson(datos):
    """
    Resuelve usando Newton-Raphson
    """
    print("\n\n[2/3] RESOLVER FLUJO DE POTENCIA - NEWTON-RAPHSON")
    print("-" * 80)
    
    try:
        import time
        solver = PowerFlowNR(datos)
        
        tiempo_inicio = time.time()
        convergencia = solver.resolver(max_iter=20, tolerancia=1e-6)
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        if not convergencia:
            print("\nEl flujo de potencia no convergió")
            return None
        
        print(f"\nConvergencia exitosa")
        print(f"  Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")
            
    except Exception as e:
        print(f"\nERROR al resolver: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generar reportes
    return generar_reportes(solver, datos, 'NR')


def resolver_gauss_seidel(datos, factor_aceleracion):
    """
    Resuelve usando Gauss-Seidel
    """
    print("\n\n[2/3] RESOLVER FLUJO DE POTENCIA - GAUSS-SEIDEL")
    print("-" * 80)
    
    try:
        import time
        solver = PowerFlowGS(datos)
        
        tiempo_inicio = time.time()
        convergencia = solver.resolver(
            max_iter=100, 
            tolerancia=1e-6, 
            factor_aceleracion=factor_aceleracion
        )
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        if not convergencia:
            print("\nEl flujo de potencia no convergió")
            print("💡 Sugerencias:")
            print("   - Aumentar el número máximo de iteraciones")
            print("   - Ajustar el factor de aceleración (probar entre 1.4 y 1.7)")
            return None
        
        print(f"\nConvergencia exitosa")
        print(f"  Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")
        print(f"  Iteraciones: {solver.iteraciones_realizadas}")
            
    except Exception as e:
        print(f"\nERROR al resolver: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generar reportes
    return generar_reportes(solver, datos, 'GS', factor_aceleracion)


def resolver_y_comparar(datos, factor_aceleracion):
    """
    Resuelve con ambos métodos y compara resultados
    """
    import time
    
    print("\n\n[2/3] RESOLVER CON AMBOS MÉTODOS")
    print("-" * 80)
    
    resultados = {
        'NR': {'solver': None, 'tiempo': 0, 'convergencia': False},
        'GS': {'solver': None, 'tiempo': 0, 'convergencia': False}
    }
    
    # Newton-Raphson
    print("\nMétodo 1: NEWTON-RAPHSON")
    print("-" * 40)
    try:
        solver_nr = PowerFlowNR(datos)
        t_inicio = time.time()
        conv_nr = solver_nr.resolver(max_iter=20, tolerancia=1e-6)
        t_nr = time.time() - t_inicio
        
        resultados['NR']['solver'] = solver_nr
        resultados['NR']['tiempo'] = t_nr
        resultados['NR']['convergencia'] = conv_nr
        
        if conv_nr:
            print(f"Convergió en {t_nr:.4f} segundos")
        else:
            print("No convergió")
    except Exception as e:
        print(f"Error: {e}")
    
    # Gauss-Seidel
    print("\nMétodo 2: GAUSS-SEIDEL")
    print("-" * 40)
    try:
        solver_gs = PowerFlowGS(datos)
        t_inicio = time.time()
        conv_gs = solver_gs.resolver(max_iter=100, tolerancia=1e-6, 
                                     factor_aceleracion=factor_aceleracion)
        t_gs = time.time() - t_inicio
        
        resultados['GS']['solver'] = solver_gs
        resultados['GS']['tiempo'] = t_gs
        resultados['GS']['convergencia'] = conv_gs
        
        if conv_gs:
            print(f"Convergió en {t_gs:.4f} segundos ({solver_gs.iteraciones_realizadas} iteraciones)")
        else:
            print("No convergió")
    except Exception as e:
        print(f"Error: {e}")
    
    # Verificar que ambos convergieron
    if not (resultados['NR']['convergencia'] and resultados['GS']['convergencia']):
        print("\nAl menos uno de los métodos no convergió")
        return None
    
    # Generar reportes comparativos
    return generar_reportes_comparativos(resultados, datos)


def generar_reportes(solver, datos, metodo, factor_aceleracion=None):
    """
    Genera reportes para un solo método
    """
    print("\n\n[3/3] GENERAR REPORTES Y GRÁFICAS")
    print("-" * 80)
    
    try:
        df_buses, df_flujos = solver.generar_reporte()
        
        # Análisis adicional
        print("\n\nANÁLISIS ADICIONAL:")
        print("-" * 80)
        
        # Balance de potencia
        P_gen_total = df_buses['P_gen (MW)'].sum()
        P_carga_total = df_buses['P_carga (MW)'].sum()
        P_perdidas = df_flujos['Perdidas_MW'].sum() if not df_flujos.empty else 0
        
        print(f"\nBalance de Potencia Activa:")
        print(f"  Generación total:  {P_gen_total:>10.2f} MW")
        print(f"  Carga total:       {P_carga_total:>10.2f} MW")
        print(f"  Pérdidas:          {P_perdidas:>10.2f} MW")
        print(f"  Balance:           {P_gen_total - P_carga_total - P_perdidas:>10.4f} MW")
        print(f"  Pérdidas (%):      {(P_perdidas/P_gen_total)*100:>10.2f} %")
        
        # Estadísticas de voltajes
        print(f"\nEstadísticas de Voltajes:")
        print(f"  Máximo:   {df_buses['V (pu)'].max():.4f} pu")
        print(f"  Mínimo:   {df_buses['V (pu)'].min():.4f} pu")
        print(f"  Promedio: {df_buses['V (pu)'].mean():.4f} pu")
        
        # Buses fuera de límites normales (0.95 - 1.05 pu)
        fuera_limites = df_buses[(df_buses['V (pu)'] < 0.95) | (df_buses['V (pu)'] > 1.05)]
        if not fuera_limites.empty:
            print(f"\nBuses fuera de límites normales (0.95-1.05 pu):")
            print(fuera_limites[['Bus', 'V (pu)']].to_string(index=False))
        else:
            print(f"\nTodos los buses dentro de límites normales")
        
        # Crear visualizaciones
        metodo_nombre = 'Newton-Raphson' if metodo == 'NR' else f'Gauss-Seidel (α={factor_aceleracion})'
        crear_graficas(solver, df_buses, df_flujos, metodo_nombre)
        
        return solver, df_buses, df_flujos
        
    except Exception as e:
        print(f"\nERROR al generar reportes: {e}")
        import traceback
        traceback.print_exc()
        return None


def generar_reportes_comparativos(resultados, datos):
    """
    Genera reportes comparando ambos métodos
    """
    print("\n\n[3/3] GENERAR REPORTES COMPARATIVOS")
    print("-" * 80)
    
    try:
        # Obtener DataFrames de ambos métodos
        df_buses_nr, df_flujos_nr = resultados['NR']['solver'].generar_reporte()
        df_buses_gs, df_flujos_gs = resultados['GS']['solver'].generar_reporte()
        
        # Tabla comparativa
        print("\n\nTABLA COMPARATIVA:")
        print("-" * 80)
        print(f"{'Métrica':<35} {'Newton-Raphson':<20} {'Gauss-Seidel':<20}")
        print("-" * 80)
        
        # Tiempos
        t_nr = resultados['NR']['tiempo']
        t_gs = resultados['GS']['tiempo']
        print(f"{'Tiempo de ejecución (s)':<35} {t_nr:<20.6f} {t_gs:<20.6f}")
        
        # Iteraciones
        iter_gs = resultados['GS']['solver'].iteraciones_realizadas
        print(f"{'Iteraciones':<35} {'~4-6 (típico)':<20} {iter_gs:<20}")
        
        # Voltajes
        v_nr_avg = df_buses_nr['V (pu)'].mean()
        v_gs_avg = df_buses_gs['V (pu)'].mean()
        print(f"{'Voltaje promedio (pu)':<35} {v_nr_avg:<20.6f} {v_gs_avg:<20.6f}")
        
        # Pérdidas
        p_nr = df_flujos_nr['Perdidas_MW'].sum() if not df_flujos_nr.empty else 0
        p_gs = df_flujos_gs['Perdidas_MW'].sum() if not df_flujos_gs.empty else 0
        print(f"{'Pérdidas totales (MW)':<35} {p_nr:<20.6f} {p_gs:<20.6f}")
        
        # Diferencias
        print(f"\n{'Diferencia absoluta':<35} {'|NR - GS|':<20}")
        print("-" * 80)
        print(f"{'Voltaje promedio (pu)':<35} {abs(v_nr_avg - v_gs_avg):<20.2e}")
        print(f"{'Pérdidas (MW)':<35} {abs(p_nr - p_gs):<20.2e}")
        
        # Gráficas comparativas
        crear_graficas_comparativas(df_buses_nr, df_buses_gs, t_nr, t_gs, iter_gs)
        
        print("\nAnálisis comparativo completado")
        
        return {
            'NR': {'solver': resultados['NR']['solver'], 'df_buses': df_buses_nr, 'df_flujos': df_flujos_nr},
            'GS': {'solver': resultados['GS']['solver'], 'df_buses': df_buses_gs, 'df_flujos': df_flujos_gs}
        }
        
    except Exception as e:
        print(f"\nERROR al generar reportes: {e}")
        import traceback
        traceback.print_exc()
        return None


def crear_graficas(solver, df_buses, df_flujos, metodo_nombre):
    """
    Crea gráficas de análisis del sistema para un método
    """
    print(f"\nGenerando gráficas ({metodo_nombre})...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Perfil de voltajes
    ax1 = plt.subplot(2, 3, 1)
    buses = df_buses['Bus'].values
    voltajes = df_buses['V (pu)'].values
    colores = ['red' if df_buses.iloc[i]['Tipo'] == 'Slack' else 
               'blue' if df_buses.iloc[i]['Tipo'] == 'PV' else 'green' 
               for i in range(len(buses))]
    
    ax1.bar(range(len(buses)), voltajes, color=colores, alpha=0.7)
    ax1.axhline(y=1.05, color='r', linestyle='--', linewidth=1, label='Límite superior')
    ax1.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='Límite inferior')
    ax1.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('Bus')
    ax1.set_ylabel('Voltaje (pu)')
    ax1.set_title(f'Perfil de Voltajes\n{metodo_nombre}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ángulos de fase
    ax2 = plt.subplot(2, 3, 2)
    angulos = df_buses['Ang (°)'].values
    ax2.bar(range(len(buses)), angulos, color='orange', alpha=0.7)
    ax2.set_xlabel('Bus')
    ax2.set_ylabel('Ángulo (°)')
    ax2.set_title('Ángulos de Fase por Bus')
    ax2.grid(True, alpha=0.3)
    
    # 3. Generación vs Carga
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(2)
    potencias = [df_buses['P_gen (MW)'].sum(), df_buses['P_carga (MW)'].sum()]
    ax3.bar(x_pos, potencias, color=['green', 'red'], alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Generación', 'Carga'])
    ax3.set_ylabel('Potencia (MW)')
    ax3.set_title('Balance de Potencia Activa')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Distribución de tipos de bus
    ax4 = plt.subplot(2, 3, 4)
    tipos = df_buses['Tipo'].value_counts()
    ax4.pie(tipos.values, labels=tipos.index, autopct='%1.1f%%', 
            colors=['red', 'blue', 'green'], startangle=90)
    ax4.set_title('Distribución de Tipos de Bus')
    
    # 5. Flujos en líneas (top 10 más cargadas)
    if not df_flujos.empty:
        ax5 = plt.subplot(2, 3, 5)
        df_flujos_sorted = df_flujos.sort_values('P_from_MW', ascending=False).head(10)
        lineas_nombres = [f"{row['from_bus']}-{row['to_bus']}" 
                         for _, row in df_flujos_sorted.iterrows()]
        
        ax5.barh(range(len(lineas_nombres)), df_flujos_sorted['P_from_MW'], 
                color='purple', alpha=0.7)
        ax5.set_yticks(range(len(lineas_nombres)))
        ax5.set_yticklabels(lineas_nombres, fontsize=8)
        ax5.set_xlabel('Flujo de Potencia (MW)')
        ax5.set_title('Top 10 Líneas Más Cargadas')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Mapa fasorial (primeros 20 buses)
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    n_plot = min(20, len(buses))
    theta = np.deg2rad(df_buses['Ang (°)'].values[:n_plot])
    r = df_buses['V (pu)'].values[:n_plot]
    
    scatter = ax6.scatter(theta, r, c=range(n_plot), cmap='viridis', s=100, alpha=0.7)
    ax6.set_ylim(0, 1.1)
    ax6.set_title(f'Diagrama Fasorial\n(primeros {n_plot} buses)', pad=20)
    plt.colorbar(scatter, ax=ax6, label='Índice de Bus')
    
    plt.suptitle(f'Análisis de Flujo de Potencia - {metodo_nombre}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    nombre_archivo = f'analisis_flujo_potencia_{metodo_nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("α=", "alpha_")}.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Gráficas guardadas en '{nombre_archivo}'")


def crear_graficas_comparativas(df_buses_nr, df_buses_gs, t_nr, t_gs, iter_gs):
    """
    Crea gráficas comparativas entre ambos métodos
    """
    print("\nGenerando gráficas comparativas...")
    
    fig = plt.figure(figsize=(15, 8))
    
    buses = df_buses_nr['Bus'].values
    x = np.arange(len(buses))
    width = 0.35
    
    # 1. Comparación de voltajes
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(x - width/2, df_buses_nr['V (pu)'], width, label='Newton-Raphson', alpha=0.8)
    ax1.bar(x + width/2, df_buses_gs['V (pu)'], width, label='Gauss-Seidel', alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Bus')
    ax1.set_ylabel('Voltaje (pu)')
    ax1.set_title('Comparación de Voltajes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Diferencias en voltajes
    ax2 = plt.subplot(2, 3, 2)
    diff_v = np.abs(df_buses_nr['V (pu)'].values - df_buses_gs['V (pu)'].values)
    ax2.bar(x, diff_v, color='red', alpha=0.7)
    ax2.set_xlabel('Bus')
    ax2.set_ylabel('|ΔV| (pu)')
    ax2.set_title('Diferencias en Magnitud de Voltaje')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparación de ángulos
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x - width/2, df_buses_nr['Ang (°)'], width, label='Newton-Raphson', alpha=0.8)
    ax3.bar(x + width/2, df_buses_gs['Ang (°)'], width, label='Gauss-Seidel', alpha=0.8)
    ax3.set_xlabel('Bus')
    ax3.set_ylabel('Ángulo (°)')
    ax3.set_title('Comparación de Ángulos')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Diferencias en ángulos
    ax4 = plt.subplot(2, 3, 4)
    diff_ang = np.abs(df_buses_nr['Ang (°)'].values - df_buses_gs['Ang (°)'].values)
    ax4.bar(x, diff_ang, color='purple', alpha=0.7)
    ax4.set_xlabel('Bus')
    ax4.set_ylabel('|Δθ| (°)')
    ax4.set_title('Diferencias en Ángulo de Fase')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Comparación de tiempos
    ax5 = plt.subplot(2, 3, 5)
    metodos = ['Newton-\nRaphson', 'Gauss-\nSeidel']
    tiempos = [t_nr, t_gs]
    bars = ax5.bar(metodos, tiempos, color=['blue', 'orange'], alpha=0.7)
    ax5.set_ylabel('Tiempo (segundos)')
    ax5.set_title('Tiempo de Ejecución')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, tiempo in zip(bars, tiempos):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{tiempo:.4f}s', ha='center', va='bottom')
    
    # 6. Estadísticas
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    diff_v_max = diff_v.max()
    diff_v_avg = diff_v.mean()
    diff_ang_max = diff_ang.max()
    diff_ang_avg = diff_ang.mean()
    
    stats_text = f"""
    COMPARACIÓN DE MÉTODOS
    
    Tiempo:
      • Newton-Raphson: {t_nr:.4f} s
      • Gauss-Seidel: {t_gs:.4f} s
      • Ratio: {t_gs/t_nr:.2f}x
    
    Iteraciones:
      • Newton-Raphson: ~4-6
      • Gauss-Seidel: {iter_gs}
    
    Diferencias:
      • delta V máximo: {diff_v_max:.2e} pu
      • delta V promedio: {diff_v_avg:.2e} pu
      • delta θ máximo: {diff_ang_max:.2e} °
      • delta θ promedio: {diff_ang_avg:.2e} °
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comparación: Newton-Raphson vs Gauss-Seidel', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacion_metodos.png', dpi=150, bbox_inches='tight')
    print("Gráficas comparativas guardadas en 'comparacion_metodos.png'")


def exportar_resultados(resultado, archivo_base, metodo):
    """
    Exporta los resultados a archivos CSV
    """
    print("\nExportando resultados...")
    
    prefijo = f"{archivo_base.replace('.raw', '')}_{metodo}"
    
    if metodo == 'AMBOS':
        # Exportar ambos métodos
        resultado['NR']['df_buses'].to_csv(f"{prefijo}_NR_buses.csv", index=False)
        resultado['GS']['df_buses'].to_csv(f"{prefijo}_GS_buses.csv", index=False)
        print(f"Buses NR: {prefijo}_NR_buses.csv")
        print(f"Buses GS: {prefijo}_GS_buses.csv")
        
        if not resultado['NR']['df_flujos'].empty:
            resultado['NR']['df_flujos'].to_csv(f"{prefijo}_NR_flujos.csv", index=False)
            print(f"Flujos NR: {prefijo}_NR_flujos.csv")
        
        if not resultado['GS']['df_flujos'].empty:
            resultado['GS']['df_flujos'].to_csv(f"{prefijo}_GS_flujos.csv", index=False)
            print(f"Flujos GS: {prefijo}_GS_flujos.csv")
    else:
        solver, df_buses, df_flujos = resultado
        df_buses.to_csv(f"{prefijo}_buses.csv", index=False)
        print(f"Buses: {prefijo}_buses.csv")
        
        if not df_flujos.empty:
            df_flujos.to_csv(f"{prefijo}_flujos.csv", index=False)
            print(f"Flujos: {prefijo}_flujos.csv")
    
    print("Exportación completada")


# ============================================
# EJECUTAR ANÁLISIS
# ============================================
if __name__ == "__main__":
    
    # ========================================
    # CONFIGURACIÓN
    # ========================================
    archivo = "IEEE 118 Bus v2.raw"
    # archivo = "IEEE 14 bus.raw"
    
    # Opciones de método:
    # 'NR'    - Solo Newton-Raphson (rápido, ~4-6 iteraciones)
    # 'GS'    - Solo Gauss-Seidel (robusto, ~30-100 iteraciones)
    # 'AMBOS' - Ejecuta y compara ambos métodos
    
    # metodo_a_usar = 'NR'  # Cambia aquí el método
    metodo_a_usar = 'GS'  # Cambia aquí el método
    
    # Factor de aceleración para Gauss-Seidel (1.0 - 1.8)
    # Valores típicos: 1.4-1.6
    alpha = 1.6
    
    # ========================================
    # EJECUCIÓN
    # ========================================
    print("\n" + "="*80)
    print(f"EJECUTANDO ANÁLISIS: {metodo_a_usar}")
    print("="*80)
    
    try:
        resultado = ejecucion_completa(archivo, metodo=metodo_a_usar, factor_aceleracion=alpha)
        
        if resultado is not None:
            print("\n" + "="*80)
            print("ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("="*80)
            
            # Opcional: exportar resultados
            # exportar_resultados(resultado, archivo, metodo_a_usar)
        else:
            print("\nEl análisis no se completó correctamente")
            
    except Exception as e:
        print(f"\nERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # EJEMPLOS DE USO ALTERNATIVO
    # ========================================
    
    # Ejemplo 1: Solo Newton-Raphson
    # resultado = ejecucion_completa("IEEE 14 bus.raw", metodo='NR')
    
    # Ejemplo 2: Solo Gauss-Seidel con factor personalizado
    # resultado = ejecucion_completa("IEEE 118 Bus v2.raw", metodo='GS', factor_aceleracion=1.4)
    
    # Ejemplo 3: Comparar ambos métodos
    # resultado = ejecucion_completa("IEEE 14 bus.raw", metodo='AMBOS')
    
    # Ejemplo 4: Múltiples archivos
    # archivos = ["IEEE 14 bus.raw", "IEEE 118 Bus v2.raw"]
    # for arch in archivos:
    #     resultado = ejecucion_completa(arch, metodo='AMBOS')
    #     if resultado:
    #         exportar_resultados(resultado, arch, 'AMBOS')