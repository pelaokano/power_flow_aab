"""
EJEMPLO COMPLETO
Leer archivo .RAW y resolver flujo de potencia AC
"""

# Importar ambos módulos (asume que están en el mismo directorio)
# from raw_parser import RawParser
# from power_flow_nr import PowerFlowNR

import numpy as np
import matplotlib.pyplot as plt
from read_raw import RawParser
from power_flow_nr import PowerFlowNR


def ejecucion_completa(archivo_raw):
    """
    Ejecuta el flujo completo: leer .raw y resolver flujo de potencia
    """
    
    print("╔" + "═"*78 + "╗")
    print("║" + " ANÁLISIS DE FLUJO DE POTENCIA AC ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\nArchivo: {archivo_raw}\n")
    
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
    # PASO 2: RESOLVER FLUJO DE POTENCIA
    # ============================================
    print("\n\n[2/3] RESOLVER FLUJO DE POTENCIA")
    print("-" * 80)
    
    try:
        solver = PowerFlowNR(datos)
        convergencia = solver.resolver(max_iter=20, tolerancia=1e-6)
        
        if not convergencia:
            print("\El flujo de potencia no convergió")
            return None
            
    except Exception as e:
        print(f"\nERROR al resolver: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================
    # PASO 3: GENERAR REPORTES
    # ============================================
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
        print(f"  Máximo:  {df_buses['V (pu)'].max():.4f} pu")
        print(f"  Mínimo:  {df_buses['V (pu)'].min():.4f} pu")
        print(f"  Promedio: {df_buses['V (pu)'].mean():.4f} pu")
        
        # Buses fuera de límites normales (0.95 - 1.05 pu)
        fuera_limites = df_buses[(df_buses['V (pu)'] < 0.95) | (df_buses['V (pu)'] > 1.05)]
        if not fuera_limites.empty:
            print(f"\nBuses fuera de límites normales (0.95-1.05 pu):")
            print(fuera_limites[['Bus', 'V (pu)']].to_string(index=False))
        else:
            print(f"\nTodos los buses dentro de límites normales (0.95-1.05 pu)")
        
        # Crear visualizaciones
        crear_graficas(solver, df_buses, df_flujos)
        
        return solver, df_buses, df_flujos
        
    except Exception as e:
        print(f"\nERROR al generar reportes: {e}")
        import traceback
        traceback.print_exc()
        return None


def crear_graficas(solver, df_buses, df_flujos):
    """
    Crea gráficas de análisis del sistema
    """
    print("\nGenerando gráficas...")
    
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
    ax1.set_title('Perfil de Voltajes por Bus')
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
    
    plt.tight_layout()
    plt.savefig('analisis_flujo_potencia.png', dpi=150, bbox_inches='tight')
    print("Gráficas guardadas en 'analisis_flujo_potencia.png'")
    
    # No mostrar en modo script, pero disponible si se ejecuta interactivamente
    # plt.show()


def exportar_resultados(solver, df_buses, df_flujos, prefijo="resultados"):
    """
    Exporta los resultados a archivos CSV
    """
    print("\nExportando resultados...")
    
    df_buses.to_csv(f"{prefijo}_buses.csv", index=False)
    print(f"Buses: {prefijo}_buses.csv")
    
    if not df_flujos.empty:
        df_flujos.to_csv(f"{prefijo}_flujos.csv", index=False)
        print(f"Flujos: {prefijo}_flujos.csv")
    
    print("\nExportación completada")


# ============================================
# EJECUTAR ANÁLISIS
# ============================================
if __name__ == "__main__":
    
    # Rutas de archivos a analizar
    archivos = [
        # "IEEE 14 bus.raw",
        "IEEE 118 Bus v2.raw",
        # "tu_sistema.raw",
    ]
    
    for archivo in archivos:
        try:
            resultado = ejecucion_completa(archivo)
            
            if resultado is not None:
                solver, df_buses, df_flujos = resultado
                
                # Opcional: exportar resultados
                # exportar_resultados(solver, df_buses, df_flujos, 
                #                    prefijo=archivo.replace('.raw', ''))
                
                print("\n" + "="*80)
                print("ANÁLISIS COMPLETADO EXITOSAMENTE")
                print("="*80 + "\n")
            else:
                print("\nEl análisis no se completó correctamente\n")
                
        except Exception as e:
            print(f"\nERROR GENERAL: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n\n")  # Separador entre archivos