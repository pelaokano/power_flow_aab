"""
EJECUTAR FLUJO DE POTENCIA TRIFÁSICO DESEQUILIBRADO
Permite simular sistemas con cargas y voltajes desequilibrados
"""

import numpy as np
import matplotlib.pyplot as plt
from read_raw import RawParser
from power_flow_des import PowerFlowTrifasico


def crear_configuracion_desequilibrada(df_cargas, nivel_desequilibrio='moderado'):

    config = {}
    
    # Distribuciones típicas por nivel de desequilibrio
    distribuciones = {
        'balanceado': [1/3, 1/3, 1/3],          # Perfectamente balanceado
        'leve': [0.35, 0.33, 0.32],             # 3% desequilibrio
        'moderado': [0.40, 0.35, 0.25],         # 15% desequilibrio
        'severo': [0.50, 0.30, 0.20],           # 30% desequilibrio
        'muy_severo': [0.60, 0.25, 0.15],       # 45% desequilibrio
    }
    
    dist_default = distribuciones.get(nivel_desequilibrio, distribuciones['moderado'])
    config['distribucion_default'] = dist_default
    
    # Configurar cargas individuales con variación
    if df_cargas is not None:
        for idx in range(len(df_cargas)):
            # Añadir variación aleatoria a cada carga
            variacion = np.random.uniform(-0.05, 0.05, 3)
            dist_carga = np.array(dist_default) + variacion
            dist_carga = np.maximum(dist_carga, 0.1)  # Mínimo 10% por fase
            dist_carga = dist_carga / dist_carga.sum()  # Normalizar
            
            config[f'carga_{idx}'] = {
                'distribucion': dist_carga.tolist()
            }
    
    return config

def crear_configuracion_desequilibrada2(df_cargas, nivel_desequilibrio='moderado'):
    import numpy as np
    
    config = {}
    
    # Configurar cargas individuales con variación
    if df_cargas is not None:
        for idx in range(len(df_cargas)):
            # Empezar con distribución equilibrada (33.33% cada fase)
            dist_equilibrada = np.array([1/3, 1/3, 1/3])
            
            # Aplicar variación aleatoria de ±1% a ±3% a cada fase
            variacion = np.random.uniform(-0.001, 0.001, 3)
            dist_carga = dist_equilibrada * (1 + variacion)
            
            # Normalizar para que sume exactamente 1
            dist_carga = dist_carga / dist_carga.sum()
            
            config[f'carga_{idx}'] = {
                'distribucion': dist_carga.tolist()
            }
    
    return config


def ejecutar_flujo_trifasico(archivo_raw, nivel_desequilibrio='moderado'):
    """
    Ejecuta flujo de potencia trifásico con cargas desequilibradas
    
    Args:
        archivo_raw: ruta del archivo .raw
        nivel_desequilibrio: nivel de desequilibrio de cargas
    """
    
    print("╔" + "═"*78 + "╗")
    print("║" + " FLUJO DE POTENCIA TRIFÁSICO DESEQUILIBRADO ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\nArchivo: {archivo_raw}")
    print(f"Nivel de desequilibrio: {nivel_desequilibrio}\n")
    
    # ============================================
    # PASO 1: LEER ARCHIVO .RAW
    # ============================================
    print("[1/4] LECTURA DE ARCHIVO .RAW")
    print("-" * 80)
    
    try:
        parser = RawParser()
        parser.leer_archivo(archivo_raw)
        datos = parser.obtener_dataframes()
        
        print(f"\n Archivo leído exitosamente")
        print(f"  - Buses: {len(datos['buses'])}")
        print(f"  - Cargas: {len(datos['cargas']) if datos['cargas'] is not None else 0}")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        return None
    
    # ============================================
    # PASO 2: CONFIGURAR DESEQUILIBRIO
    # ============================================
    print("\n\n[2/4] CONFIGURAR DESEQUILIBRIO DE CARGAS")
    print("-" * 80)
    
    config_3ph = crear_configuracion_desequilibrada2(
        datos['cargas'], 
        nivel_desequilibrio
    )

    print(datos['cargas'])
    
    # print(f"✓ Configuración creada")
    # print(f"  Distribución por defecto: ")
    # dist = config_3ph['distribucion_default']
    # print(f"    Fase A: {dist[0]*100:.1f}%")
    # print(f"    Fase B: {dist[1]*100:.1f}%")
    # print(f"    Fase C: {dist[2]*100:.1f}%")
    
    # ============================================
    # PASO 3: RESOLVER FLUJO TRIFÁSICO
    # ============================================
    print("\n\n[3/4] RESOLVER FLUJO DE POTENCIA TRIFÁSICO")
    print("-" * 80)
    
    try:
        solver = PowerFlowTrifasico(datos, config_3ph)
        convergencia = solver.resolver(max_iter=50, tolerancia=5e+1)
        
        if not convergencia:
            print("\n✗ El flujo de potencia no convergió")
            return None
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================
    # PASO 4: GENERAR REPORTES
    # ============================================
    print("\n\n[4/4] GENERAR REPORTES Y GRÁFICAS")
    print("-" * 80)
    
    try:
        df_resultados = solver.generar_reporte()
        
        # Crear visualizaciones
        # crear_graficas_trifasicas(solver, df_resultados, nivel_desequilibrio)
        
        return solver, df_resultados
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def crear_graficas_trifasicas(solver, df_resultados, nivel_desequilibrio):
    """
    Crea gráficas específicas para análisis trifásico
    """
    print("\nGenerando gráficas trifásicas...")
    
    fig = plt.figure(figsize=(16, 10))
    
    n_buses = solver.n_buses
    V_complex = solver.V_complex
    
    # 1. Voltajes por fase
    ax1 = plt.subplot(2, 3, 1)
    buses = np.arange(n_buses)
    
    Va = [abs(V_complex[3*i]) for i in range(n_buses)]
    Vb = [abs(V_complex[3*i + 1]) for i in range(n_buses)]
    Vc = [abs(V_complex[3*i + 2]) for i in range(n_buses)]
    
    ax1.plot(buses, Va, 'r-o', label='Fase A', markersize=4, alpha=0.7)
    ax1.plot(buses, Vb, 'g-s', label='Fase B', markersize=4, alpha=0.7)
    ax1.plot(buses, Vc, 'b-^', label='Fase C', markersize=4, alpha=0.7)
    ax1.axhline(y=1.05, color='gray', linestyle='--', linewidth=1)
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Bus')
    ax1.set_ylabel('|V| (pu)')
    ax1.set_title('Perfil de Voltajes por Fase')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ángulos por fase
    ax2 = plt.subplot(2, 3, 2)
    ang_a = [np.rad2deg(np.angle(V_complex[3*i])) for i in range(n_buses)]
    ang_b = [np.rad2deg(np.angle(V_complex[3*i + 1])) for i in range(n_buses)]
    ang_c = [np.rad2deg(np.angle(V_complex[3*i + 2])) for i in range(n_buses)]
    
    ax2.plot(buses, ang_a, 'r-o', label='Fase A', markersize=4, alpha=0.7)
    ax2.plot(buses, ang_b, 'g-s', label='Fase B', markersize=4, alpha=0.7)
    ax2.plot(buses, ang_c, 'b-^', label='Fase C', markersize=4, alpha=0.7)
    ax2.set_xlabel('Bus')
    ax2.set_ylabel('Ángulo (°)')
    ax2.set_title('Ángulos de Fase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Desequilibrio de voltajes
    ax3 = plt.subplot(2, 3, 3)
    desequilibrios = []
    for i in range(n_buses):
        Va_i = V_complex[3*i]
        Vb_i = V_complex[3*i + 1]
        Vc_i = V_complex[3*i + 2]
        V_avg = (Va_i + Vb_i + Vc_i) / 3
        if abs(V_avg) > 0:
            dev_max = max(abs(Va_i - V_avg), abs(Vb_i - V_avg), abs(Vc_i - V_avg))
            deseq = (dev_max / abs(V_avg)) * 100
        else:
            deseq = 0
        desequilibrios.append(deseq)
    
    ax3.bar(buses, desequilibrios, color='orange', alpha=0.7)
    ax3.axhline(y=5, color='r', linestyle='--', label='Límite 5%')
    ax3.set_xlabel('Bus')
    ax3.set_ylabel('Desequilibrio (%)')
    ax3.set_title('Desequilibrio de Voltajes por Bus')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Diagrama fasorial (bus con mayor desequilibrio)
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    bus_max_deseq = np.argmax(desequilibrios)
    
    Va_max = V_complex[3*bus_max_deseq]
    Vb_max = V_complex[3*bus_max_deseq + 1]
    Vc_max = V_complex[3*bus_max_deseq + 2]
    
    ax4.plot([0, np.angle(Va_max)], [0, abs(Va_max)], 'r-o', linewidth=2, 
             markersize=8, label=f'Va ({abs(Va_max):.3f}∠{np.rad2deg(np.angle(Va_max)):.1f}°)')
    ax4.plot([0, np.angle(Vb_max)], [0, abs(Vb_max)], 'g-s', linewidth=2, 
             markersize=8, label=f'Vb ({abs(Vb_max):.3f}∠{np.rad2deg(np.angle(Vb_max)):.1f}°)')
    ax4.plot([0, np.angle(Vc_max)], [0, abs(Vc_max)], 'b-^', linewidth=2, 
             markersize=8, label=f'Vc ({abs(Vc_max):.3f}∠{np.rad2deg(np.angle(Vc_max)):.1f}°)')
    
    ax4.set_ylim(0, 1.2)
    ax4.set_title(f'Diagrama Fasorial - Bus {solver.idx_to_bus_num[bus_max_deseq]}\n(Mayor desequilibrio: {desequilibrios[bus_max_deseq]:.2f}%)', 
                  pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 5. Comparación de magnitudes
    ax5 = plt.subplot(2, 3, 5)
    x = np.arange(3)
    V_fase_avg = [np.mean(Va), np.mean(Vb), np.mean(Vc)]
    colores_fase = ['red', 'green', 'blue']
    
    bars = ax5.bar(x, V_fase_avg, color=colores_fase, alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Fase A', 'Fase B', 'Fase C'])
    ax5.set_ylabel('|V| promedio (pu)')
    ax5.set_title('Voltaje Promedio por Fase')
    ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, val in zip(bars, V_fase_avg):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # 6. Distribución de potencia por fase
    ax6 = plt.subplot(2, 3, 6)
    P_fase_a = sum(solver.P_spec[i] for i in range(0, solver.n_nodes, 3)) * solver.base_mva
    P_fase_b = sum(solver.P_spec[i] for i in range(1, solver.n_nodes, 3)) * solver.base_mva
    P_fase_c = sum(solver.P_spec[i] for i in range(2, solver.n_nodes, 3)) * solver.base_mva
    
    potencias = [abs(P_fase_a), abs(P_fase_b), abs(P_fase_c)]
    bars = ax6.bar(x, potencias, color=colores_fase, alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Fase A', 'Fase B', 'Fase C'])
    ax6.set_ylabel('Potencia (MW)')
    ax6.set_title('Distribución de Carga por Fase')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores y porcentajes
    P_total = sum(potencias)
    for bar, val in zip(bars, potencias):
        height = bar.get_height()
        pct = (val / P_total) * 100
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MW\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.suptitle(f'Análisis Trifásico - Desequilibrio {nivel_desequilibrio.upper()}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    nombre_archivo = f'flujo_trifasico_{nivel_desequilibrio}.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en '{nombre_archivo}'")


def comparar_balanceado_vs_desequilibrado(archivo_raw):
    """
    Compara resultados entre flujo balanceado y desequilibrado
    """
    from power_flow_nr import PowerFlowNR
    
    print("\n" + "="*80)
    print("COMPARACIÓN: BALANCEADO vs DESEQUILIBRADO")
    print("="*80)
    
    # Leer archivo
    parser = RawParser()
    parser.leer_archivo(archivo_raw)
    datos = parser.obtener_dataframes()
    
    # 1. Flujo balanceado (monofásico equivalente)
    print("\n▶ FLUJO BALANCEADO (Monofásico):")
    print("-" * 40)
    solver_bal = PowerFlowNR(datos)
    conv_bal = solver_bal.resolver(max_iter=20, tolerancia=1e-6)
    
    if conv_bal:
        df_buses_bal, df_flujos_bal = solver_bal.generar_reporte()
        print(f"✓ Convergió")
    
    # 2. Flujo desequilibrado moderado
    print("\n▶ FLUJO DESEQUILIBRADO (Trifásico - Moderado):")
    print("-" * 40)
    config_3ph = crear_configuracion_desequilibrada2(datos['cargas'], 'moderado')
    solver_deseq = PowerFlowTrifasico(datos, config_3ph)
    conv_deseq = solver_deseq.resolver(max_iter=20, tolerancia=1e-6)
    
    if conv_deseq:
        df_resultados_deseq = solver_deseq.generar_reporte()
        print(f"✓ Convergió")
    
    # Comparación
    if conv_bal and conv_deseq:
        print("\n\nCOMPARACIÓN DE RESULTADOS:")
        print("-" * 80)
        
        # Comparar voltajes promedio
        V_bal_avg = df_buses_bal['V (pu)'].mean()
        
        V_a_avg = np.mean([abs(solver_deseq.V_complex[3*i]) for i in range(solver_deseq.n_buses)])
        V_b_avg = np.mean([abs(solver_deseq.V_complex[3*i+1]) for i in range(solver_deseq.n_buses)])
        V_c_avg = np.mean([abs(solver_deseq.V_complex[3*i+2]) for i in range(solver_deseq.n_buses)])
        V_deseq_avg = (V_a_avg + V_b_avg + V_c_avg) / 3
        
        print(f"Voltaje promedio:")
        print(f"  Balanceado:    {V_bal_avg:.6f} pu")
        print(f"  Desequilibrado: {V_deseq_avg:.6f} pu (promedio fases)")
        print(f"    Fase A: {V_a_avg:.6f} pu")
        print(f"    Fase B: {V_b_avg:.6f} pu")
        print(f"    Fase C: {V_c_avg:.6f} pu")
        print(f"  Diferencia: {abs(V_bal_avg - V_deseq_avg):.2e} pu")


def analizar_sensibilidad_desequilibrio(archivo_raw):
    """
    Analiza cómo diferentes niveles de desequilibrio afectan el sistema
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE SENSIBILIDAD AL DESEQUILIBRIO")
    print("="*80)
    
    parser = RawParser()
    parser.leer_archivo(archivo_raw)
    datos = parser.obtener_dataframes()
    
    niveles = ['balanceado', 'leve', 'moderado', 'severo']
    resultados = []
    
    for nivel in niveles:
        print(f"\n▶ Analizando nivel: {nivel.upper()}")
        print("-" * 40)
        
        config_3ph = crear_configuracion_desequilibrada2(datos['cargas'], nivel)
        solver = PowerFlowTrifasico(datos, config_3ph)
        convergencia = solver.resolver(max_iter=20, tolerancia=1e-6)
        
        if convergencia:
            # Calcular desequilibrio promedio
            deseq_total = 0
            for i in range(solver.n_buses):
                Va = solver.V_complex[3*i]
                Vb = solver.V_complex[3*i + 1]
                Vc = solver.V_complex[3*i + 2]
                V_avg = (Va + Vb + Vc) / 3
                if abs(V_avg) > 0:
                    dev_max = max(abs(Va - V_avg), abs(Vb - V_avg), abs(Vc - V_avg))
                    deseq = (dev_max / abs(V_avg)) * 100
                    deseq_total += deseq
            
            deseq_promedio = deseq_total / solver.n_buses
            
            resultados.append({
                'Nivel': nivel,
                'Deseq_promedio (%)': deseq_promedio,
                'Convergencia': 'Sí'
            })
            
            print(f"✓ Desequilibrio promedio: {deseq_promedio:.2f}%")
        else:
            resultados.append({
                'Nivel': nivel,
                'Deseq_promedio (%)': 0,
                'Convergencia': 'No'
            })
            print("✗ No convergió")
    
    # Crear gráfica comparativa
    df_sens = pd.DataFrame(resultados)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_sens['Nivel'], df_sens['Deseq_promedio (%)'], 
            color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Nivel de Desequilibrio de Cargas')
    plt.ylabel('Desequilibrio de Voltaje Promedio (%)')
    plt.title('Sensibilidad del Sistema al Desequilibrio de Cargas')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('sensibilidad_desequilibrio.png', dpi=150)
    print("\n✓ Gráfica de sensibilidad guardada en 'sensibilidad_desequilibrio.png'")


# ============================================
# EJECUTAR ANÁLISIS
# ============================================
if __name__ == "__main__":
    
    archivo = "IEEE 118 Bus v2.raw"
    # archivo = "IEEE 14 bus.raw"

    
    # ========================================
    # OPCIÓN 1: Flujo trifásico simple
    # ========================================
    print("="*80)
    print("OPCIÓN 1: FLUJO TRIFÁSICO DESEQUILIBRADO")
    print("="*80)
    
    # Niveles disponibles: 'balanceado', 'leve', 'moderado', 'severo', 'muy_severo'
    resultado = ejecutar_flujo_trifasico(archivo, nivel_desequilibrio='moderado')
    
    if resultado:
        solver, df_resultados = resultado
        print("\nAnálisis completado exitosamente")
    
    # ========================================
    # OPCIÓN 2: Comparación (descomentar para ejecutar)
    # ========================================
    # print("\n\n" + "="*80)
    # print("OPCIÓN 2: COMPARACIÓN BALANCEADO vs DESEQUILIBRADO")
    # print("="*80)
    # comparar_balanceado_vs_desequilibrado(archivo)
    
    # ========================================
    # OPCIÓN 3: Análisis de sensibilidad (descomentar para ejecutar)
    # ========================================
    # print("\n\n" + "="*80)
    # print("OPCIÓN 3: ANÁLISIS DE SENSIBILIDAD")
    # print("="*80)
    # analizar_sensibilidad_desequilibrio(archivo)