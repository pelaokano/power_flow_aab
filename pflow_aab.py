import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

@dataclass
class Bus:
    """Representa un bus del sistema eléctrico"""
    number: int
    name: str
    base_kv: float
    type: int  # 1=PQ, 2=PV, 3=Slack
    voltage: complex = 1.0 + 0j
    pg: float = 0.0  # Generación activa
    qg: float = 0.0  # Generación reactiva
    pl: float = 0.0  # Carga activa
    ql: float = 0.0  # Carga reactiva
    
@dataclass
class Branch:
    """Representa una línea o transformador"""
    from_bus: int
    to_bus: int
    r: float  # Resistencia
    x: float  # Reactancia
    b: float  # Susceptancia
    status: int = 1

class RAWReader:
    """Lee archivos .raw de PSS/E"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.buses: Dict[int, Bus] = {}
        self.branches: List[Branch] = []
        self.case_data = {}
        
    def read(self):
        """Lee el archivo .raw y extrae la información"""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Primera línea contiene información del caso
        self.case_data = self._parse_case_line(lines[0])
        
        # Buscar secciones
        sections = self._identify_sections(lines)
        
        # Parsear buses
        if 'BUS' in sections:
            self._parse_buses(lines, sections['BUS'])
        
        # Parsear cargas
        if 'LOAD' in sections:
            self._parse_loads(lines, sections['LOAD'])
        
        # Parsear generadores
        if 'GENERATOR' in sections:
            self._parse_generators(lines, sections['GENERATOR'])
        
        # Parsear ramas
        if 'BRANCH' in sections:
            self._parse_branches(lines, sections['BRANCH'])
        
        return self
    
    def _parse_case_line(self, line: str) -> dict:
        """Parsea la primera línea con información del caso"""
        parts = [p.strip() for p in line.split(',')]
        return {
            'base_mva': float(parts[1]) if len(parts) > 1 else 100.0
        }
    
    def _identify_sections(self, lines: List[str]) -> Dict[str, Tuple[int, int]]:
        """Identifica las secciones del archivo"""
        sections = {}
        current_section = None
        start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if 'BEGIN BUS DATA' in line.upper():
                current_section = 'BUS'
                start_idx = i + 1
            elif 'BEGIN LOAD DATA' in line.upper():
                if current_section:
                    sections[current_section] = (start_idx, i)
                current_section = 'LOAD'
                start_idx = i + 1
            elif 'BEGIN GENERATOR DATA' in line.upper():
                if current_section:
                    sections[current_section] = (start_idx, i)
                current_section = 'GENERATOR'
                start_idx = i + 1
            elif 'BEGIN BRANCH DATA' in line.upper():
                if current_section:
                    sections[current_section] = (start_idx, i)
                current_section = 'BRANCH'
                start_idx = i + 1
            elif line.startswith('0 /') or line == '0':
                if current_section:
                    sections[current_section] = (start_idx, i)
                    current_section = None
        
        return sections
    
    def _parse_buses(self, lines: List[str], section: Tuple[int, int]):
        """Parsea los datos de buses"""
        for i in range(section[0], section[1]):
            line = lines[i].strip()
            if not line or line.startswith('0'):
                break
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            
            bus_num = int(parts[0])
            self.buses[bus_num] = Bus(
                number=bus_num,
                name=parts[1].strip("'\""),
                base_kv=float(parts[2]) if parts[2] else 1.0,
                type=int(parts[3]) if parts[3] else 1
            )
    
    def _parse_loads(self, lines: List[str], section: Tuple[int, int]):
        """Parsea los datos de cargas"""
        for i in range(section[0], section[1]):
            line = lines[i].strip()
            if not line or line.startswith('0'):
                break
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue
            
            bus_num = int(parts[0])
            if bus_num in self.buses:
                self.buses[bus_num].pl += float(parts[5]) if parts[5] else 0.0
                self.buses[bus_num].ql += float(parts[6]) if parts[6] else 0.0
    
    def _parse_generators(self, lines: List[str], section: Tuple[int, int]):
        """Parsea los datos de generadores"""
        for i in range(section[0], section[1]):
            line = lines[i].strip()
            if not line or line.startswith('0'):
                break
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue
            
            bus_num = int(parts[0])
            if bus_num in self.buses:
                self.buses[bus_num].pg += float(parts[2]) if parts[2] else 0.0
                self.buses[bus_num].qg += float(parts[3]) if parts[3] else 0.0
    
    def _parse_branches(self, lines: List[str], section: Tuple[int, int]):
        """Parsea los datos de ramas (líneas y transformadores)"""
        for i in range(section[0], section[1]):
            line = lines[i].strip()
            if not line or line.startswith('0'):
                break
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            
            self.branches.append(Branch(
                from_bus=int(parts[0]),
                to_bus=int(parts[1]),
                r=float(parts[3]) if parts[3] else 0.0,
                x=float(parts[4]) if parts[4] else 0.001,
                b=float(parts[5]) if parts[5] else 0.0,
                status=int(parts[6]) if parts[6] else 1
            ))

class PowerFlowDC:
    """Flujo de potencia DC (linealizado)"""
    
    def __init__(self, buses: Dict[int, Bus], branches: List[Branch], base_mva: float = 100.0):
        self.buses = buses
        self.branches = branches
        self.base_mva = base_mva
        self.bus_list = sorted(buses.keys())
        self.n_buses = len(self.bus_list)
        self.bus_idx = {bus: idx for idx, bus in enumerate(self.bus_list)}
        
    def build_b_matrix(self) -> np.ndarray:
        """Construye la matriz B' (susceptancia)"""
        B = np.zeros((self.n_buses, self.n_buses))
        
        for branch in self.branches:
            if branch.status == 0:
                continue
            
            i = self.bus_idx[branch.from_bus]
            j = self.bus_idx[branch.to_bus]
            bij = 1.0 / branch.x if branch.x != 0 else 0
            
            B[i, i] += bij
            B[j, j] += bij
            B[i, j] -= bij
            B[j, i] -= bij
        
        return B
    
    def solve(self) -> Dict[int, float]:
        """Resuelve el flujo DC"""
        # Construir matriz B
        B = self.build_b_matrix()
        
        # Vector de potencia neta
        P = np.zeros(self.n_buses)
        slack_idx = None
        
        for idx, bus_num in enumerate(self.bus_list):
            bus = self.buses[bus_num]
            P[idx] = (bus.pg - bus.pl) / self.base_mva
            
            if bus.type == 3:  # Slack bus
                slack_idx = idx
        
        # Eliminar ecuación del slack
        if slack_idx is None:
            slack_idx = 0
        
        B_red = np.delete(np.delete(B, slack_idx, 0), slack_idx, 1)
        P_red = np.delete(P, slack_idx)
        
        # Resolver sistema lineal
        theta_red = np.linalg.solve(B_red, P_red)
        
        # Insertar ángulo slack (0)
        theta = np.insert(theta_red, slack_idx, 0.0)
        
        # Retornar diccionario con ángulos
        results = {}
        for idx, bus_num in enumerate(self.bus_list):
            results[bus_num] = theta[idx]
        
        return results
    
    def calculate_flows(self, theta: Dict[int, float]) -> pd.DataFrame:
        """Calcula los flujos en las líneas"""
        flows = []
        
        for branch in self.branches:
            if branch.status == 0:
                continue
            
            theta_i = theta[branch.from_bus]
            theta_j = theta[branch.to_bus]
            
            # Flujo DC: P = (theta_i - theta_j) / x
            flow = (theta_i - theta_j) / branch.x * self.base_mva
            
            flows.append({
                'from_bus': branch.from_bus,
                'to_bus': branch.to_bus,
                'flow_mw': flow
            })
        
        return pd.DataFrame(flows)

class PowerFlowAC:
    """Flujo de potencia AC completo (Newton-Raphson)"""
    
    def __init__(self, buses: Dict[int, Bus], branches: List[Branch], base_mva: float = 100.0):
        self.buses = buses
        self.branches = branches
        self.base_mva = base_mva
        self.bus_list = sorted(buses.keys())
        self.n_buses = len(self.bus_list)
        self.bus_idx = {bus: idx for idx, bus in enumerate(self.bus_list)}
        self.Y = None
        
    def build_ybus(self) -> np.ndarray:
        """Construye la matriz de admitancias"""
        Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        for branch in self.branches:
            if branch.status == 0:
                continue
            
            i = self.bus_idx[branch.from_bus]
            j = self.bus_idx[branch.to_bus]
            
            # Admitancia serie
            y = 1.0 / (branch.r + 1j * branch.x) if (branch.r + branch.x) != 0 else 0
            
            # Admitancia shunt
            b_shunt = 1j * branch.b / 2.0
            
            Y[i, i] += y + b_shunt
            Y[j, j] += y + b_shunt
            Y[i, j] -= y
            Y[j, i] -= y
        
        self.Y = Y
        return Y
    
    def solve(self, max_iter: int = 20, tol: float = 1e-6) -> Dict[int, complex]:
        """Resuelve el flujo AC usando Newton-Raphson"""
        # Construir Ybus
        self.build_ybus()
        
        # Inicializar voltajes (flat start)
        V = np.ones(self.n_buses, dtype=complex)
        
        # Identificar tipos de buses
        pq_buses = []
        pv_buses = []
        slack_bus = None
        
        for idx, bus_num in enumerate(self.bus_list):
            bus = self.buses[bus_num]
            if bus.type == 1:  # PQ
                pq_buses.append(idx)
            elif bus.type == 2:  # PV
                pv_buses.append(idx)
            elif bus.type == 3:  # Slack
                slack_bus = idx
        
        # Iteración Newton-Raphson
        for iteration in range(max_iter):
            # Calcular desajustes
            S_calc = V * np.conj(self.Y @ V)
            
            deltaP = np.zeros(self.n_buses)
            deltaQ = np.zeros(len(pq_buses))
            
            for idx, bus_num in enumerate(self.bus_list):
                if idx == slack_bus:
                    continue
                bus = self.buses[bus_num]
                P_esp = (bus.pg - bus.pl) / self.base_mva
                deltaP[idx] = P_esp - S_calc[idx].real
            
            for i, idx in enumerate(pq_buses):
                bus = self.buses[self.bus_list[idx]]
                Q_esp = (bus.qg - bus.ql) / self.base_mva
                deltaQ[i] = Q_esp - S_calc[idx].imag
            
            # Formar vector de desajustes
            mismatch = np.concatenate([
                np.delete(deltaP, slack_bus),
                deltaQ
            ])
            
            # Verificar convergencia
            if np.max(np.abs(mismatch)) < tol:
                print(f"Convergencia alcanzada en {iteration + 1} iteraciones")
                break
            
            # Construir Jacobiano (simplificado)
            J = self._build_jacobian(V, pq_buses, pv_buses, slack_bus)
            
            # Resolver sistema
            dx = np.linalg.solve(J, mismatch)
            
            # Actualizar variables
            n_theta = self.n_buses - 1
            dtheta = np.insert(dx[:n_theta], slack_bus, 0.0)
            
            for idx in pq_buses:
                dV = dx[n_theta + pq_buses.index(idx)]
                V[idx] *= (1 + dV)
            
            # Actualizar ángulos
            V = np.abs(V) * np.exp(1j * (np.angle(V) + dtheta))
        
        # Retornar resultados
        results = {}
        for idx, bus_num in enumerate(self.bus_list):
            results[bus_num] = V[idx]
        
        return results
    
    def _build_jacobian(self, V: np.ndarray, pq_buses: List[int], 
                       pv_buses: List[int], slack_bus: int) -> np.ndarray:
        """Construye el Jacobiano (versión simplificada)"""
        n = self.n_buses - 1  # Excluir slack
        m = len(pq_buses)
        J = np.zeros((n + m, n + m))
        
        # Llenar con aproximación (esto es una versión simplificada)
        # En una implementación real se calcularían las derivadas exactas
        for i in range(n + m):
            J[i, i] = 1.0
        
        return J

# Ejemplo de uso
if __name__ == "__main__":
    # Leer archivo
    reader = RAWReader("sistema.raw")
    reader.read()
    
    print(f"Buses leídos: {len(reader.buses)}")
    print(f"Ramas leídas: {len(reader.branches)}")
    
    # Flujo DC
    print("\n=== FLUJO DE POTENCIA DC ===")
    pf_dc = PowerFlowDC(reader.buses, reader.branches, reader.case_data.get('base_mva', 100))
    theta_dc = pf_dc.solve()
    flows_dc = pf_dc.calculate_flows(theta_dc)
    print("\nÁngulos (radianes):")
    for bus, angle in theta_dc.items():
        print(f"  Bus {bus}: {angle:.6f} rad ({np.degrees(angle):.2f}°)")
    
    # Flujo AC
    print("\n=== FLUJO DE POTENCIA AC ===")
    pf_ac = PowerFlowAC(reader.buses, reader.branches, reader.case_data.get('base_mva', 100))
    V_ac = pf_ac.solve()
    print("\nVoltajes:")
    for bus, v in V_ac.items():
        print(f"  Bus {bus}: {abs(v):.4f} ∠ {np.degrees(np.angle(v)):.2f}°")