"""
IEEE 33-Bus Distribution System simulator.

Wraps OpenDSS via opendssdirect to run AC power flow for arbitrary
load scenarios and switch configurations obtained by the classical
Baran & Wu (1989) branch-exchange reconfiguration algorithm.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import opendssdirect as dss

# 32 sectionalizing branches (normally-closed) (from_bus, to_bus, R_ohm, X_ohm)
IEEE33_LINE_DATA: List[Tuple[str, str, float, float]] = [
    ('1',  '2',  0.0922, 0.0477),
    ('2',  '3',  0.4930, 0.2511),
    ('3',  '4',  0.3660, 0.1864),
    ('4',  '5',  0.3811, 0.1941),
    ('5',  '6',  0.8190, 0.7070),
    ('6',  '7',  0.1872, 0.6188),
    ('7',  '8',  1.7114, 1.2351),
    ('8',  '9',  1.0300, 0.7400),
    ('9',  '10', 1.0440, 0.7400),
    ('10', '11', 0.1966, 0.0650),
    ('11', '12', 0.3744, 0.1238),
    ('12', '13', 1.4680, 1.1550),
    ('13', '14', 0.5416, 0.7129),
    ('14', '15', 0.5910, 0.5260),
    ('15', '16', 0.7463, 0.5450),
    ('16', '17', 1.2890, 1.7210),
    ('17', '18', 0.7320, 0.5740),
    ('2',  '19', 0.1640, 0.1565),
    ('19', '20', 1.5042, 1.3554),
    ('20', '21', 0.4095, 0.4784),
    ('21', '22', 0.7089, 0.9373),
    ('3',  '23', 0.4512, 0.3083),
    ('23', '24', 0.8980, 0.7091),
    ('24', '25', 0.8960, 0.7011),
    ('6',  '26', 0.2030, 0.1034),
    ('26', '27', 0.2842, 0.1447),
    ('27', '28', 1.0590, 0.9337),
    ('28', '29', 0.8042, 0.7006),
    ('29', '30', 0.5075, 0.2585),
    ('30', '31', 0.9744, 0.9630),
    ('31', '32', 0.3105, 0.3619),
    ('32', '33', 0.3410, 0.5302),
]

# 5 tie switches (normally-open)
IEEE33_TIE_LINES: List[Tuple[str, str, float, float]] = [
    ('8',  '21', 2.0000, 2.0000),
    ('9',  '15', 2.0000, 2.0000),
    ('12', '22', 2.0000, 2.0000),
    ('18', '33', 0.5000, 0.5000),
    ('25', '29', 0.5000, 0.5000),
]

IEEE33_ALL_BRANCHES = IEEE33_LINE_DATA + IEEE33_TIE_LINES

# Nominal load: (P_kW, Q_kvar) at each load bus
IEEE33_LOAD_DATA: Dict[str, Tuple[float, float]] = {
    '2':  (100,  60),  '3':  (90,   40),  '4':  (120,  80),
    '5':  (60,   30),  '6':  (60,   20),  '7':  (200, 100),
    '8':  (200, 100),  '9':  (60,   20),  '10': (60,   20),
    '11': (45,   30),  '12': (60,   35),  '13': (60,   35),
    '14': (120,  80),  '15': (60,   10),  '16': (60,   20),
    '17': (60,   20),  '18': (90,   40),  '19': (90,   40),
    '20': (90,   40),  '21': (90,   40),  '22': (90,   40),
    '23': (90,   50),  '24': (420, 200),  '25': (420, 200),
    '26': (60,   25),  '27': (60,   25),  '28': (60,   20),
    '29': (120,  80),  '30': (200, 100),  '31': (150,  70),
    '32': (210, 100),  '33': (60,   40),
}

# Canonical bus ordering (string-sorted for reproducibility)
ALL_BUSES: List[str] = sorted(
    {b for line in IEEE33_ALL_BRANCHES for b in (line[0], line[1])}
    | set(IEEE33_LOAD_DATA.keys())
)

assert len(ALL_BUSES) == 33
assert len(IEEE33_ALL_BRANCHES) == 37

class IEEE33BusSimulator:
    """
    Generates randomised (load, topology) -> (V, θ) scenarios.

    Each scenario is produced by:
      1. Branch-exchange reconfiguration  ->  feasible radial topology 
      (Closing one + opening a sectionalizing branch creates valid radial topologies)
      2. Gaussian load sampling with fixed power-factor per bus
      3. 3-phase AC power flow via OpenDSS
    """

    BASEKV = 12.66   # kV line-to-line
    BASE_KVA = 1000.0  # 1 MVA base for per-unit conversion

    def __init__(self) -> None:
        self.sectionalizing = IEEE33_LINE_DATA
        self.tie_lines      = IEEE33_TIE_LINES
        self.all_branches   = IEEE33_ALL_BRANCHES
        self.load_data      = IEEE33_LOAD_DATA
        self.bus_set        = ALL_BUSES

    def sample_load_scenario(self, load_std_fraction: float = 0.15) -> Dict[str, float]:
        """
        Sample P, Q at every load bus.

        Power-factor is preserved per bus; only the kVA magnitude varies
        as N(P_base, σ=0.15*P_base), clipped to a minimum of 0.1 kW.
        """
        rng = np.random.default_rng()
        scenario: Dict[str, float] = {}

        for bus, (p_base, q_base) in self.load_data.items():
            s_base  = math.hypot(p_base, q_base)
            pf      = p_base / s_base
            tan_phi = math.tan(math.acos(pf))

            p_rand = float(max(0.1, rng.normal(p_base, load_std_fraction * p_base)))
            q_rand = p_rand * tan_phi

            scenario[f"P_{bus}"] = p_rand
            scenario[f"Q_{bus}"] = q_rand

        return scenario

    def generate_radial_topology(self) -> nx.Graph:
        """
        Return a feasible radial topology as a NetworkX graph with
        edge attributes {r, x}.

        Algorithm (Baran & Wu 1989):
          1. Start with the base radial tree (32 sectionalizing lines).
          2. Close one tie switch at random  ->  creates one loop.
          3. Find cycle via NetworkX and open one sectionalizing branch on that loop at random.
          4. Assert the result is still a spanning tree.
        """
        G = nx.Graph()
        for fb, tb, r, x in self.sectionalizing:
            G.add_edge(fb, tb, r=r, x=x)

        tie = random.choice(self.tie_lines)
        fb_t, tb_t, r_t, x_t = tie
        G.add_edge(fb_t, tb_t, r=r_t, x=x_t)

        cycle = nx.find_cycle(G)

        # Identify sectionalizing branches in the cycle
        sec_set = {(fb, tb) for fb, tb, *_ in self.sectionalizing}
        sec_set |= {(tb, fb) for fb, tb, *_ in self.sectionalizing}

        candidates = [
            (u, v) for (u, v, *_) in cycle
            if (u, v) in sec_set
        ]

        u_open, v_open = random.choice(candidates)
        G.remove_edge(u_open, v_open)

        assert nx.is_tree(G), "Branch-exchange produced a non-tree graph."
        return G

    def topology_to_switch_vector(self, G: nx.Graph) -> np.ndarray:
        """Binary vector of length 37: 1 = branch active, 0 = open."""
        s = np.array(
            [1 if G.has_edge(fb, tb) or G.has_edge(tb, fb) else 0
             for fb, tb, *_ in self.all_branches],
            dtype=np.int32,
        )
        assert s.sum() == len(self.bus_set) - 1, \
            f"Radiality violated: {s.sum()} active branches for {len(self.bus_set)} buses."
        return s

    def enabled_lines_from_vector(
        self, switch_vector: np.ndarray
    ) -> List[Tuple[str, str, float, float]]:
        return [
            branch for branch, active in zip(self.all_branches, switch_vector)
            if active == 1
        ]

    def _build_dss_circuit(
        self,
        load_scenario: Dict[str, float],
        switch_vector: np.ndarray,
    ) -> None:
        dss.Basic.ClearAll()
        dss.Text.Command(
            f"New Circuit.IEEE33 bus1=1 basekv={self.BASEKV} pu=1.05 phases=3"
        )
        # Lines (enabled/disabled via switch vector)
        for idx, (fb, tb, r, x) in enumerate(self.all_branches):
            name    = f"L{idx}"
            enabled = "Yes" if switch_vector[idx] == 1 else "No"
            dss.Text.Command(
                f"New Line.{name} bus1={fb} bus2={tb} length=1 units=mi "
                f"phases=3 r1={r} x1={x}"
            )
            dss.Text.Command(f"Edit Line.{name} Enabled={enabled}")

        # Loads
        for bus in self.bus_set:
            p_kw   = load_scenario.get(f"P_{bus}", 0.0)
            q_kvar = load_scenario.get(f"Q_{bus}", 0.0)
            if p_kw < 1e-6:
                continue
            dss.Text.Command(
                f"New Load.Load_{bus} bus1={bus} phases=3 kV={self.BASEKV} "
                f"kW={p_kw} kvar={q_kvar} model=1 conn=wye"
            )

        # Voltage bases must be set AFTER all elements are defined
        dss.Text.Command(f"Set voltagebases=[{self.BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

    def solve(
        self,
        load_scenario: Optional[Dict[str, float]] = None,
        switch_vector: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Run one AC power-flow scenario.

        Returns a flat dict with keys  V_{bus}, Angle_{bus}, P_{bus}, Q_{bus}
        for every bus in ALL_BUSES, or None if the power flow did not converge.
        """
        if load_scenario is None:
            load_scenario = self.sample_load_scenario()
        if switch_vector is None:
            G = self.generate_radial_topology()
            switch_vector = self.topology_to_switch_vector(G)

        self._build_dss_circuit(load_scenario, switch_vector)

        if not dss.Solution.Converged():
            return None

        results: Dict[str, float] = {}
        for bus in self.bus_set:
            dss.Circuit.SetActiveBus(bus)
            vm_ang = dss.Bus.puVmagAngle()   # [V1, ang1, V2, ang2, V3, ang3]
            # Take phase-1 values (balanced 3-phase assumption)
            results[f"V_{bus}"]     = float(vm_ang[0])
            results[f"Angle_{bus}"] = float(vm_ang[1])
            results[f"P_{bus}"]     = load_scenario.get(f"P_{bus}", 0.0)
            results[f"Q_{bus}"]     = load_scenario.get(f"Q_{bus}", 0.0)

        results["__switch_vector__"] = switch_vector.tolist()
        return results