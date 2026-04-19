"""
IEEE 69-Bus Radial Distribution System simulator.

Grid parameters from:
  Baran & Wu (1989), "Optimal capacitor placement on radial distribution
  systems", IEEE Trans. Power Delivery, 4(1):725-734.
  Das (2008), IJEPES 30:361-367.

System specs:
  • 69 buses, 68 main branches + 5 tie switches
  • Nominal voltage: 12.66 kV (line-to-line)
  • Total load: ~3801.9 kW, ~2694.6 kvar
  • Slack bus: bus "1" (substation), V=1.05 pu
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import opendssdirect as dss

# Static grid data — 68 sectionalizing branches
# (from_bus, to_bus, R_ohm, X_ohm)
IEEE69_LINE_DATA: List[Tuple[str, str, float, float]] = [
    # Feeder trunk (buses 1-27 main path)
    ('1',  '2',  0.0100, 0.0050),  # clamped: original bus-bar impedance unrealistically low
    ('2',  '3',  0.0100, 0.0050),  # clamped
    ('3',  '4',  0.0100, 0.0050),  # clamped
    ('4',  '5',  0.0251, 0.0294),
    ('5',  '6',  0.3660, 0.1864),
    ('6',  '7',  0.3811, 0.1941),
    ('7',  '8',  0.0922, 0.0470),
    ('8',  '9',  0.0493, 0.0251),
    ('9',  '10', 0.8190, 0.2707),
    ('10', '11', 0.1872, 0.0619),
    ('11', '12', 0.7114, 0.2351),
    ('12', '13', 1.0300, 0.3400),
    ('13', '14', 1.0440, 0.3450),
    ('14', '15', 1.0580, 0.3496),
    ('15', '16', 0.1966, 0.0650),
    ('16', '17', 0.3744, 0.1238),
    ('17', '18', 0.0047, 0.0016),
    ('18', '19', 0.3276, 0.1083),
    ('19', '20', 0.2106, 0.0696),
    ('20', '21', 0.3416, 0.1129),
    ('21', '22', 0.0140, 0.0046),
    ('22', '23', 0.1591, 0.0526),
    ('23', '24', 0.3463, 0.1145),
    ('24', '25', 0.7488, 0.2475),
    ('25', '26', 0.3089, 0.1021),
    ('26', '27', 0.1732, 0.0572),
    # Lateral from bus 3
    ('3',  '28', 0.0044, 0.0108),
    ('28', '29', 0.0640, 0.1565),
    ('29', '30', 0.3978, 0.1315),
    ('30', '31', 0.0702, 0.0232),
    ('31', '32', 0.3510, 0.1160),
    ('32', '33', 0.8390, 0.2816),
    ('33', '34', 1.7080, 0.5646),
    ('34', '35', 1.4740, 0.4873),
    # Lateral from bus 3 (second)
    ('3',  '36', 0.0044, 0.0108),
    ('36', '37', 0.0640, 0.1565),
    ('37', '38', 0.1053, 0.1230),
    ('38', '39', 0.0304, 0.0355),
    ('39', '40', 0.0018, 0.0021),
    ('40', '41', 0.7283, 0.8509),
    ('41', '42', 0.3100, 0.3623),
    ('42', '43', 0.0410, 0.0478),
    ('43', '44', 0.0092, 0.0116),
    ('44', '45', 0.1089, 0.1373),
    ('45', '46', 0.0009, 0.0012),
    ('46', '47', 0.0034, 0.0084),
    ('47', '48', 0.0851, 0.2083),
    ('48', '49', 0.2898, 0.7091),
    ('49', '50', 0.0822, 0.2011),
    # Lateral from bus 6
    ('6',  '51', 0.0928, 0.0473),
    ('51', '52', 0.3319, 0.1114),
    ('52', '53', 0.1740, 0.0886),
    ('53', '54', 0.2030, 0.1034),
    # Lateral from bus 11
    ('11', '55', 0.2842, 0.1447),
    ('55', '56', 0.2813, 0.1433),
    ('56', '57', 1.5900, 0.5337),
    ('57', '58', 0.7837, 0.2630),
    ('58', '59', 0.3042, 0.1006),
    # Lateral from bus 13
    ('13', '60', 0.5075, 0.2585),
    ('60', '61', 0.9744, 0.3616),
    ('61', '62', 0.1900, 0.0627),
    ('62', '63', 0.2320, 0.0767),
    ('63', '64', 0.1640, 0.0542),
    ('64', '65', 1.4780, 0.4880),
    # Lateral from bus 27
    ('27', '66', 0.1591, 0.0526),
    ('66', '67', 0.3463, 0.1145),
    ('67', '68', 0.7488, 0.2475),
    ('68', '69', 0.3089, 0.1021),
]

# 5 tie switches (normally-open)
IEEE69_TIE_LINES: List[Tuple[str, str, float, float]] = [
    ('11', '43', 0.5000, 0.5000),
    ('13', '21', 0.5000, 0.5000),
    ('15', '46', 0.5000, 0.5000),
    ('50', '59', 0.5000, 0.5000),
    ('27', '65', 0.5000, 0.5000),
]

IEEE69_ALL_BRANCHES = IEEE69_LINE_DATA + IEEE69_TIE_LINES

# Nominal load: (P_kW, Q_kvar) at each load bus
# Source: Das 2008 / Baran-Wu 69-bus appendix
IEEE69_LOAD_DATA: Dict[str, Tuple[float, float]] = {
    '2':  (0.0,   0.0),
    '3':  (0.0,   0.0),
    '4':  (0.0,   0.0),
    '5':  (2.6,   2.2),
    '6':  (40.4,  30.0),
    '7':  (75.0,  54.0),
    '8':  (30.0,  22.0),
    '9':  (28.0,  19.0),
    '10': (145.0, 104.0),
    '11': (145.0, 104.0),
    '12': (8.0,   5.0),
    '13': (8.0,   5.5),
    '14': (0.0,   0.0),
    '15': (45.5,  30.0),
    '16': (60.0,  35.0),
    '17': (60.0,  35.0),
    '18': (0.0,   0.0),
    '19': (1.0,   0.6),
    '20': (114.0, 81.0),
    '21': (5.0,   3.5),
    '22': (0.0,   0.0),
    '23': (28.0,  20.0),
    '24': (0.0,   0.0),
    '25': (14.0,  10.0),
    '26': (14.0,  10.0),
    '27': (26.0,  18.6),
    '28': (0.0,   0.0),
    '29': (26.0,  18.6),
    '30': (0.0,   0.0),
    '31': (0.0,   0.0),
    '32': (0.0,   0.0),
    '33': (14.0,  10.0),
    '34': (19.5,  14.0),
    '35': (6.0,   4.3),
    '36': (0.0,   0.0),
    '37': (26.0,  18.55),
    '38': (26.0,  18.55),
    '39': (0.0,   0.0),
    '40': (24.0,  17.0),
    '41': (24.0,  17.0),
    '42': (1.2,   1.0),
    '43': (0.0,   0.0),
    '44': (6.0,   4.3),
    '45': (39.22, 26.3),
    '46': (39.22, 26.3),
    '47': (0.0,   0.0),
    '48': (79.0,  56.4),
    '49': (384.7, 274.5),
    '50': (384.7, 274.5),
    '51': (40.5,  28.3),
    '52': (3.6,   2.7),
    '53': (4.35,  3.5),
    '54': (26.4,  19.0),
    '55': (24.0,  17.2),
    '56': (0.0,   0.0),
    '57': (0.0,   0.0),
    '58': (0.0,   0.0),
    '59': (100.0, 72.0),
    '60': (0.0,   0.0),
    '61': (1244.0,888.0),
    '62': (32.0,  23.0),
    '63': (0.0,   0.0),
    '64': (227.0, 162.0),
    '65': (59.0,  42.0),
    '66': (18.0,  13.0),
    '67': (18.0,  13.0),
    '68': (28.0,  20.0),
    '69': (28.0,  20.0),
}

# Canonical bus ordering
ALL_BUSES_69: List[str] = sorted(
    {b for line in IEEE69_ALL_BRANCHES for b in (line[0], line[1])}
    | set(IEEE69_LOAD_DATA.keys()),
    key=lambda b: int(b),
)

assert len(ALL_BUSES_69) == 69, f"Expected 69 buses, got {len(ALL_BUSES_69)}"
assert len(IEEE69_LINE_DATA) == 68, f"Expected 68 sectionalizing branches, got {len(IEEE69_LINE_DATA)}"

class IEEE69BusSimulator:
    """
    Generates randomised (load, topology) → (V, θ) scenarios for IEEE 69-bus.

    Advantages over 33-bus for this project:
      1. Larger graph (69 vs 33 nodes) -> richer zero-shot transfer test
      2. More varied R/X ratios across laterals -> harder kernel generalisation
      3. ~2× more training nodes per full graph -> better operator learning
    """

    BASEKV    = 12.66    # kV line-to-line (same as IEEE 33)
    BASE_KVA  = 1000.0   # 1 MVA base

    def __init__(self) -> None:
        self.sectionalizing = IEEE69_LINE_DATA
        self.tie_lines      = IEEE69_TIE_LINES
        self.all_branches   = IEEE69_ALL_BRANCHES
        self.load_data      = IEEE69_LOAD_DATA
        self.bus_set        = ALL_BUSES_69

    def sample_load_scenario(self, load_std_fraction: float = 0.15) -> Dict[str, float]:
        """Sample P, Q at every load bus with Gaussian variation."""
        rng = np.random.default_rng()
        scenario: Dict[str, float] = {}

        for bus, (p_base, q_base) in self.load_data.items():
            if p_base < 1e-6:
                scenario[f"P_{bus}"] = 0.0
                scenario[f"Q_{bus}"] = 0.0
                continue

            s_base  = math.hypot(p_base, q_base)
            pf      = p_base / s_base
            tan_phi = q_base / p_base  # preserve per-bus power factor

            p_rand  = float(max(0.1, rng.normal(p_base, load_std_fraction * p_base)))
            q_rand  = p_rand * tan_phi

            scenario[f"P_{bus}"] = p_rand
            scenario[f"Q_{bus}"] = q_rand

        return scenario

    def generate_radial_topology(self) -> nx.Graph:
        """
        Baran & Wu branch-exchange: close one tie, open one sectionalizing branch
        on the resulting loop to restore radiality.
        """
        G = nx.Graph()
        for fb, tb, r, x in self.sectionalizing:
            G.add_edge(fb, tb, r=r, x=x)

        tie = random.choice(self.tie_lines)
        fb_t, tb_t, r_t, x_t = tie
        G.add_edge(fb_t, tb_t, r=r_t, x=x_t)

        cycle = nx.find_cycle(G)

        sec_set = {(fb, tb) for fb, tb, *_ in self.sectionalizing}
        sec_set |= {(tb, fb) for fb, tb, *_ in self.sectionalizing}

        candidates = [(u, v) for (u, v, *_) in cycle if (u, v) in sec_set]
        u_open, v_open = random.choice(candidates)
        G.remove_edge(u_open, v_open)

        assert nx.is_tree(G), "Branch-exchange produced a non-tree graph."
        return G

    def topology_to_switch_vector(self, G: nx.Graph) -> np.ndarray:
        s = np.array(
            [1 if G.has_edge(fb, tb) or G.has_edge(tb, fb) else 0
             for fb, tb, *_ in self.all_branches],
            dtype=np.int32,
        )
        assert s.sum() == len(self.bus_set) - 1
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
            f"New Circuit.IEEE69 bus1=1 basekv={self.BASEKV} pu=1.05 phases=3"
        )
        for idx, (fb, tb, r, x) in enumerate(self.all_branches):
            name    = f"L{idx}"
            enabled = "Yes" if switch_vector[idx] == 1 else "No"
            dss.Text.Command(
                f"New Line.{name} bus1={fb} bus2={tb} length=1 units=mi "
                f"phases=3 r1={r} x1={x}"
            )
            dss.Text.Command(f"Edit Line.{name} Enabled={enabled}")

        for bus in self.bus_set:
            p_kw   = load_scenario.get(f"P_{bus}", 0.0)
            q_kvar = load_scenario.get(f"Q_{bus}", 0.0)
            if p_kw < 1e-6:
                continue
            dss.Text.Command(
                f"New Load.Load_{bus} bus1={bus} phases=3 kV={self.BASEKV} "
                f"kW={p_kw} kvar={q_kvar} model=1 conn=wye"
            )

        dss.Text.Command(f"Set voltagebases=[{self.BASEKV}]")
        dss.Text.Command("Calcvoltagebases")
        dss.Text.Command("Solve")

    def solve(
        self,
        load_scenario: Optional[Dict[str, float]] = None,
        switch_vector: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
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
            vm_ang = dss.Bus.puVmagAngle()
            results[f"V_{bus}"]     = float(vm_ang[0])
            results[f"Angle_{bus}"] = float(vm_ang[1])
            results[f"P_{bus}"]     = load_scenario.get(f"P_{bus}", 0.0)
            results[f"Q_{bus}"]     = load_scenario.get(f"Q_{bus}", 0.0)

        results["__switch_vector__"] = switch_vector.tolist()
        return results