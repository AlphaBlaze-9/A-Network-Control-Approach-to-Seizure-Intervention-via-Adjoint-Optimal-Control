# Optimization Results (Week 11)

## Core data package
We package the structural adjacency matrix $A$, node excitability weights ($I_{ext}$), actuator matrix $B$, node coordinates, and simulation metadata into a single reproducible repository artifact (`week11_core_data.npz`) plus CSV exports.

## Optimization results: stimulation targets selected by the CBF-QP
At each time step, the controller solves a minimum-energy quadratic program to keep voltages below the safety threshold. The resulting control signal $u(t)$ is analyzed per node using an energy proxy $E_i = \sum_t u_i(t)^2$.

- Nodes: **50**
- Steps: **8000** (dt = **0.05 s**, total T ≈ **400.00 s**)
- Safety threshold: **1.000**
- Focus node (seizure onset in this simulation): **0**

### Top stimulated nodes (ranked by $\sum_t u_i(t)^2$)
| Rank | Node | Energy $E_i$ | Mean $|u_i|$ | Peak $|u_i|$ | Active fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 6867.427 | 0.5185 | 2.5073 | 0.346 |
| 2 | 25 | 716.551 | 0.1247 | 1.1542 | 0.236 |
| 3 | 26 | 713.027 | 0.1242 | 1.1604 | 0.235 |
| 4 | 24 | 711.308 | 0.1241 | 1.1634 | 0.235 |
| 5 | 21 | 709.282 | 0.1237 | 1.1688 | 0.234 |
| 6 | 23 | 706.110 | 0.1236 | 1.1607 | 0.235 |
| 7 | 27 | 705.036 | 0.1232 | 1.1586 | 0.234 |
| 8 | 28 | 701.198 | 0.1228 | 1.1642 | 0.234 |
| 9 | 20 | 697.793 | 0.1226 | 1.1499 | 0.233 |
| 10 | 12 | 697.548 | 0.1225 | 1.1475 | 0.234 |

## Baseline comparison (Controllability Gramian hubs)
We compute a standard linear controllability Gramian ranking (Week 8 baseline) and compare it to the closed-loop CBF controller from Weeks 5–7.

- Baseline hub nodes (top 3): **[2, 25, 42]**
- Energy (CBF): **37867.77**
- Energy (Baseline constant suppression): **96000.00** (mag=2.0)
- Efficiency ratio (Baseline / CBF): **2.54×**
- Percent energy saved vs baseline: **60.6%**
- Overlap between CBF top-K stimulated nodes and baseline hubs: **1**
