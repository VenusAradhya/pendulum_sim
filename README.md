

<!-- AUTO_RESULTS_START -->
## Latest Auto-Generated Run Summary

### RL (latest run)
- Seed: `29886`
- Passive RMS x2: `0.000 mm`
- RL RMS x2: `0.005 mm`
- Improvement factor (passive/RL): `0.00x`
- Reward initial/final: `-111.0867 -> -222.2289`
- No-noise regulation final |x2|: `11.019 mm`
- Interpretation: If improvement is < 1.0x, the policy is still underperforming passive isolation and reward scaling/actuation strategy should be revisited.

### Simple controls / LQR (latest run)
- Seed: `21302`
- Passive RMS x2: `0.000 mm`
- LQR RMS x2: `0.000 mm`
- Improvement factor (passive/LQR): `3.51x`
- Interpretation: This is your near-equilibrium model-based baseline; RL should eventually match or exceed this over repeated seeds.

### Unified evaluation modes (same seed)
- Seed: `29886`
- RL-only RMS x2: `0.005 mm`
- LQR-only RMS x2: `0.000 mm`
- Cascade RMS x2: `0.001 mm`
- Bad-LQR RMS x2: `0.000 mm`
- Bad-Cascade RMS x2: `0.001 mm`
- Cascade alpha: `1.00`
- Bad-LQR scale: `0.35`

### How to read the plots
- **Time-domain x2 plot**: smaller oscillation envelope means better isolation of the bottom mirror displacement.
- **ASD plot**: each point is displacement amplitude per √Hz at that frequency; lower curve means less motion/noise coupling at that band.
- **Controller comparison bars**: direct RMS comparison for RL-only, LQR-only, cascade, and bad-LQR stress tests using the same seed.

### Physics notes for LIGO context
- Lower RMS and lower ASD in the microseismic band imply better suspension isolation and reduced motion coupling into interferometer sensing.
- A strong learning curve without RMS/ASD gain usually means the cost function is being optimized in a way that is not physically aligned with disturbance rejection.
<!-- AUTO_RESULTS_END -->
