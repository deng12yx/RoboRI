# Calibrated RI-Mem heads

Provide one calibrated head per backbone:

- `framesamp-modul.npz`
- `tokendrop-modul.npz`
- `simplememvla.npz`

Generate these with `robomme-ri calibrate`. The loader validates backbone identity,
parameter dimensions and calibration metadata. Initialization-only heads are
rejected for enhanced evaluation. Model weights are not distributed here.
