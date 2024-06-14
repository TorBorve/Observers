# Observers
![Build test](https://github.com/TorBorve/Observers/actions/workflows/rust.yml/badge.svg)

Several state observers implemented in Rust.

## Available Observers

- Luenberger observers
- Kalman filter
- Steady state Kalman filter
- Extended kalman filter


## TODO
- [X] Implement Extended Kalman filter
- [ ] Implement Unscented Kalman filter
- [ ] Implement Square root kalman filter
- [ ] Implement Square root extended kalman filter
- [ ] Implement Particle filter
- [ ] Implement benchmarking for different filters
- [ ] Make Embedded systems friendly. I.e. no_std
- [ ] Make custom matrix/linalg library.
- [ ] Observers compatible with both static and dynamic sizes.
- [X] Automatic testing using Github Workflows