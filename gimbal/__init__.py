"""Pan/tilt gimbal module (Ticket 4).

Controls a Pololu Micro Maestro 6-channel USB servo controller
driving two hobby 25 kg servos (pan + tilt). Exposes:

    MaestroDriver     — thin USB/serial wrapper (compact protocol)
    GimbalController  — angle ↔ µs, software limits, slew-rate limiter
    GimbalManager     — background thread, auto-tracks fused target
                        when user presses TRACK, manual otherwise

Graceful degradation: if no Maestro is found, GimbalManager still
runs and publishes ``connected=False`` state so the GUI can grey
out the dpad without crashing the rest of the app.
"""
