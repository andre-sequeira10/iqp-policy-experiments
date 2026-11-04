"""Shared device creation for QASP's circuit and algorithm modules."""

import pennylane as qml


def resolve_mpi(mpi=None, enable_mpi=None):
    """Resolve mpi and the legacy enable_mpi alias to one boolean.

    None means unspecified. If both values are supplied their truth values
    must agree, otherwise ValueError is raised; both omitted resolves False."""
    if mpi is not None and enable_mpi is not None and bool(mpi) != bool(enable_mpi):
        raise ValueError("mpi and enable_mpi must agree when both are supplied.")
    return bool(enable_mpi if mpi is None else mpi)


def create_device(device, *, wires, shots=None, mpi=False, device_kwargs=None, cutting=None):
    """Create or reuse a PennyLane device without silently discarding MPI requests.

    Args:
        device: Plugin name or an existing PennyLane Device.
        wires, shots: Passed when creating a new plugin device. An existing
            device is reused as-is; these arguments do not reconfigure it.
        mpi: Forward mpi=True to a new plugin, or require an existing device
            to report MPI enabled. Unsupported requests propagate plugin errors.
        device_kwargs: Additional plugin options, e.g. batch_obs or c_dtype.
            wires/shots/mpi must use the explicit arguments; options cannot
            reconfigure an existing device.
        cutting: Optional cut configuration. Return a metadata device for
            create_qnode; actual device allocation is deferred to fragments.

    Returns the device. No serial fallback is attempted. QASP's internal
    lightning.pp is available through the cutting metadata route or the
    legacy CircuitBuilder adapter, not through qml.device."""
    options = dict(device_kwargs or {})
    if cutting is not None:
        # A logical device carries execution settings; only fragment devices
        # allocate state during execution. In particular lightning.pp is not a
        # PennyLane plugin and must never be passed to qml.device directly.
        from qasp.cutting.specs import as_config
        logical_shots = device.shots if shots is None and isinstance(device, qml.devices.Device) else shots
        proxy = qml.device("default.qubit", wires=wires, shots=logical_shots)
        proxy._qasp_cutting = as_config(cutting)
        proxy._qasp_fragment_device = device
        proxy._qasp_fragment_options = {"device_kwargs": options, "mpi": mpi}
        return proxy
    reserved = {"wires", "shots", "mpi"}.intersection(options)
    if reserved:
        raise ValueError(
            f"Use the explicit arguments for {sorted(reserved)}, not device_kwargs."
        )
    if isinstance(device, qml.devices.Device):
        if options:
            raise ValueError("device_kwargs cannot configure an existing device.")
        if mpi and not getattr(device, "_mpi", False):
            raise ValueError(
                "mpi=True requires an existing device constructed with mpi=True; "
                "alternatively pass its device name to QASP."
            )
        return device
    if mpi:
        options["mpi"] = True
    return qml.device(device, wires=wires, shots=shots, **options)


def create_qnode(func, device, *, cutting=None, **kwargs):
    """Create a regular or cutting QNode, preserving logical device settings."""
    config = cutting if cutting is not None else getattr(device, "_qasp_cutting", None)
    if config is None:
        return qml.QNode(func, device, **kwargs)
    from qasp.cutting import CuttingQNode
    options = dict(getattr(device, "_qasp_fragment_options", {}))
    options.update(kwargs)
    options.setdefault("shots", device.shots)
    return CuttingQNode(
        func, cutting=config,
        device=getattr(device, "_qasp_fragment_device", device), **options,
    )
