# Coalescence of non-Newtonian drops

Direct numerical simulation of the coalescence of viscoelastic drops, developed
as a controlled physical proxy for the fusion of biomolecular condensates.

## Why this repository exists

Condensate material properties are routinely inferred from a fusion movie. Two
droplets are brought into contact, the neck is tracked, and the relaxation is
fitted to a viscocapillary law to return an inverse capillary velocity. That
inversion assumes a Newtonian drop with time-independent properties. Condensates
are neither. They are viscoelastic, they age, and they sit in a dilute phase that
is itself not a passive gas.

The scientific target is therefore not another neck-growth law. It is what a
fusion measurement can and cannot determine once the drop has memory, once that
memory evolves during the event, and once the exterior phase carries stress of
its own.

## Scope

Planned, in order.

1. Reproduce the established Newtonian coalescence regimes against published
   high-resolution benchmarks, and state the resolution at which the initial
   Stokes regime is recovered.
2. Map neck growth across Deborah number, including the strongly elastic limit,
   rather than a small set of relaxation times at fixed material properties.
3. Add a viscoelastic exterior phase, the configuration relevant to a condensate
   in a crowded dilute phase.
4. Allow the constitutive state to evolve during the event, which is the
   mechanical content of ageing.
5. Treat the inverse problem explicitly. Given a synthetic neck trace degraded to
   realistic imaging resolution and noise, establish which material parameters
   are recoverable and which are not.

## Status

Scaffold. No solver, no cases and no results yet. Background, prior art and the
detailed problem statement are held in the project context repository and are not
mirrored here.
