# Exported controller

`reach_policy.c` is the current best network as one C file. It needs only `<math.h>`, allocates
nothing, and runs one control tick per call. It was trained in simulation only; see the
[project README](../README.md) for what it assumes about the arm and the calling convention,
and [docs/REACH.md](../docs/REACH.md) for how it was trained and evaluated.

Compile check:

```sh
cc -O2 -c reach_policy.c -o reach_policy.o
```

Regenerate from a training run with `python -m robot_ai.control.c_export --run <run dir> --output reach_policy.c`.
