# clang-tidy workflow

*MCP-owned — canonical source for the mandatory clang-tidy static-analysis step.*

## Requirement

**clang-tidy on the changed C/C++ files is a required step before committing any
C++ change** in the C++ repos (`pypto`, `PTOAS`, `pto-isa`, `simpler`,
`pypto-lib`). `verify_ladder` returns `static_checks: ["clang-tidy"]` whenever a
changed path is a C/C++ file in one of those repos — treat that as part of the
minimal verify set, alongside the suggested pytest tasks. Python-only changes
do **not** need it (e.g. the ergonomic `pld.*` collective wrappers are pure
Python — no clang-tidy).

## When it applies

Any changed `*.c` / `*.cc` / `*.cpp` / `*.cxx` / `*.h` / `*.hh` / `*.hpp` /
`*.hxx` / `*.inl` file under `pypto/`, `PTOAS/`, `pto-isa/`, `simpler/`, or
`pypto-lib/`.

## Prerequisite: a compilation database

clang-tidy needs `compile_commands.json` (a compile db) to resolve includes and
flags. Generate it from a configured CMake build:

```bash
# Configure once with the compile-db exporter enabled
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$(nproc)     # explicit -jN — see the golden rule below
```

If the build dir already exists without the exporter, reconfigure:
`cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Some repos keep
`compile_commands.json` at the repo root — symlink it from `build/` if needed.

> **Golden rule (shared with the sim-Docker workflow):** never run
> `cmake --build build --parallel` (unbounded `-j` saturates the host). Always
> use an explicit `-jN`.

## Run on changed files only

```bash
# From the repo root; scope to the branch's changed C/C++ files
git diff --name-only origin/main...HEAD \
  | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx|inl)$' \
  | xargs -r -n1 clang-tidy -p build
```

Or run the whole tree (slow; CI-style):

```bash
run-clang-tidy -p build -j$(nproc)
```

## Checks / configuration

Repos may ship a `.clang-tidy` at the root (otherwise LLVM defaults + project
rules). Fix findings in the same commit as the code — never suppress without a
documented reason.

## Relation to other gates

- **pre-commit** runs `clang-format` / `cpplint` but **not** clang-tidy (no
  build). clang-tidy is complementary and needs the compile db, so it is a
  separate, mandatory step for C++ work.
- **verify_ladder** surfaces it as a `static_check` for C++ changes;
  **gate_pr_script** lists it in `agent_instructions` when the branch touches
  C++ files.
- clang-tidy is a static check needing only the compile db — the NPU-or-sim
  routing policy does **not** apply to it.

## Per-repo notes

| Repo | Notes |
|------|-------|
| `pypto` | Compiler + codegen C++ under `src/`, `include/`; compile db from `build/` |
| `PTOAS` | MLIR/LLVM-based; `build/` via `ninja` — the LLVM configure exports the compile db |
| `pto-isa` | Header-heavy tile library under `include/pto/` |
| `simpler` | Runtime C++ under `src/{arch}/runtime/`; built via `pip install -e .` (scikit-build-core) — export the compile db via `SKBUILD_CMAKE_ARGS=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` if needed |
| `pypto-lib` | Mostly Python; C++ only in extern/fused kernels — clang-tidy only if a `.cc`/`.h` changed |
