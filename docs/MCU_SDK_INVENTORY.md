# MCU+SDK Inventory for AWR2944P firmware build

Host: Windows. Inventory date 2026-05-02. All paths verified by listing.

## Versions

| Tool | Version | Path |
|---|---|---|
| MCU+SDK AWR2X44P | 10.02.00 (10_02_00_04) | `C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04` |
| ti-arm-clang (TI ARM CLANG) | 4.0.2.LTS | `C:\ti\ti-cgt-armllvm_4.0.2.LTS\bin\tiarmclang.exe` |
| sysconfig | 1.23.0.4000 | `C:\ti\sysconfig_1.23.0` |
| gmake (GNU Make Win32) | 4.4.1 | `C:\ti\make-portable\bin\gmake.exe` |

Version evidence:
- SDK: `.metadata\product.json` -> `"version": "10.02.00"`.
- tiarmclang `--version` -> `TI Arm Clang Compiler 4.0.2.LTS, Target: arm-ti-none-eabi`.
- sysconfig: install log `sysconfig_setup_1.23.0.4000_install.log`.
- gmake `--version` -> `GNU Make 4.4.1, Built for Windows32`.
- SDK 10.02 release notes (`docs\api_guide_awr2x44p\RELEASE_NOTES_10_02_00_PAGE.html`) call out TI ARM CLANG 4.0.2 -> exact match.

## SDK structure verified

- `C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\imports.mak` present.
- `C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\makefile.awr2x44p` present (top-level per-device makefile).
- Example sbl_qspi tree at `examples\drivers\boot\sbl_qspi\awr2x44p-evm\r5fss0-0_nortos\ti-arm-clang\` already contains prebuilt `sbl_qspi.release.tiimage` and `sbl_qspi.debug.tiimage`. Indicates TI shipped pre-built reference binaries; clean rebuild still required for our flow.

`imports.mak` defaults assume CCS at `C:\ti\ccs2010\ccs` and resolves `CGT_TI_ARM_CLANG_PATH` to the standalone install if CCS path is absent (`ifeq ($(wildcard $(CGT_TI_ARM_CLANG_PATH)),)` fallback to `$(TOOLS_PATH)/ti-cgt-armllvm_4.0.2.LTS`). CCS is NOT installed on this host; standalone fallback is satisfied.

## Build invocation

Canonical command for sbl_qspi release build:

```
SET PATH=C:\ti\make-portable\bin;%PATH%
gmake -s -C C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\examples\drivers\boot\sbl_qspi\awr2x44p-evm\r5fss0-0_nortos\ti-arm-clang all PROFILE=release
```

Required env (only needed if overriding imports.mak defaults; current install matches defaults):

```
SET TOOLS_PATH=C:/ti
SET CG_TOOL_ROOT=C:/ti/ti-cgt-armllvm_4.0.2.LTS
SET SYSCFG_PATH=C:/ti/sysconfig_1.23.0
SET MCU_PLUS_SDK_PATH=C:/ti/mcu_plus_sdk_awr2x44p_10_02_00_04
```

Output (per makefile, `BOOTIMAGE_NAME_GP`):

```
C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\examples\drivers\boot\sbl_qspi\awr2x44p-evm\r5fss0-0_nortos\ti-arm-clang\sbl_qspi.release.tiimage
```

## Blockers

1. CCS not installed (`C:\ti\ccs2010\ccs` missing). Not blocking: `imports.mak` falls back to `C:\ti\ti-cgt-armllvm_4.0.2.LTS`, which is present. Cygwin utilities (`mkdir`, `rm`, `cp`, `sed`, `cat`, `touch`) are referenced from `$(CCS_PATH)\utils\cygwin` and will be missing -- builds may fail on first `MKDIR`/`RMDIR`/`SED` invocation. Mitigation: install Code Composer Studio (CCS) or symlink Git Bash binaries; recommended path is the official CCS bundle: https://www.ti.com/tool/CCSTUDIO -> "CCS 20.x for Windows". CCS bundles compatible cygwin and a known-good gmake under `ccs\utils\bin\`.
2. `gmake` is on disk but not on PATH by default. Prepend `C:\ti\make-portable\bin` before invoking, or point env `MAKE` at it.
3. `python` required by makefile (for image signing/cert tooling). Verify a `python` on PATH before first build (Python 3.11/3.13 already installed per Start menu inventory).
4. No mismatch in compiler/sysconfig versions vs SDK 10.02 expectations.

## Verified callable

- `C:\ti\make-portable\bin\gmake.exe --version` -> `GNU Make 4.4.1`.
- `C:\ti\ti-cgt-armllvm_4.0.2.LTS\bin\tiarmclang.exe --version` -> `TI Arm Clang Compiler 4.0.2.LTS`.
