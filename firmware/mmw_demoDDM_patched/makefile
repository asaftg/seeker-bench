###################################################################################
# mmWave Demo Level makefile
###################################################################################
ifeq ($(CCS_MAKEFILE_BASED_BUILD), 1)
ifeq ($(OS),Windows_NT)
include $(MMWAVE_SDK_INSTALL_PATH)/scripts/windows/setenv.mak
else
include $(MMWAVE_SDK_INSTALL_PATH)/scripts/unix/setenv.mak
endif
endif

M4_RELEASE_OPT ?= 1

include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak
include ./makefile_awr2x44P.mak
# SEEKER FORK: dropped ECO + LC variant includes -- bench is awr2x44P only.

BOOTIMAGE_TEMP_OUT_FILE  = temp_stdout.txt
OUTRPRC_CMD = $(SYSCFG_NODE) $(MCU_PLUS_INSTALL_PATH)/tools/boot/out2rprc/elf2rprc.js
MULTI_CORE_IMAGE_GEN = $(SYSCFG_NODE) $(MCU_PLUS_INSTALL_PATH)/tools/boot/multicoreImageGen/multicoreImageGen.js

# Include OOB Demo Path
R5F_INCLUDE     += -I$(MMWAVE_SDK_INSTALL_PATH)/ti/demo/awr2x44P/mmw_ddm/mss/mssgenerated \

C66_COMMON_INCLUDE    += -I$(MMWAVE_SDK_INSTALL_PATH)/ti/demo/awr2x44P/mmw_ddm/dss/dssgenerated \

M4_COMMON_INCLUDE     += -I$(MMWAVE_SDK_INSTALL_PATH)/ti/demo/awr2x44P/mmw_ddm/dss_cm4/m4generated \

###################################################################################
# Standard Targets which need to be implemented by each mmWave SDK module. This
# plugs into the release scripts.
###################################################################################
.PHONY: all clean
.NOTPARALLEL:

# SEEKER FORK: only awr2x44P + UART/LVDS path. Drop Enet variant + ECO/LC.
all: mmwDemoDDM

# Clean: This cleans all the objects
clean: mmwDemoDDMClean

# Help: This displays the MAKEFILE Usage.
help:
	@echo "****************************************************************************************"
	@echo '* Makefile Targets for the DEMO '
	@echo 'all             -> Builds DDM mmw Demo and appimage'
	@echo 'clean           -> Cleans DDM mmw Demo and appimage'
	@echo 'syscfg          -> Generates the Syscfg files (MSS, DSS_CM4 and DSS binaries)'
	@echo 'mmwDemoDDM      -> Builds the DDM mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44P'
	@echo 'mmwDemoDDMClean -> Cleans the DDM mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44P'
	@echo 'mmwEcoDemoDDM      -> Builds the DDM mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44ECO'
	@echo 'mmwEcoDemoDDMClean -> Cleans the DDM mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44ECO'
	@echo 'mmwLcDemoDDM      -> Builds the DDM mmw Demo (MSS and DSS_CM4 binaries) and appimage for awr2x44LC'
	@echo 'mmwLcDemoDDMClean -> Cleans the DDM mmw Demo (MSS and DSS_CM4 binaries) and appimage for awr2x44LC'
	@echo 'syscfg-gui      -> opens syscfg gui for DDM (MSS, DSS_CM4 and DSS binaries) demo application'
	@echo 'syscfg_enet     -> Generates the Enet Syscfg files (MSS, DSS_CM4 and DSS binaries)'
	@echo 'mmwDemoDDMEnet  -> Builds the DDM Enet mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44P'
	@echo 'mmwDemoDDMEnetClean -> Cleans the DDM Enet mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44P'
	@echo 'mmwEcoDemoDDMEnet  -> Builds the DDM Enet mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44ECO'
	@echo 'mmwEcoDemoDDMEnetClean -> Cleans the DDM Enet mmw Demo (MSS, DSS_CM4 and DSS binaries) and appimage for awr2x44ECO'
	@echo 'enetsyscfg-gui  -> opens syscfg gui for DDM Enet (MSS, DSS_CM4 and DSS binaries) demo application'
	@echo "****************************************************************************************"
