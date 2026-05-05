###################################################################################
# Millimeter Wave Demo
###################################################################################

.PHONY: dssDemo dssDemoClean dssDemoObjClean dssDemoEnet

###################################################################################
# Setup the VPATH:
###################################################################################
VPATH_DDM = $(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpc/objectdetection/objdethwaDDMA/src
VPATH_COM = $(MMWAVE_SDK_INSTALL_PATH)/ti/demo/utils \
		  	./dss

DSS_CPU := C66
DSS_CPU_INSTANCE := c66

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
DSS_MMW_DEMO_STD_LIBS = $($(DSS_CPU)_COMMON_STD_LIB) \
						-llibmathutils.$($(DSS_CPU)_LIB_EXT) \
						-lmathlib.$($(DSS_CPU)_LIB_EXT) \
						-llibdpedma_hwa_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT) \
						-ldsplib.$($(DSS_CPU)_LIB_EXT) \

DSS_MMW_DEMO_LOC_LIBS = $($(DSS_CPU)_COMMON_LOC_LIB) \
						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib \
						-i$($(DSS_CPU)x_MATHLIB_INSTALL_PATH)/packages/ti/mathlib/lib \
						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpedma/lib \
						-i$($(DSS_CPU)x_DSPLIB_INSTALL_PATH)/packages/ti/dsplib/lib \

DSS_MMW_DEMO_DDM_LIBS = $(DSS_MMW_DEMO_STD_LIBS) \

DSS_MMW_DEMO_DDM_LOC_LIBS = $(DSS_MMW_DEMO_LOC_LIBS) \

###################################################################################
# Millimeter Wave Demo
###################################################################################
DSS_MMW_CFG_PREFIX       = mmw_dss
DSS_MMW_DEMO_MAP         = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN).map
DSS_MMW_DEMO_OUT         = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN).$($(DSS_CPU)_EXE_EXT)
DSS_MMW_DEMO_RPRC        = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN).rprc
DSS_MMW_DEMO_CMD         = dss/mmw_dss_linker.cmd

DSS_MMW_ENET_DEMO_MAP    = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN)Enet.map
DSS_MMW_ENET_DEMO_OUT    = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN)Enet.$($(DSS_CPU)_EXE_EXT)
DSS_MMW_ENET_DEMO_RPRC   = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_dss$(PROC_CHAIN)Enet.rprc
DSS_MMW_ENET_DEMO_CMD    = dss/mmw_dss_linker_enet.cmd

DSS_MMW_DEMO_DDM_SOURCES = dss_main.c \

ifeq ($(MSS_AOA_ENABLED), 0)
DSS_MMW_DEMO_DDM_SOURCES += objectdetection_elevEst.c
endif

DSS_MMW_DEMO_SOURCES_GEN  = ti_board_config.c	\
							ti_board_open_close.c	\
							ti_dpl_config.c	\
							ti_drivers_config.c	\
							ti_pinmux_config.c	\
							ti_power_clock_config.c	\
							ti_drivers_open_close.c

DSS_MMW_DDM_DEMO_DEPENDS     = $(addprefix $(PLATFORM_OBJDIR)/, $(DSS_MMW_DEMO_DDM_SOURCES:.c=.$($(DSS_CPU)_DEP_EXT)))
DSS_MMW_DDM_DEMO_OBJECTS     = $(addprefix $(PLATFORM_OBJDIR)/, $(DSS_MMW_DEMO_DDM_SOURCES:.c=.$($(DSS_CPU)_OBJ_EXT)))
DSS_MMW_DEMO_OBJECTS_GEN = $(addprefix $(PLATFORM_OBJDIR)/dssgenerated/, $(DSS_MMW_DEMO_SOURCES_GEN:.c=.$($(DSS_CPU)_OBJ_EXT)))

###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
dssDemo: $(DSS_CPU)_CFLAGS += -i$($(DSS_CPU)x_MATHLIB_INSTALL_PATH)/packages \
                        	--define=APP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
                        	--define=DebugP_LOG_ENABLED \
                            --define=MSS_AOA_ENABLED=$(MSS_AOA_ENABLED)

		VPATH=$(VPATH_$(PROC_CHAIN)):$(VPATH_COM)

dssDemo: buildDirectories dssbuildDirectories $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN)
		$($(DSS_CPU)_LD) $($(DSS_CPU)_LDFLAGS) $(DSS_MMW_DEMO_$(PROC_CHAIN)_LOC_LIBS) $(DSS_MMW_DEMO_$(PROC_CHAIN)_LIBS) \
		--map_file=$(DSS_MMW_DEMO_MAP) $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN) \
		$(DSS_MMW_DEMO_CMD) -o $(DSS_MMW_DEMO_OUT)
		@echo "******************************************************************************"
		@echo 'Built the DSS for Millimeter Wave Demo'
		@echo "******************************************************************************"

###################################################################################
# Build Ethernet stream enabled Millimeter Wave Demo
###################################################################################
dssDemoEnet: $(DSS_CPU)_CFLAGS += -i$($(DSS_CPU)x_MATHLIB_INSTALL_PATH)/packages \
                        	--define=APP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
                        	--define=DebugP_LOG_ENABLED \
							-DENET_STREAM \
                            --define=MSS_AOA_ENABLED=$(MSS_AOA_ENABLED)

		VPATH=$(VPATH_$(PROC_CHAIN)):$(VPATH_COM)

dssDemoEnet: buildDirectories dssbuildDirectories $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN)
		$($(DSS_CPU)_LD) $($(DSS_CPU)_LDFLAGS) $(DSS_MMW_DEMO_$(PROC_CHAIN)_LOC_LIBS) $(DSS_MMW_DEMO_$(PROC_CHAIN)_LIBS) \
		--map_file=$(DSS_MMW_ENET_DEMO_MAP) $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN) \
		$(DSS_MMW_ENET_DEMO_CMD) -o $(DSS_MMW_ENET_DEMO_OUT)
		@echo "******************************************************************************"
		@echo 'Built the DSS for Millimeter Wave Demo'
		@echo "******************************************************************************"

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
dssDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo DSS Objects'
	@rm -f $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN)
	@rm -f $(DSS_MMW_DEMO_OUT) $(DSS_MMW_DEMO_MAP) $(DSS_MMW_$(PROC_CHAIN)_DEMO_DEPENDS)
	@rm -f $(DSS_MMW_ENET_DEMO_MAP) $(DSS_MMW_ENET_DEMO_OUT)
	@$(DEL) $(PLATFORM_OBJDIR)

dssDemoObjClean:
	@echo 'Cleaning the Millimeter Wave Demo DSS Objects'
	@rm -f $(DSS_MMW_$(PROC_CHAIN)_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN)
	@rm -f $(DSS_MMW_$(PROC_CHAIN)_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(DSS_MMW_DDM_DEMO_DEPENDS)
