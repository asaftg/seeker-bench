###################################################################################
# Millimeter Wave Demo
###################################################################################

.PHONY: dssCM4EcoDemo dssCM4EcoDemoClean dssCM4DemoObjClean dssCM4EcoDemoEnet

###################################################################################
# Setup the VPATH:
###################################################################################

VPATH_DDM = $(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpc/objectdetection/objdethwaDDMA/src \

vpath %.c $(VPATH_$(PROC_CHAIN)) \
		 $(MMWAVE_SDK_INSTALL_PATH)/ti/demo/utils \
		 ./dss_cm4

M4_CPU := M4
M4_CPU_INSTANCE := m4

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
M4_MMW_ECO_DEMO_STD_LIBS = $($(M4_CPU)_COMMON_OPT_LIB) \
						   -llibmathutils.$($(M4_CPU)_LIB_EXT) \
						   -llibdpedma_hwa_$(MMWAVE_SDK_DEVICE_TYPE).$($(M4_CPU)_LIB_EXT) \

M4_MMW_ECO_DEMO_LOC_LIBS = $($(M4_CPU)_COMMON_LOC_LIB) \
						   -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib \
						   -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpedma/lib \

M4_MMW_ECO_DEMO_DDM_DP_LIBS = -llibrangeproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(M4_CPU)_LIB_EXT) \
							  -llibdopplerproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(M4_CPU)_LIB_EXT) \
							  -llibrangecfarproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(M4_CPU)_LIB_EXT) \

M4_MMW_ECO_DEMO_DDM_DP_LOC_LIBS = -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpu/rangeprocDDMA/lib \
								  -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpu/dopplerprocDDMA/lib \
								  -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpu/rangecfarprocDDMA/lib \

M4_MMW_ECO_DEMO_DDM_LIBS = $(M4_MMW_ECO_DEMO_STD_LIBS) \
						   $(M4_MMW_ECO_DEMO_DDM_DP_LIBS) \

M4_MMW_ECO_DEMO_DDM_LOC_LIBS = $(M4_MMW_ECO_DEMO_LOC_LIBS) \
							   $(M4_MMW_ECO_DEMO_DDM_DP_LOC_LIBS) \

###################################################################################
# Millimeter Wave Demo
###################################################################################
M4_MMW_ECO_DEMO_MAP     = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN).map
M4_MMW_ECO_DEMO_OUT     = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN).$($(M4_CPU)_EXE_EXT)
M4_MMW_ECO_DEMO_RPRC    = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN).rprc
M4_MMW_ECO_DEMO_CMD     = dss_cm4/mmw_dss_cm4_linker.cmd
M4_ECO_LINK_CMD         = $(MMWAVE_SDK_INSTALL_PATH)/ti/platform/awr2x44x/awr2x44ECO/m4_linker.cmd

M4_MMW_ECO_ENET_DEMO_MAP    = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN)Enet.map
M4_MMW_ECO_ENET_DEMO_OUT    = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN)Enet.$($(M4_CPU)_EXE_EXT)
M4_MMW_ECO_ENET_DEMO_RPRC   = awr2x44ECO_mmw_demo_dss_cm4$(PROC_CHAIN)Enet.rprc
M4_MMW_ECO_ENET_DEMO_CMD    = $(MMWAVE_SDK_INSTALL_PATH)/ti/platform/awr2x44x/awr2x44ECO/m4_linker_enet.cmd

M4_MMW_ECO_DEMO_DDM_SOURCES = dss_cm4_main.c \
							  objectdetection.c \

M4_MMW_ECO_DEMO_SOURCES_GEN  = ti_board_config.c	\
							   ti_board_open_close.c	\
							   ti_dpl_config.c	\
							   ti_drivers_config.c	\
							   ti_pinmux_config.c	\
							   ti_power_clock_config.c	\
							   ti_drivers_open_close.c

M4_MMW_ECO_DDM_DEMO_DEPENDS = $(addprefix $(PLATFORM_OBJDIR)/, $(M4_MMW_ECO_DEMO_DDM_SOURCES:.c=.$($(M4_CPU)_DEP_EXT)))
M4_MMW_ECO_DDM_DEMO_OBJECTS = $(addprefix $(PLATFORM_OBJDIR)/, $(M4_MMW_ECO_DEMO_DDM_SOURCES:.c=.$($(M4_CPU)_OBJ_EXT)))
M4_MMW_ECO_DEMO_OBJECTS_GEN = $(addprefix $(PLATFORM_OBJDIR)/m4generated/, $(M4_MMW_ECO_DEMO_SOURCES_GEN:.c=.$($(M4_CPU)_OBJ_EXT)))


###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
dssCM4EcoDemo: $(M4_CPU)_CFLAGS += -DAPP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
							-DMMWDEMO_$(PROC_CHAIN) -DDebugP_ASSERT_ENABLED=0 -DDebugP_LOG_ENABLED=0 \
							-DSOC_AWR2x44ECO \
                            -DMSS_AOA_ENABLED=$(ECO_MSS_AOA_ENABLED)
dssCM4EcoDemo: $(M4_CPU)_LDFLAGS += -Wl,--define=MSS_AOA_ENABLED=$(ECO_MSS_AOA_ENABLED)

dssCM4EcoDemo: buildDirectories m4buildDirectories $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN)
		$($(M4_CPU)_LD) $($(M4_CPU)_LDFLAGS) $(M4_MMW_ECO_DEMO_$(PROC_CHAIN)_LOC_LIBS) -Wl,-m=$(M4_MMW_ECO_DEMO_MAP) \
		-o $(M4_MMW_ECO_DEMO_OUT) $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN) \
		$(M4_MMW_ECO_DEMO_$(PROC_CHAIN)_LIBS) $(M4_ECO_LINK_CMD) $(M4_MMW_ECO_DEMO_CMD)
		@echo "******************************************************************************"
		@echo 'Built the M4 for Millimeter Wave Demo'
		@echo "******************************************************************************"

###################################################################################
# Build Ethernet stream enabled Millimeter Wave Demo
###################################################################################
dssCM4EcoDemoEnet: $(M4_CPU)_CFLAGS += -DAPP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
							-DMMWDEMO_$(PROC_CHAIN) \
							-DENET_STREAM -DDebugP_ASSERT_ENABLED=0 -DDebugP_LOG_ENABLED=0 \
							-DSOC_AWR2x44ECO \
                            -DMSS_AOA_ENABLED=$(ECO_MSS_AOA_ENABLED)
dssCM4EcoDemoEnet: $(M4_CPU)_LDFLAGS += -Wl,--define=MSS_AOA_ENABLED=$(ECO_MSS_AOA_ENABLED) \
							-Wl,--define=ENET_STREAM

dssCM4EcoDemoEnet: buildDirectories m4buildDirectories $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN)
		$($(M4_CPU)_LD) $($(M4_CPU)_LDFLAGS) $(M4_MMW_ECO_DEMO_$(PROC_CHAIN)_LOC_LIBS) -Wl,-m=$(M4_MMW_ECO_ENET_DEMO_MAP) \
		-o $(M4_MMW_ECO_ENET_DEMO_OUT) $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN) \
		$(M4_MMW_ECO_DEMO_$(PROC_CHAIN)_LIBS) $(M4_MMW_ECO_ENET_DEMO_CMD) $(M4_MMW_ECO_DEMO_CMD)
		@echo "******************************************************************************"
		@echo 'Built the M4 for Millimeter Wave Demo for Enet'
		@echo "******************************************************************************"

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
dssCM4EcoDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo M4 Objects'
	@rm -f $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN)
	@rm -f $(M4_MMW_ECO_DEMO_MAP) $(M4_MMW_ECO_ENET_DEMO_OUT) $(M4_MMW_$(PROC_CHAIN)_DEMO_DEPENDS)
	@rm -f $(M4_MMW_ECO_ENET_DEMO_MAP) $(M4_MMW_ECO_ENET_DEMO_OUT) $(M4_MMW_ECO_DEMO_OUT)
	@$(DEL) $(PLATFORM_OBJDIR)

dssCM4EcoDemoObjClean:
	@echo 'Cleaning the Millimeter Wave Demo M4 Objects'
	@rm -f $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_OBJECTS) $(M4_MMW_ECO_DEMO_OBJECTS_GEN)
	@rm -f $(M4_MMW_ECO_$(PROC_CHAIN)_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(M4_MMW_ECO_DDM_DEMO_DEPENDS)
