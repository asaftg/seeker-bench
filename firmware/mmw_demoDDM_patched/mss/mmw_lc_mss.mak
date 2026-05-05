###################################################################################
# Millimeter Wave Demo
###################################################################################
.PHONY: mssLcDemo mssLcDemoClean mssLcDemoObjClean
###################################################################################
# Setup the VPATH:
###################################################################################
vpath %.c $(MMWAVE_SDK_INSTALL_PATH)/ti/demo/utils \
		  $(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpc/objectdetection/objdethwaDDMA/src \
          ./mss

MSS_CPU := R5F
MSS_CPU_INSTANCE := r5f

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
MSS_MMW_LC_DEMO_STD_LIBS = $($(MSS_CPU)_COMMON_STD_LIB) \
                            -lmmwavelink_$(MSS_CPU_INSTANCE).lib \
                            -llibmmwave_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
                            -llibmathutils.$($(MSS_CPU)_LIB_EXT) \
                            -llibcli_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
                            -llibhsiheader_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT)

MSS_MMW_LC_DEMO_LOC_LIBS = $($(MSS_CPU)_COMMON_LOC_LIB)  \
							-Wl,-i$(MMWAVE_AWR294X_DFP_INSTALL_PATH)/ti/control/mmwavelink/lib	\
							-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/mmwave/lib  \
							-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/cli/lib   \
							-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib \
							-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/hsiheader/lib

###################################################################################
# Millimeter Wave Demo
###################################################################################
R5F_LC_LINK_CMD         = $(MMWAVE_SDK_INSTALL_PATH)/ti/platform/awr2x44x/awr2x44LC/r5f_linker.cmd
MSS_MMW_LC_DEMO_MAP     = awr2x44LC_mmw_demo_mss$(PROC_CHAIN).map
MSS_MMW_LC_DEMO_OUT     = awr2x44LC_mmw_demo_mss$(PROC_CHAIN).$($(MSS_CPU)_EXE_EXT)
MSS_MMW_LC_DEMO_RPRC    = awr2x44LC_mmw_demo_mss$(PROC_CHAIN).rprc
MSS_MMW_LC_DEMO_CMD     = mss/mmw_mss_linker.cmd
MSS_MMW_LC_DEMO_SOURCES = mmwdemo_adcconfig.c \
                           mmwdemo_monitor.c \
                           mmw_cli.c \
                           mmw_lvds_stream.c \
                           mmwdemo_flash.c \
                           mss_main.c \
                           mmwdemo_rfparser.c \
                           mmwdemo_board.c \
						   objectdetection_elevEst.c \

MSS_MMW_LC_DEMO_SOURCES_GEN  = ti_board_config.c \
								ti_board_open_close.c \
								ti_dpl_config.c \
								ti_drivers_config.c \
								ti_drivers_open_close.c \
								ti_pinmux_config.c \
								ti_power_clock_config.c \

MSS_MMW_LC_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_LC_DEMO_SOURCES:.c=.$($(MSS_CPU)_DEP_EXT)))
MSS_MMW_LC_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_LC_DEMO_SOURCES:.c=.$($(MSS_CPU)_OBJ_EXT)))
MSS_MMW_LC_DEMO_OBJECTS_GEN  = $(addprefix $(PLATFORM_OBJDIR)/mssgenerated/, $(MSS_MMW_LC_DEMO_SOURCES_GEN:.c=.$($(MSS_CPU)_OBJ_EXT)))

###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
mssLcDemo: $(MSS_CPU)_CFLAGS += -DAPP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
                              -DDebugP_LOG_ENABLED \
                              -DMMWDEMO_$(PROC_CHAIN) \
                              -DSOC_AWR2x44LC \
                              -DMSS_AOA_ENABLED=1

mssLcDemo: buildDirectories mssbuildDirectories $(MSS_MMW_LC_DEMO_OBJECTS) $(MSS_MMW_LC_DEMO_OBJECTS_GEN)
	$($(MSS_CPU)_LD) $($(MSS_CPU)_LDFLAGS) $(MSS_MMW_LC_DEMO_LOC_LIBS) -Wl,-m=$(MSS_MMW_LC_DEMO_MAP) \
	-o $(MSS_MMW_LC_DEMO_OUT) $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_LC_DEMO_OBJECTS_GEN) $(MSS_MMW_LC_DEMO_STD_LIBS) \
	$(R5F_LC_LINK_CMD) $(MSS_MMW_LC_DEMO_CMD)
	@echo "******************************************************************************"
	@echo 'Built the MSS for Millimeter Wave Demo'
	@echo "******************************************************************************"

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
mssLcDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_LC_DEMO_OBJECTS_GEN)
	@rm -f $(MSS_MMW_LC_DEMO_MAP) $(MSS_MMW_LC_DEMO_OUT) $(MSS_MMW_LC_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

mssLcDemoObjClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_LC_DEMO_OBJECTS_GEN) $(MSS_MMW_LC_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(MSS_MMW_LC_DEMO_DEPENDS)
