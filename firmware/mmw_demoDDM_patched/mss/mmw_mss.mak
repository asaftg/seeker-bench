###################################################################################
# Millimeter Wave Demo
###################################################################################
.PHONY: mssDemo mssDemoClean mssDemoEnet mssDemoObjClean
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
MSS_MMW_DEMO_STD_LIBS = $($(MSS_CPU)_COMMON_STD_LIB) \
                        -lmmwavelink_$(MSS_CPU_INSTANCE).lib \
                        -llibmmwave_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
                        -llibmathutils.$($(MSS_CPU)_LIB_EXT) \
                        -llibcli_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
                        -llibhsiheader_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT)

MSS_MMW_DEMO_LOC_LIBS = $($(MSS_CPU)_COMMON_LOC_LIB)  \
                        -Wl,-i$(MMWAVE_AWR294X_DFP_INSTALL_PATH)/ti/control/mmwavelink/lib	\
                        -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/mmwave/lib  \
                        -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/cli/lib   \
                        -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib \
                        -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/hsiheader/lib

MSS_MMW_ENET_DEMO_STD_LIBS = $(MSS_MMW_DEMO_STD_LIBS) \
                             -lenet-cpsw.awr2x44p.r5f.ti-arm-clang.$(LIB_TYPE).lib \
                             -llwipif-cpsw-freertos.awr2x44p.r5f.ti-arm-clang.$(LIB_TYPE).lib \
                             -llwip-freertos.awr2x44p.r5f.ti-arm-clang.$(LIB_TYPE).lib \
                             -llwip-contrib-freertos.awr2x44p.r5f.ti-arm-clang.$(LIB_TYPE).lib \

MSS_MMW_ENET_DEMO_LOC_LIBS = $(MSS_MMW_DEMO_LOC_LIBS) \
                            -Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/demo/awr2x44P/mmw_ddm/mss/mssgenerated \
							-Wl,-i${MCU_PLUS_INSTALL_PATH}/source/networking/enet/lib \
							-Wl,-i${MCU_PLUS_INSTALL_PATH}/source/networking/lwip/lib \

###################################################################################
# Additional Include which are required to build the Enet Stream enabled DEMO:
###################################################################################
mssDemoEnet: R5F_INCLUDE		+= -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet/utils/include \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet/core \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet/core/include \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet/core/include/phy \
                   -I${MCU_PLUS_INSTALL_PATH}/source/networking/enet/core/include/core \
                   -I${MCU_PLUS_INSTALL_PATH}/source/networking/enet/soc/awr2x44p \
                   -I${MCU_PLUS_INSTALL_PATH}/source/networking/enet/hw_include \
                   -I${MCU_PLUS_INSTALL_PATH}/source/networking/enet/hw_include/mdio/V4 \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/lwip/lwip-stack/src/include \
                   -I${MCU_PLUS_INSTALL_PATH}/source/networking/lwip/lwip-port/include \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/lwip/lwip-port/freertos/include \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/enet/core/lwipif/inc \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/lwip/lwip-contrib \
                   -I$(MCU_PLUS_INSTALL_PATH)/source/networking/lwip/lwip-config/awr2x44p \
                   -I$(MMWAVE_SDK_INSTALL_PATH)/ti/demo/awr2x44P/mmw_ddm/mss \

###################################################################################
# Millimeter Wave Demo
###################################################################################
R5F_ENET_LINK_CMD        = $(MMWAVE_SDK_INSTALL_PATH)/ti/platform/awr2x44x/$(MMWAVE_SDK_DEVICE_TYPE)/r5f_linker_enet.cmd
MSS_MMW_ENET_DEMO_MAP    = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN)ENET.map
MSS_MMW_ENET_DEMO_OUT    = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN)ENET.$($(MSS_CPU)_EXE_EXT)
MSS_MMW_ENET_DEMO_RPRC   = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN)ENET.rprc
MSS_MMW_DEMO_MAP         = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN).map
MSS_MMW_DEMO_OUT         = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN).$($(MSS_CPU)_EXE_EXT)
MSS_MMW_DEMO_RPRC        = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo_mss$(PROC_CHAIN).rprc
MSS_MMW_DEMO_CMD         = mss/mmw_mss_linker.cmd
MSS_MMW_DEMO_SOURCES     = mmwdemo_adcconfig.c \
                           mmwdemo_monitor.c \
                           mmw_cli.c \
                           mmw_lvds_stream.c \
                           mmwdemo_flash.c \
                           mss_main.c \
                           mmwdemo_rfparser.c \
                           mmwdemo_board.c \

MSS_MMW_DEMO_SOURCES_GEN  = ti_board_config.c \
                            ti_board_open_close.c \
                            ti_dpl_config.c \
                            ti_drivers_config.c \
                            ti_drivers_open_close.c \
                            ti_pinmux_config.c \
                            ti_power_clock_config.c \

MSS_MMW_DEMO_ENET_SOURCES = $(MSS_MMW_DEMO_SOURCES) \
                            enet_stream.c \
                            enet_cpswconfighandler.c \
                            enet_tcpclient.c

MSS_MMW_DEMO_SOURCES_GEN_ENET  = ti_enet_config.c \
                                 ti_enet_open_close.c \
                                 ti_enet_soc.c \
                                 ti_enet_lwipif.c \

ifeq ($(MSS_AOA_ENABLED), 1)
MSS_MMW_DEMO_SOURCES += objectdetection_elevEst.c
MSS_MMW_DEMO_ENET_SOURCES += objectdetection_elevEst.c
endif

MSS_MMW_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$($(MSS_CPU)_DEP_EXT)))
MSS_MMW_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$($(MSS_CPU)_OBJ_EXT)))
MSS_MMW_DEMO_OBJECTS_GEN  = $(addprefix $(PLATFORM_OBJDIR)/mssgenerated/, $(MSS_MMW_DEMO_SOURCES_GEN:.c=.$($(MSS_CPU)_OBJ_EXT)))

MSS_MMW_ENET_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_ENET_SOURCES:.c=.$($(MSS_CPU)_DEP_EXT)))
MSS_MMW_ENET_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_ENET_SOURCES:.c=.$($(MSS_CPU)_OBJ_EXT)))
MSS_MMW_DEMO_ENET_OBJECTS_GEN  = $(addprefix $(PLATFORM_OBJDIR)/mssgenerated/, $(MSS_MMW_DEMO_SOURCES_GEN_ENET:.c=.$($(MSS_CPU)_OBJ_EXT)))

###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
mssDemo: $(MSS_CPU)_CFLAGS += -DAPP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
                              -DDebugP_LOG_ENABLED \
                              -DMMWDEMO_$(PROC_CHAIN) \
                              -DMSS_AOA_ENABLED=$(MSS_AOA_ENABLED)

mssDemo: buildDirectories mssbuildDirectories $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN)
	$($(MSS_CPU)_LD) $($(MSS_CPU)_LDFLAGS) $(MSS_MMW_DEMO_LOC_LIBS) -Wl,-m=$(MSS_MMW_DEMO_MAP) \
	-o $(MSS_MMW_DEMO_OUT) $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN) $(MSS_MMW_DEMO_STD_LIBS) \
	$(PLATFORM_$(MSS_CPU)_LINK_CMD) $(MSS_MMW_DEMO_CMD)
	@echo "******************************************************************************"
	@echo 'Built the MSS for Millimeter Wave Demo'
	@echo "******************************************************************************"

###################################################################################
# Build Ethernet stream enabled Millimeter Wave Demo
###################################################################################
OBJ := $(MSS_MMW_ENET_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN) $(MSS_MMW_DEMO_ENET_OBJECTS_GEN)
mssDemoEnet: $(MSS_CPU)_CFLAGS += -DAPP_RESOURCE_FILE='<ti/demo/awr2x44P/mmw_ddm/mmw_res$(PROC_CHAIN).h>' \
                            -DENET_STREAM \
                            -DDebugP_LOG_ENABLED \
                            -DMMWDEMO_$(PROC_CHAIN) \
                            -DMSS_AOA_ENABLED=$(MSS_AOA_ENABLED)

mssDemoEnet: buildDirectories mssbuildDirectories $(OBJ)
	$($(MSS_CPU)_LD) $($(MSS_CPU)_LDFLAGS) $(MSS_MMW_ENET_DEMO_LOC_LIBS) -Wl,-m=$(MSS_MMW_ENET_DEMO_MAP) \
	-o $(MSS_MMW_ENET_DEMO_OUT) $(OBJ) $(MSS_MMW_ENET_DEMO_STD_LIBS) $(R5F_ENET_LINK_CMD) \
	$(MSS_MMW_DEMO_CMD)
	@echo "******************************************************************************"
	@echo 'Built the MSS for Millimeter Wave Demo for ENET'
	@echo "******************************************************************************"

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
mssDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN)
	@rm -f $(MSS_MMW_DEMO_MAP) $(MSS_MMW_DEMO_OUT) $(MSS_MMW_DEMO_DEPENDS)
	@rm -f $(MSS_MMW_ENET_DEMO_MAP) $(MSS_MMW_ENET_DEMO_OUT) $(MSS_MMW_ENET_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

mssDemoObjClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN) $(MSS_MMW_DEMO_DEPENDS)
	@rm -f $(MSS_MMW_ENET_DEMO_OBJECTS) $(MSS_MMW_DEMO_ENET_OBJECTS_GEN) $(MSS_MMW_ENET_DEMO_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(MSS_MMW_DEMO_DEPENDS)
