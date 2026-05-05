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

MSS_AOA_ENABLED?=1

include ./mss/mmw_mss.mak
include ./dss_cm4/mmw_dss_cm4.mak
include ./dss/mmw_dss.mak

MMW_DEMO_BIN         = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo$(PROC_CHAIN).appimage
MMW_ENET_DEMO_BIN    = $(MMWAVE_SDK_DEVICE_TYPE)_mmw_demo$(PROC_CHAIN)Enet.appimage

MMW_DEMO_BIN_HS      =  $(MMW_DEMO_BIN).hs
MMW_ENET_DEMO_BIN_HS =  $(MMW_ENET_DEMO_BIN).hs

ifeq ($(OS),Windows_NT)
PYTHON=python
else
PYTHON=python3
endif

#Sign and encrypt the appimage for authentication and decryption in HS-SE devices
SIGNING_TOOL_PATH=$(MCU_PLUS_INSTALL_PATH)/tools/boot/signing
KD_SALT=$(SIGNING_TOOL_PATH)/kd_salt.txt
APP_SIGNING_KEY=$(SIGNING_TOOL_PATH)/mcu_custMpk.pem
APP_ENCRYPTION_KEY=$(SIGNING_TOOL_PATH)/mcu_custMek.key
APP_IMAGE_SIGN_CMD = $(SIGNING_TOOL_PATH)/mcu_appimage_x509_cert_gen.py

ENC_ENABLED?=no
RSASSAPSS_ENABLED?=no
APP_SIGNING_HASH_ALGO?=sha512
APP_SIGNING_SALT_LENGTH?=0
APP_SIGNING_KEY_KEYRING_ID?=0
APP_ENCRYPTION_KEY_KEYRING_ID?=0

###################################################################################
# Standard Targets which need to be implemented by each mmWave SDK module. This
# plugs into the release scripts.
###################################################################################
.PHONY: all clean
.NOTPARALLEL:

mmwDemo: syscfg mssDemo dssCM4Demo dssDemo
mmwDemoEnet: syscfg_enet mssDemoEnet dssCM4DemoEnet dssDemoEnet

bin:
	$(OUTRPRC_CMD) $(MSS_MMW_DEMO_OUT) >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(OUTRPRC_CMD) $(M4_MMW_DEMO_OUT) >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(OUTRPRC_CMD) $(DSS_MMW_DEMO_OUT) >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(MULTI_CORE_IMAGE_GEN) --devID 55 --out $(MMW_DEMO_BIN) $(MSS_MMW_DEMO_RPRC)@0 $(M4_MMW_DEMO_RPRC)@2 $(DSS_MMW_DEMO_RPRC)@1 $(AWR2X44P_RADARSS_IMAGE_BIN)@3  >> $(BOOTIMAGE_TEMP_OUT_FILE)
	@$(DEL) $(MSS_MMW_DEMO_RPRC) $(M4_MMW_DEMO_RPRC) $(DSS_MMW_DEMO_RPRC) $(BOOTIMAGE_TEMP_OUT_FILE)
ifeq ($(ENC_ENABLED),no)
ifeq ($(RSASSAPSS_ENABLED),no)
	@echo App-image signing: Encryption is disabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_DEMO_BIN) --key $(APP_SIGNING_KEY) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --output $(MMW_DEMO_BIN_HS)
else
	@echo App-image signing: Encryption is disabled. RSASSAPSS is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_DEMO_BIN) --key $(APP_SIGNING_KEY) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --pss_saltlen $(APP_SIGNING_SALT_LENGTH) --output $(MMW_DEMO_BIN_HS) --rsassa_pss
endif
else
ifeq ($(RSASSAPSS_ENABLED),no)
	@echo App-image signing: Encryption is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_DEMO_BIN) --key $(APP_SIGNING_KEY) --enc y --enckey $(APP_ENCRYPTION_KEY) --kd-salt $(KD_SALT) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --enc_key_id $(APP_ENCRYPTION_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --output $(MMW_DEMO_BIN_HS)
	@$(DEL) $(BOOTIMAGE_NAME)-enc
else
	@echo App-image signing: Encryption is enabled. RSASSAPSS is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_DEMO_BIN) --key $(APP_SIGNING_KEY) --enc y --enckey $(APP_ENCRYPTION_KEY) --kd-salt $(KD_SALT) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --enc_key_id $(APP_ENCRYPTION_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --pss_saltlen $(APP_SIGNING_SALT_LENGTH) --output $(MMW_DEMO_BIN_HS) --rsassa_pss
	@$(DEL) $(BOOTIMAGE_NAME)-enc
endif
endif

enetbin:
	$(OUTRPRC_CMD) $(MSS_MMW_ENET_DEMO_OUT) >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(OUTRPRC_CMD) $(M4_MMW_ENET_DEMO_OUT)  >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(OUTRPRC_CMD) $(DSS_MMW_ENET_DEMO_OUT) >> $(BOOTIMAGE_TEMP_OUT_FILE)
	$(MULTI_CORE_IMAGE_GEN) --devID 55 --out $(MMW_ENET_DEMO_BIN) $(MSS_MMW_ENET_DEMO_RPRC)@0 $(M4_MMW_ENET_DEMO_RPRC)@2 $(DSS_MMW_ENET_DEMO_RPRC)@1 $(AWR2X44P_RADARSS_IMAGE_BIN)@3  >> $(BOOTIMAGE_TEMP_OUT_FILE)
	@$(DEL) $(MSS_MMW_ENET_DEMO_RPRC) $(M4_MMW_ENET_DEMO_RPRC) $(DSS_MMW_ENET_DEMO_RPRC) $(BOOTIMAGE_TEMP_OUT_FILE)
ifeq ($(ENC_ENABLED),no)
ifeq ($(RSASSAPSS_ENABLED),no)
	@echo App-image signing: Encryption is disabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_ENET_DEMO_BIN) --key $(APP_SIGNING_KEY) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --output $(MMW_ENET_DEMO_BIN_HS)
else
	@echo App-image signing: Encryption is disabled. RSASSAPSS is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_ENET_DEMO_BIN) --key $(APP_SIGNING_KEY) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --pss_saltlen $(APP_SIGNING_SALT_LENGTH) --output $(MMW_ENET_DEMO_BIN_HS) --rsassa_pss
endif
else
ifeq ($(RSASSAPSS_ENABLED),no)
	@echo App-image signing: Encryption is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_ENET_DEMO_BIN) --key $(APP_SIGNING_KEY) --enc y --enckey $(APP_ENCRYPTION_KEY) --kd-salt $(KD_SALT) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --enc_key_id $(APP_ENCRYPTION_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --output $(MMW_ENET_DEMO_BIN_HS)
	@$(DEL) $(MMW_ENET_DEMO_BIN)-enc
else
	@echo App-image signing: Encryption is enabled. RSASSAPSS is enabled.
	$(PYTHON) $(APP_IMAGE_SIGN_CMD) --bin $(MMW_ENET_DEMO_BIN) --key $(APP_SIGNING_KEY) --enc y --enckey $(APP_ENCRYPTION_KEY) --kd-salt $(KD_SALT) --sign_key_id $(APP_SIGNING_KEY_KEYRING_ID) --enc_key_id $(APP_ENCRYPTION_KEY_KEYRING_ID) --hash_algo $(APP_SIGNING_HASH_ALGO) --pss_saltlen $(APP_SIGNING_SALT_LENGTH) --output $(MMW_ENET_DEMO_BIN_HS) --rsassa_pss
	@$(DEL) $(MMW_ENET_DEMO_BIN)-enc
endif
endif

binClean:
	@$(DEL) $(MMW_DEMO_BIN) $(MMW_ENET_DEMO_BIN) $(MMW_DEMO_BIN_HS) $(MMW_ENET_DEMO_BIN_HS)

CORE_0=--script ./mss/mss.syscfg --context r5fss0-0 --output ./mss/mssgenerated/
CORE_1=--script ./dss_cm4/dss_cm4.syscfg --context m4ss0-1 --output ./dss_cm4/m4generated/
CORE_2=--script ./dss/dss.syscfg --context c66ss0 --output ./dss/dssgenerated/
CORE_0_ENET=--script ./mss/mss_enet.syscfg --context r5fss0-0 --output ./mss/mssgenerated/

CORES = \
    $(CORE_2) \
    $(CORE_1) \
    $(CORE_0) \

CORES_ENET = \
    $(CORE_2) \
    $(CORE_1) \
    $(CORE_0_ENET) \

syscfg-gui:
	$(SYSCFG_NWJS) $(SYSCFG_CLI_PATH) --product $(SYSCFG_SDKPRODUCT) --device AWR2X44P --part Default --package $(PACKAGE_TYPE) $(CORES)

enetsyscfg-gui:
	$(SYSCFG_NWJS) $(SYSCFG_CLI_PATH) --product $(SYSCFG_SDKPRODUCT) --device AWR2X44P --part Default --package $(PACKAGE_TYPE) $(CORES_ENET)

mmwDemoDDM:
	$(MAKE) PROC_CHAIN=DDM objClean mmwDemo bin
mmwDemoDDMEnet:
	$(MAKE) PROC_CHAIN=DDM objClean mmwDemoEnet enetbin

mmwDemoDDMClean:
	$(MAKE) PROC_CHAIN=DDM objClean mssDemoClean dssCM4DemoClean dssDemoClean binClean
mmwDemoDDMEnetClean:
	$(MAKE) PROC_CHAIN=DDM objClean mssDemoClean dssCM4DemoClean dssDemoClean binClean

# syscfg: This generates syscfg files
syscfg:
	@echo Generating SysConfig files ...
	$(SYSCFG_NODE) $(SYSCFG_CLI_PATH)/dist/cli.js --product $(SYSCFG_SDKPRODUCT) $(CORES)

syscfg_enet:
	@echo Generating SysConfig files ...
	$(SYSCFG_NODE) $(SYSCFG_CLI_PATH)/dist/cli.js --product $(SYSCFG_SDKPRODUCT) $(CORES_ENET)

# syscfg: This cleans syscfg files
syscfgclean:
	@echo 'Cleaning the syscfg files'
	@$(DEL) mss/mssgenerated
	@$(DEL) dss/dssgenerated
	@$(DEL) dss_cm4/m4generated


objClean: mssDemoObjClean dssDemoObjClean dssCM4DemoObjClean syscfgclean
