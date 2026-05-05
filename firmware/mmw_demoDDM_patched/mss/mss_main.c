/**
 *   @file  mss_main.c
 *
 *   @brief
 *      This is the main file which implements the millimeter wave Demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2018-26 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @mainpage Millimeter Wave (mmw) Demo (DDM) for AWR2X44P/AWR2X44ECO/AWR2X44LC
 * [TOC]
 *  @section intro_sec Introduction
 *
 *  @image html toplevel.png
 *
 *  The millimeter wave demo shows some of the capabilities of the AWR2X44P/AWR2X44ECO/AWR2X44LC SOC
 *  using the drivers in the mmWave SDK (Software Development Kit).
 *  It allows user to specify the chirping profile and displays the detected
 *  objects and other information in real-time.
 *
 *  Following is a high level description of the features of this demo:
 *  - Be able to specify desired chirping profile through command line interface (CLI)
 *    on a UART port or through the TI Gallery App - **mmWave Demo Visualizer** -
 *    that allows user to provide a variety of profile configurations via the UART input port
 *    and displays the streamed detected output from another UART port in real-time,
 *    as seen in picture above.
 *  - Some sample profile configurations have been provided in the demo directory that can be
 *    used with CLI directly or via **mmWave Demo Visualizer** under following directory:
 *    @verbatim mmw_ddm/profiles @endverbatim
 *  - Do 1D, 2D, CFAR, Azimuth and Elevation processing and stream out velocity
 *    and three spatial coordinates (x,y,z) of the detected objects in real-time.
 *    The demo can also be configured to do 2D only detection (velocity and x,y coordinates).
 *  - Various display options besides object detection like Doppler-range heat map.
 *
 *  @section procChain Processing Chains
 *    DDM processing chains is available for the AWR2X44P/AWR2X44ECO/AWR2X44LC demo.
      Check @verbatim
      ti/datapath/dpc/objectdetection/objdethwaDDMA/docs/doxygen/html/index.html
      @endverbatim
 *    for details on the processing chains. The CLI arguments for DDM chain are available in
 *    the User Guide.
 *  - DDM (Doppler Division Multiplexing): Here, all transmitters are active at the same time.
 *    At the receiver antennae, the received chirps need to be disambiguated (demodulation)
 *    to obtain samples corresponding to each Tx antenna. The antenna support for this chain is
 *    (AzimTx, ElevTx, Rx) = (3,1,4) for AWR2X44P/AWR2X44ECO/AWR2X44LC.
 *
 *   Feature | AWR2X44P-DDM | AWR2X44ECO-DDM | AWR2X44LC-DDM
 *  :----|:------|:-------|:------------:
 *   Legacy Frame | Yes | Yes | Yes
 *   Advanced subframe | Yes | Yes | Yes
 *   LVDS Streaming | Yes | Yes | Yes
 *   Enet Streaming | Yes | Yes | No
 *   Range Bias Estimation/Compensation | No | Yes | Yes
 *   RX Gain/Phase Estimation/Compensation | Yes | Yes | Yes
 *
 *  @section limit Limitations
 *  - Datapath initialization and configuration APIs are placed in L3_RAM. This code is replaced by the data during
 *    run-time after sensor start. Thus, datapath re-configuration using APIs is not supported.
 *
 *  @section systemFlow System Execution Flow
 *  The millimeter wave demo runs on ARM Cortex R5F (MSS) and M4 (DSS_CM4) with DSP(DSS) idle. Following diagram shows the system
 *  execution flow for AWR2x44P/AWR2x44ECO/AWR2x44LC device:
 *
 *  @image html system_flow.png "System Execution Flow"
 *
 *  This demo runs on MSS and DSS_CM4.
 * \n
 *    **MSS**
 * \n
 *   Following tasks (FreeRTOS) runs on the MSS:
 *    - @ref MmwDemo_initTask. This task is created/launched by @ref main and is a
 *      one-time active task whose main functionality is to initialize drivers (\<driver\>_init),
 *      MMWave module (MMWave_init), DPM module (DPM_init), open UART and other drivers (SPI),
 *      and create/launch the following tasks
 *      (the @ref CLI_task is launched indirectly by calling @ref CLI_open).
 *    - @ref CLI_task. This command line interface task provides a simplified 'shell' interface
 *      which allows the configuration of the BSS via the mmWave interface (MMWave_config).
 *      It parses input CLI configuration commands like chirp profile and GUI configuration.
 *      When sensor start CLI command is parsed, all actions related to starting sensor and
 *      starting the processing the data path are taken.
 *      When sensor stop CLI command is parsed, all actions related to stopping the sensor and
 *      stopping the processing of the data path are taken
 *    - @ref MmwDemo_mmWaveCtrlTask. This task is used to provide an execution
 *      context for the mmWave control, it calls in an endless loop the MMWave_execute API.
 *    - @ref mmwDemo_mssUartDataExportTask. This task is used to export the data on UART. This task
 *      pends for @ref MmwDemo_MSS_MCB_t::UartExportSemHandle in an endless loop, which is posted when current frame
 *      processing and previous frame's uart transmission is completed. When AoA elevation processing is
 *      programmed to run on MSS, this task performs the angle estimation to obtain the spatial coordinates
 *      using the DPC's \ref DPC_ObjDet_estimateXYZ function.
 *
 *    **DSS_CM4**
 *  \n
 *   There is no real-time operating system (NORTOS) running on DSS_CM4, for the sake of application
 *    simplicity. Following diagram shows the main states that run on this core, primarily responsible for DPC initialization, configuration,
 *    start and execution.
 *    @image html dssCM4_FSM.png "Application State Machine on DSS_CM4"
 *
 *    - Initialization: Initialize drivers (\<driver\>_init), DPM module (DPM_init), data path related drivers (EDMA, HWA),
 *      Datapath Chain (DPC) and Datapath Units (DPUs). After the initilation, application enters the \ref MMWDEMO_DPC_COMMONCFG state.
 *    - Configuration: Application waits for common config to be available from MSS. Once the common config is received, application copies the config,
 *      send the acknowledgement, and update the state to \ref MMWDEMO_DPC_CFG state. Again, in this state, same set of actions is repeated as in
 *      the previous state.
 *    - Start: Software trigger the execution of datapath before sensor start. So, that per chirp processing can happen as the data comes. After
 *      pre-start configuration and DPC start, instead of just sending back the acknowledgement, application sends the
 *      \ref DPC_ObjectDetection_ElevEstCfg structure. to be used by MSS for communicating the cfg to DSS.
 *    - Execution: In \ref MMWDEMO_DPC_EXECUTE state, DPU process functions are executed. After the completion of Doppler DPU, DSS is notified to
 *      further process the data (angle estimation) to obtain spatial coordinates.
 *
 *  Once application reaches the \ref MMWDEMO_DPC_EXECUTE state, it always remains here. This is because config APIs are replaced by data after sensor start and thus,
 *  other states become invalid as reconfiguration is not possible using APIs. Based on frame start event, DPC_execution takes place. In case of sensor stop,
 *  application keeps waiting in this state for the frame start event.
 *
 *    **DSS**
 * \n
 *   Following tasks (FreeRTOS) runs on the DSS on AWR2x44P/AWR2x44ECO:
 *    - @ref MmwDemo_dssInitTask. This task is created/launched by @ref main and is used to provide an execution context.
 *     It initializes the DPM module (DPM_init), synchronizes the DPM (DPM_synch). If AoA elevation processing is configured
 *     to run on DSS, this task configures the DPC for elevation estimation and then waits for results ready notification from
 *     DSS_CM4 in an endless loop. DSS performs the angle estimation to obtain the spatial coordinates. When the DPC's
 *     \ref DPC_ObjDet_estimateXYZ function produces the detected objects and other results, they are reported to MSS
 *     where they are transmitted out of the UART port for display using the visualizer. But if AoA elevation processing is
 *     configured to run on MSS, the DSS remains idle waiting for the configuration semaphore post.
 *
 *   Following diagram shows the system execution flow when AoA elevation processing is configured to run on DSS:
 *
 *   @image html system_flow_aoa_dss.png "System Execution Flow with AoA elevation processing on DSS"
 *
 *  @section datapath Data Path
 *   @image html datapath_overall.png "Top Level Data Path Processing Chain"
 *   \n
 *   @image html datapath_overall_timingDDM.png "Top Level Data Path Timing - DDM"
 *
 *   The data path processing consists of taking ADC samples as input and producing
 *   detected objects (point-cloud and other information) to be shipped out of
 *   UART port to the PC. The algorithm processing is realized using
 *   the Object Detection DPC. The details of the processing in DPC
 *   can be seen from the following doxygen documentation:
 *   @verbatim
      ti/datapath/dpc/objectdetection/objdethwaDDMA/docs/doxygen/html/index.html
     @endverbatim
 *
 *  @section DDMP Phase shifters (DDM)
 *   ti/datapath/dpc/objectdetection/objdethwaDDMA/docs/doxygen/html/index.html can be checked for information
 *   on why phase shifts are needed in the DDMA processing chain and what their values will be.
 *   CLI ddmPhaseShiftAntOrder takes antennas indices in increasing order of phase shift value,
 *   if all of them were enabled. For example, a value of {0,3,1,2} would mean that phase shifts for a particular chirp are
 *   in the following order of magnitude- tx0ChirpPhase < tx3ChirpPhase < tx1ChirpPhase < tx2ChirpPhase
 *   Even if the user does not intend to use all the tx antennas, the order should be programmed assuming
 *   that all the Tx were enabled. The phase shift values for the ones that are not enabled will be
 *   configured to 0 by the code. \n
 *
 *   Note that in the DDMA case, the elevation antenna(s) should always come at the end of this array.
 *   Basically, phaseShift(azimuth) < phaseShift(elevation) must be ensured. Hence, {0, 2, 3, 1} should be configured
 *   for AWR2944ETS, since Tx0, Tx2, Tx3 are azimuth antennas and Tx1 is elevation antenna.
 *
 *  @section antGeometryCfg Antenna Geometry Configuration (DDM)
 *  DDM Demo supports different antenna configurations configurable via CLI antGeometryCfg. User can change the layout using the antGeometryCfg command.
 *  In this command, the row and column index of each antenna define the virtual antennas' physical location index (0, 1, 2, ...)
 *  in the elevation and azimuth domains, respectively, as illustrated in below figure.
 *  In other words, for the antenna pattern shown below, these index values indicate two elevation rows (specified by 0 and 1), and 12 azimuth columns (specified by
 *  0, 1, 2, ... 11) followed by the last two arguments representing the unit of cell spacing in azimuth and elevation dimension in units of &lambda;.
 *
 *  @verbatim
    antGeometryCfg <Row(Tx0Rx0)> <Col(Tx0Rx0)> <Row(Tx0Rx1)> <Col(Tx0Rx1)> ... <Row(Tx0Rx[R-1])> <Col(Tx0Rx[R-1])> <Row(Tx1Rx0)> <Col(Tx1Rx0)> ... <Row(Tx[T-1]Rx[R-1])> <Col(Tx[T-1]Rx[R-1])>
    @endverbatim
 *  \n
 *
 *  @image html LOP_antenna.png "Virtual antenna pattern and corresponding antenna indices"
 *
 *  For the antenna pattern shown in figure, CLI command should be given as
 *  @verbatim
    antGeometryCfg 1 0 1 2 1 7 1 9 1 3 1 5 1 10 1 12 1 6 1 8 1 13 1 15 0 9 0 11 0 16 0 18 0.5 0.5
    @endverbatim
 *
 *  From the entered antenna geometry, antenna rearrangement order and zero insertion mask in azimuth and elevation row
 *  is computed, required for azimuth and elevation angle estimation. Provided antenna calibration parameters are also
 *  rearranged based on the computed rearrangement order.
 *
 * To configure the DDM phase shifters, following TX antenna order needs to be provided for the above antenna array.
 *  @verbatim
    ddmPhaseShiftAntOrder 0 1 2 3
    @endverbatim
 *
 *  @section optimization DDMA Optimizations
 *  In order to reduce the processing time of DDMA chain, primarily following optimizations are done:
 *  - HWA/DMA/DSS_CM4 parallelization.
 *  - EDMA Polling instead of Interrupts.
 *  - Using Linear transfers in EDMA instead of Transpose transfers.
 *  - Optimization of DSS_CM4 code of DDMA Demodulation - mapped to HWA and EDMA for rearrangement.
 *  - Parallelizing AoA processing with the next frame.
 *  - Parallelizing UART data sending the next frame.
 *  - Reduction of number of Azimuth Bins from 48 to 32.
 *  - Disable of Range CFAR / Sum TX.
 *  - Optimization of DSP code of AoA processing.
 *  - Optimization of decompression stage for lesser number of range bins per compressed block.
 *
 * AoA uses DSP and 1D processing is done entirely on HWA, so both of them can be parallelized.
 * Similarly, UART TX run entirely on R5F, so it can be parallelized with the processes running on HWA and DSS_CM4.
 * So, after the parallel processing optimizations, AoA processing and UART transfer of the current frame can run in
 * parallel with the 1D and 2D processing of the next frame.
 *
 *  @section output Output information sent to host
 *      @subsection output_general Output Packet
 *      Output packets with the detection information are sent out every frame
 *      through the UART. Each packet consists of the header @ref MmwDemo_output_message_header_t
 *      and the number of TLV items containing various data information with
 *      types enumerated in @ref MmwDemo_output_message_type_e. The numerical values
 *      of the types can be found in @ref mmw_output.h. Each TLV
 *      item consists of type, length (@ref MmwDemo_output_message_tl_t) and payload information.
 *      The structure of the output packet is illustrated in the following figure.
 *      Since the length of the packet depends on the number of detected objects
 *      it can vary from frame to frame. The end of the packet is padded so that
 *      the total packet length is always multiple of 32 Bytes.
 *
 *      @image html output_packet_uart.png "Output packet structure sent to UART"
 *
 *      The following subsections describe the structure of each TLV.
 *
 *      @subsection tlv1 List of detected objects
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_DETECTED_POINTS)
 *
 *       Length: (Number of detected objects) x (size of @ref DPIF_PointCloudCartesian_t)
 *
 *       Value: Array of detected objects. The information of each detected object
 *       is as per the structure @ref DPIF_PointCloudCartesian_t. When the number of
 *       detected objects is zero, this TLV item is not sent. The maximum number of
 *       objects that can be detected in a sub-frame/frame is @ref DPC_OBJDET_MAX_NUM_OBJECTS.
 *
 *       The orientation of x,y and z axes relative to the sensor is as per the
 *       following figure.
 *
 *        @image html coordinate_geometry.png "Coordinate Geometry"
 *
 *       The whole detected objects TLV structure is illustrated in figure below.
 *
 *       @image html detected_objects_tlv.png "Detected objects TLV"
 *
 *      @subsection tlv2 Range profile
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_RANGE_PROFILE)
 *
 *       Length: (Range FFT size) x (size of uint16_t)
 *
 *       Value: Array of profile points at 0th Doppler (stationary objects).
 *       The points represent the sum of log2 magnitudes of received antennas
 *       expressed in Q9 format.
 *
 *      @subsection tlv3 Noise floor profile
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_NOISE_PROFILE)
 *
 *       Length: (Range FFT size) x (size of uint16_t)
 *
 *       Value: This is the same format as range profile but the profile
 *       is at the maximum Doppler bin (maximum speed
 *       objects). In general, for stationary scene, there would be no objects or clutter at
 *       maximum speed so the range profile at such speed represents the
 *       receiver noise floor.
 *
 *      @subsection tlv6 Stats information
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_STATS )
 *
 *       Length: (size of @ref MmwDemo_output_message_stats_t)
 *
 *       Value: Timing information as per @ref MmwDemo_output_message_stats_t.
 *       See timing diagram below related to the stats.
 *
 *      @image html processing_timing.png "Processing timing"
 *
 *       Note:
 *       -#  The @ref MmwDemo_output_message_stats_t::interChirpProcessingMargin is not
 *           computed (it is always set to 0). This is because there is no CPU involvement
 *           in the 1D processing (only HWA and EDMA are involved), and it is not possible to
 *           know how much margin is there in chirp processing without CPU being notified
 *           at every chirp when processing begins (chirp event) and when the HWA-EDMA
 *           computation ends. The CPU is intentionally kept free during 1D processing
 *           because a real application may use this time for doing some post-processing
 *           algorithm execution.
 *       -#  While the
 *           @ref MmwDemo_output_message_stats_t::interFrameProcessingTime reported
 *           will be of the current sub-frame/frame,
 *           the @ref MmwDemo_output_message_stats_t::interFrameProcessingMargin and
 *           @ref MmwDemo_output_message_stats_t::transmitOutputTime
 *           will be of the previous sub-frame (of the same
 *           @ref MmwDemo_output_message_header_t::subFrameNumber as that of the
 *           current sub-frame) or of the previous frame.
 *       -#  The @ref MmwDemo_output_message_stats_t::interFrameProcessingMargin excludes
 *           the UART transmission time (available as
 *           @ref MmwDemo_output_message_stats_t::transmitOutputTime). This is done
 *           intentionally to inform the user of a genuine inter-frame processing margin
 *           without being influenced by a slow transport like UART, this transport time
 *           can be significantly longer for example when streaming out debug information like
 *           heat maps. Also, in a real product deployment, higher speed interfaces (e.g LVDS)
 *           are likely to be used instead of UART. User can calculate the margin
 *           that includes transport overhead (say to determine the max frame rate
 *           that a particular demo configuration will allow) using the stats
 *           because they also contain the UART transmission time.
 *
 *      The CLI command \"guiMonitor\" specifies which TLV element will
 *      be sent out within the output packet. The arguments of the CLI command are stored
 *      in the structure @ref MmwDemo_GuiMonSel_t.
 *
 *      @subsection tlv7 Side information of detected objects
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO)
 *
 *       Length: (Number of detected objects) x (size of @ref DPIF_PointCloudSideInfo_t)
 *
 *       Value: Array of detected objects side information. The side information
 *       of each detected object is as per the structure @ref DPIF_PointCloudSideInfo_t).
 *       When the number of detected objects is zero, this TLV item is not sent.
 *
 *      @subsection tlv9 Temperature Stats
 *       Type: (@ref MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS)
 *
 *       Length: (size of @ref MmwDemo_temperatureStats_t)
 *
 *       Value: Structure of detailed temperature report as obtained from Radar front end.
 *       @ref MmwDemo_temperatureStats_t::tempReportValid is set to return value of
 *       rlRfGetTemperatureReport. If @ref MmwDemo_temperatureStats_t::tempReportValid is 0,
 *       values in @ref MmwDemo_temperatureStats_t::temperatureReport are valid else they should
 *       be ignored. This TLV is sent along with Stats TLV described in @ref tlv6
 *
 *  @section Calibration_section Rx Channel Gain/Phase Measurement and Compensation
 *
 *     Because of imperfections in antenna layouts on the board, RF delays in SOC, etc,
 *     there is need to calibrate the sensor to compensate for bias in the
 *     receive channel gain and phase imperfections. The following figure illustrates
 *     the calibration procedure.
 *
 *     @anchor Figure_calibration_ladder_diagram
 *     @image html calibration_ladder_diagram.png "Calibration procedure ladder diagram"
 *
 *      The calibration procedure includes the following steps:
 *     -# Set a strong target like corner reflector at the distance of X meter
 *     (X less than 50 cm is not recommended) at boresight.
 *     -# Set the following command
 *     in the configuration profile in .../profiles/profile_calibration.cfg,
 *     to reflect the position X as follows:
 *     @verbatim
       measureRangeBiasAndRxChanPhase 1 X D
       @endverbatim
 *     where D (in meters) is the distance of window around X where the peak will be searched.
 *     The purpose of the search window is to allow the test environment from not being overly constrained
 *     say because it may not be possible to clear it of all reflectors that may be stronger than the one used
 *     for calibration. The window size is recommended to be at least the distance equivalent of a few range bins.
 *     One range bin for the calibration profile (profile_calibration.cfg) is about 5 cm.
 *     The first argument "1" is to enable the measurement. The stated configuration
 *     profile (.cfg) must be used otherwise the calibration may not work as expected
 *     (this profile ensures all transmit and receive antennas are engaged among
 *     other things needed for calibration).
 *     -# Start the sensor with the configuration file.
 *     -# In the configuration file, the measurement is enabled because of which
 *     the DPC will be configured to perform the measurement and generate
 *     the measurement result (@ref Measure_compRxChannelBiasCfg in DDM) in its
 *     result structure (@ref DPC_ObjectDetection_ExecuteResult_t::compRxChanBiasMeasurement),
 *     the measurement results are written out on the CLI port (@ref MmwDemo_measurementResultOutput)
 *     in the format below:
 *     @verbatim
       DDM: compRxChanPhase <Im(0,0)> <Re0,0)> <Im(0,1)> <Re(0,1)> ... <Im(0,R-1)> <Re(0,R-1)> <Im(1,0)> <Re(1,0)> ... <Im(T-1,R-1)> <Re(T-1,R-1)>
       where TX antenna order is such that azimuth Tx antennas are followed by the elevation TX antennas.
       @endverbatim

       For details of how DPC performs the measurement, see the DPC documentation.
 *     -# The command printed out on the CLI now can be copied and pasted in any
 *     configuration file for correction purposes. This configuration will be
 *     passed to the DPC for the purpose  of applying compensation during angle
 *     computation, the details of this can be seen in the DPC documentation.
 *     If compensation is not desired,
 *     the following command should be given
 *     @verbatim
       DDM: antennaCalibParams 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
       @endverbatim
 *     Above sets the phase coefficients to unity so that there is no correction. Note
 *     the two commands must always be given in any configuration file, typically
 *     the measure commmand will be disabled when the correction command is the
 *     desired one. It is recommended to use the measureRangeBiasAndRxChanPhase command only in the
 *     in the frame based chirps (dfeDataOutput=1) mode.
 * 
 *     Note: For validation of the antenna calibration parameters, user can program the parameters returned in compRxChanPhase
 *     into antennaCalibParams CLI in the same calibration profile (profile_calibration.cfg) and disable measureRangeBiasAndRxChanPhase.
 *     Now, the Range Doppler Heatmap should show strong peak at the corner reflector without any extra points at the same range.
 *     If such a peak is not observed, user should increase fftOutputScaling and capture the calibration parameters again.
 *
 *  @section LVDSStreamingNotes Streaming data over LVDS
 *
 *    The LVDS streaming feature enables the streaming of HW data (ADC data)
 *    and/or user specific SW data through LVDS interface.
 *    The streaming is done mostly by the CBUFF and EDMA peripherals with minimal CPU intervention.
 *    The streaming is configured through the @ref MmwDemo_LvdsStreamCfg_t CLI command which
 *    allows control of HSI header, enable/disable of HW and SW data and data format choice for the HW data. Note
 *    that only HW data without HSI header is supported as of now.
 *    The choices for data formats for HW data are:
 *      - @ref MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED
 *      - @ref MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_ADC
 *
 *    When HW data LVDS streaming is enabled, the ADC data is streamed per
 *    chirp on every chirp event.
 *    -# For HW data, the inter-chirp duration should be sufficient to stream out the desired amount
 *       of data. For example, if the HW data-format is ADC and HSI header is enabled (note: HSI
 *       header sending is not enabled in this demo),
 *       then the total amount of data generated per chirp
 *       is:\n
 *        (numAdcSamples * numRxChannels * 4 (size of complex sample) +
 *       52 [sizeof(HSIDataCardHeader_t) + sizeof(HSISDKHeader_t)] )
 *       rounded up to multiples of 256 [=sizeof(HSIHeader_t)] bytes.\n
 *       The chirp time Tc in us = idle time + ramp end time in the profile configuration.
 *       For n-lane LVDS with each lane at a maximum of B Mbps,\n
 *       maximum number of bytes that can be send per chirp = Tc * n * B / 8 which
 *       should be greater than the total amount of data generated per chirp i.e\n
 *       Tc * n * B / 8 >= round-up(numAdcSamples * numRxChannels * 4 + 52, 256). \n
 *       E.g if n = 2, B = 600 Mbps, idle time = 7 us, ramp end time = 44 us, numAdcSamples = 512,
 *       numRxChannels = 4, then 7650 >= 8448 is violated so this configuration will not work.
 *       If the idle-time is doubled in the above example, then we have 8700 > 8448, so
 *       this configuration will work.
 *    -# The total amount of data to be transmitted in a HW or SW packet must be greater than the
 *       minimum required by CBUFF, which is 64 bytes or 32 CBUFF Units (this is the definition
 *       CBUFF_MIN_TRANSFER_SIZE_CBUFF_UNITS in the CBUFF driver implementation).
 *       If this threshold condition is violated, the CBUFF driver will return an error during
 *       configuration and the demo will generate a fatal exception as a result.
 *       When HSI header is enabled, the total transfer size is ensured to be at least
 *       256 bytes, which satisfies the minimum. If HSI header is disabled, for the HW session,
 *       this means that numAdcSamples * numRxChannels * 4 >= 64. Although mmwavelink
 *       allows minimum number of ADC samples to be 2, the demo is supported
 *       for numAdcSamples >= 64. So HSI header is not required to be enabled for HW only case.
 *
 *   @subsection lvdsImpl Implementation Notes
 *
 *    -# The LVDS implementation is mostly present in mmw_lvds_stream.h and
 *      mmw_lvds_stream.c with calls in mss_main.c. See the
 *      @ref MmwDemo_BoardInit function for register configuration related to HSI clock.
 *    -# EDMA channel resources for CBUFF/LVDS are in the global resource file
 *      (mmw_res.h, see @ref resourceAlloc) along with other EDMA resource allocation.
 *      The user data header and two user payloads are configured as three user buffers in the CBUFF driver.
 *      Hence SW allocation for EDMA provides for three sets of EDMA resources as seen in
 *      the SW part (swSessionEDMAChannelTable[.]) of @ref MmwDemo_LVDSStream_EDMAInit.
 *    -# Although the CBUFF driver is configured for two sessions (hw and sw),
 *      at any time only one can be active. So depending on the LVDS CLI configuration
 *      and whether advanced frame or not, there is logic to activate/deactivate
 *      HW and SW sessions as necessary.
 *    -# The CBUFF session (HW/SW) configure-create and delete
 *      depends on whether or not re-configuration is required after the
 *      first time configuration.
 *      -# For HW session, re-configuration is done during sub-frame switching to
 *        re-configure for the next sub-frame but when there is no advanced frame
 *        (number of sub-frames = 1), the HW configuration does not need to
 *        change so HW session does not need to be re-created.
 *    -# SW-trigger-based streaming is not supported in the current release.
 *
 *    The following figure shows a timing diagram for the LVDS streaming
 *    (the figure is not to scale as actual durations will vary based on configuration).
 *    Note: SW streaming is not supported in the current version of the demo.
 *
 *      @image html lvdstiming.png "LVDS timing diagram"
 *
 *  @section EnetStreamingNotes Streaming data over Ethernet
 *
 *    This demo supports a simple use case of transferring detected objects data over Ethernet using
 *    TCP protocol and LwIP stack. The data transfer is done using a client-server based architecture
 *    with the EVM acting as the client and the PC acting as a server. This utility is a modified
 *    example of the TCPECHO app that comes packages as a part of LwIP in the mcu_plus_sdk.
 *    For connection and usage details, please refer to the MMWAVE SDK User Guide.
 *
 *   @subsection enetImpl Implementation Notes
 *
 *    -# The enet streaming utility comes as a part of the files enet_cpswconfighandler.c, enet_stream.c
 *       enet_tcpclient.c and tcpserver.py files.
 *    -# enetTask in enet_stream.c is the main task that initializes all the components and tasks, and has a static IP address (192.168.1.200)
 *       assigned to the device (optionally user can enable IP acquisition through DHCP server.
 *    -# Once the IP is assigned to the device, the user can run the server application tcpserver.py and wait for
 *       connection establishment.
 *    -# Command queryLocalIp can be used through CLI to return the local IP (IP address acquired by EVM). Refer to the
 *       User Guide for more details.
 *    -# The user will send the local IP through the enetStreamCfg command using CLI. Refer to the
 *       User Guide for more details. Once this configuration is done, the semaphore EnetCfgDoneSemHandle
 *       will be posted, and connection will be established (see tcpclient.c).
 *    -# The user will be able to see the object data getting printed on the console of tcpserver.py.
 *    -# Note that the communication port is pre-configured as "7".
 *    -# It must be noted that the LwIP stack requires extra memory (L3 RAM), and care must be taken to
 *       ensure that the demo L3 requirements do not result in higher L3 memory usage than what is available.
 *
 *  @section bypassCLI How to bypass CLI
 *
 *    Re-implement the file mmw_cli.c as follows:
 *
 *    -# @ref MmwDemo_CLIInit should just create a task with input taskPriority.
 *       Lets say the task is called "MmwDemo_sensorConfig_task".
 *    -# All other functions are not needed
 *    -# Implement the MmwDemo_sensorConfig_task as follows:
 *       - Fill gMmwMssMCB.cfg.openCfg
 *       - Fill gMmwMssMCB.cfg.ctrlCfg
 *       - Add profiles and chirps using @ref MMWave_addProfile and @ref MMWave_addChirp functions
 *       - Call @ref MmwDemo_CfgUpdate for every offset in @ref configStoreOffsets
 *         (MMWDEMO_xxx_OFFSET in mmw_mss.h)
 *       - Fill gMmwMssMCB.objDetCommonCfg.preStartCommonCfg
 *       - Call @ref MmwDemo_openSensor
 *       - Call @ref MmwDemo_configSensor
 *       - Call @ref MmwDemo_startSensor (One can use helper function
 *         @ref MmwDemo_isAllCfgInPendingState to know if all dynamic config was provided)
 *    -# The user can also use the CLI_BYPASS API in the CLI library to directly bypass the CLI
 *       command send over UART.
 *
 *  @section resourceAlloc Hardware Resource Allocation
 *    The Object Detection DPC needs to configure the DPUs hardware resources (HWA, EDMA).
 *    Even though the hardware resources currently are only required to be allocated
 *    for this one and only DPC in the system, the resource partitioning is shown to be in
 *    the ownership of the demo. This is to illustrate the general
 *    case of resource allocation across more than one DPCs and/or demo's own
 *    processing that is post-DPC processing. This partitioning
 *    can be seen in the mmw_res.h file. This file is passed as a compiler command line
 *    define @verbatim "--define=APP_RESOURCE_FILE="<ti/demo/awr2x44P/mmw/mmw_resDDM.h>" @endverbatim
 *    in mmw_mss.mak and mmw_dss_cm4.mak when building the DPC sources as part of building the demo application and
 *    is referred in object detection DPC sources where needed as
 *    @verbatim #include APP_RESOURCE_FILE @endverbatim
 *
 *  @section poweropt Power Reduction techniques and supporting features
 *
 *   In order to reduce the power consumption of the Radar device, primarily following optimizations can be done:
 *   (Note: The power optimization hooks are not supported with advanced frame enabled.)
 *
 *   @subsection cgStatic Power Reduction Techniques: Clock Gating Static
 *
 *    **Clock Gate unused peripherals**
 *
 *    The device includes several peripherals and modules, and several instances of them. Depending on the application,
 *    some of these may be active while the others remain totally unused. The unused peripherals (RTIs, SPIs, I2C, CSI2, UARTs)
 *    can be clock-gated to save power. Further, not all the HSDIVs and not all outputs of each HSDIV need to be active.
 *
 *    Feature | Application
 *    :-----------------------------------|:-----------------------------------:
 *    Window of operation : transition to IDLE mode | Always. Applicable for the entire power cycle.
 *    Power save Measures in IDLE mode | Clock gating of all unused peripherals and modules across the SoC.
 *    Allocation | API Invoked and Implemented by Customer Application
 *    References | Reference APIs in the SDK: powerMeasPerClockGating; User guide for default HSDIV configuration
 *
 *    **SDK feature**:
 *
 *    - **Clock Gate unused peripherals**:
 *       Provided example clock gates the below peripherals:
 *       - MSS: SPIA , I2C, MII100, MII10
 *       - CSIRX, OBSCLKOUT, PMICCLKOUT, TRCCLKOUT
 *       - DSS: SCIA, CBUFF
 *       - RSS: CSI2A
 *    - **Disable LVDS Streaming**:
 *      By default SDK enables LVDS streaming (Raw ADC data), which is a debug interface.
 *      User can disable this for additional power savings by commenting LVDS_STREAM macro definition in mmw_mss.h file.
 *
 *
 *  @subsection fsdyn Power Reduction Techniques: Frequency Scaling Dynamic
 *
 *  **DSS**
 *
 *     In typical applications, the DSP computations for one measurement cycle (or frame) get completed well before the computations for the next one start.
 *     In the intermittent time, the DSP clock rate can be reduced from the normal rates to the XTAL frequency to keep it active and responsive to interrupts, while saving power.
 *     The frequency may be lowered either by:\n
 *        a. Changing the clock divider value,\n
 *        b. Switching the clock source (to a lower frequency clock like XTAL)
 *
 *     Feature | Application
 *     :-----------------------------------|:-----------------------------------:
 *     Window of operation : transition to IDLE mode | After the processing operation of the core is complete.
 *     Power save Measures in IDLE mode | The overall active power consumption is reduced due to lowered frequency of operation.
 *     Allocation | API Invoked and Implemented by Customer Application
 *     References | Reference APIs in the SDK: powerMeasDspStateAfterFrameProc
 *
 *     **SDK feature**:
 *
 *      - **DSS Under clock after Frame Processing**: DSP clock is switched to XTAL after AoA processing is completed.
 *        It is switched back to normal rate before Doppler processing of the next frame.
 *
 *  Note: DSP underclocking affects the DSS_L2 memory access speed. As M4 core accesses scratch buffers placed in DSS_L2 during Doppler processing,
 *  DSP clock is switched back to normal rate before Doppler processing.
 *  For further power saving, it is possible to switch DSP clock back to normal rate just before AoA processing on DSP,
 *  provided that scratch buffers are placed in different memory (like MSS_L2) to avoid timing impact.
 *
 *  **BSS**
 *
 *     The TI Firmware supports dynamic frequency scaling by enabling the "BSS Underclocking feature" . This would require:
 *     - performing the necessary clock configurations of the RSS clock and FRC clock sources
 *     - reserving MSS RTIC for BSS use for maintaining ticks across the modes
 *     - handling the functional safety aspects of FRC and WDT
 *          - This can be achieved by performing a "Logical monitoring of the Frame timing" in the application
 *     - Enabling the feature and necessary configurations before unhalting the BSS core
 *          - Refer to the DFP ICD for more details
 *          - Refer to SDK User Guide section 3.8 Default SBL Clock configuration
 *
 *  **SDK feature**:
 *
 *     - **BSS Dynamic clocking**: The BSS clock source switches to XTAL clock when the core is idle. This feature is enabled in the SBL before unhalting BSS. Hence this feature cannot be configured through CLI.
 *     Refer SOC_rcmPopulateBSSControl API in "mcu_plus_sdk_awr2x44p_<ver>\source\drivers\soc\awr2x44p\soc_rcm.c" to enable or disable BSS Dynamic clocking feature. Note: MSS RTIC is used by the BSS when Dynamic clocking feature is enabled.
 *
 *  @subsection cgDyn Power Reduction Techniques : Clock Gating Dynamic
 *
 *     Dyanamic clock gating is divided into 2 categories:
 *     1. Clock gating of processing cores by invocation of WFI instruction
 *          - Applicable cores:
 *              - MSS ARM Cortex R5F
 *              - HSM ARM Cortex M4F
 *              - DSS ARM Cortex M4F
 *          - The HW based clock gating of the logic is supported. Any interrupt would ungate the logic and wakeup the core.
 *     2. Explicit clock gating of modules by the control core
 *          - The clock gating and ungating is controlled by a control processing core.
 *
 *     Feature | Application
 *     :-----------------------------------|:-----------------------------------:
 *     Window of operation : transition to IDLE mode | The clock gating is applied to the core or module during the IDLE periods and ungated during the active windows.
 *     Power save Measures in IDLE mode | Clock gating of processing cores by invocation of WFI instruction. Explicit clock gating of modules by the control core
 *     Allocation | API Invoked and Implemented by Customer Application
 *     References | Reference APIs in the SDK: powerMeasHwaDynamicClockGating, powerMeasHwaStateAfterFrameProc
 *
 *     **SDK feature**:
 *
 *      - **HWA Dynamic clock gating**: It enables the capability to clock gate the 4 Radar Accelerator core IPs (FFT datapath,CFAR,Memory compression,Local Maxima)
 *      based on the ParamSet being executed.
 *
 *      - **HWA Clock gate after frame processing**: HWA clock is gated after the frame processing is completed. Clock is ungated in frame start ISR.
 *
 *  @subsection pg Power Reduction Techniques : Power Gating
 *
 *   @subsubsection dsppg DSP Dynamic power gating
 *
 *     In addition to dynamic under-clocking as explained earlier, the DSP's power supply can be gated off using a power-switch dedicated to the DSP.
 *     In comparison with the dynamic under-clocking explained earlier, dynamic power-gating needs more sophisticated implementation, consumes more state-transition time
 *     (a few micro-seconds more) but provides additional power saving.
 *
 *     Feature | Application
 *     :-----------------------------------|:-----------------------------------:
 *     Window of operation : transition to IDLE mode | After the processing operations of the C66x DSP for a given frame are complete.
 *     Power save Measures in IDLE mode | The overall active and leakage power consumption by the DSP are saved by power gating.
 *     Allocation | API Invoked and Implemented by Customer Application
 *     References | Reference APIs in the SDK: powerMeasDspStateAfterFrameProc;\n Documentation: mcu_plus_sdk_awr2x44p_<ver>/docs/api_guide_awr2x44p/KERNEL_DPL_C66_CONTEXT_SAVE_RESTORE_PAGE.html
 *
 *     **SDK feature**:
 *
 *     - **DSS Power gate after Frame Processing**: After C66x completes AoA processing, the DSP core is powered down after saving context. DSP_PD_TRIGGER_WAKUP is configured as wakeup source.
 *     The M4 core wakes up the DSP core before Doppler processing of the next frame. DSP wakes up and resumes execution after restoring the context.
 *
 *     Note:\n
 *     1. The contents of the L1 memory are not retained when powered down. Hence, it is recommended to use the L1 memories as cache and writeback the dirty lines before powering down the DSP.\n
 *     2. L2 memory contents are retained, but it cannot be acccessed by other cores (R5F and HWA_CM4) when DSP is powered down.
 *     3. When DSP is power gated, M4 cannot access DSS_L2 memory which is needed for Doppler processing as the scratch buffers are placed in DSS_L2.
 *        For further power saving, DSP can be powered up just before AoA processing on DSP, provided the scratch buffers are moved to a different memory
 *        (like MSS_L2) by updating linker file.
 *
 *   @subsection timing Power Reduction Techniques : Timing Diagram
 *
 *      @image html powerOptTiming.png "Power Optimization Timing Diagram"
 *
 *  @section demoDesignNotes Design Notes
 *
 *  Due to the limitation of DPM local queue size, for DPM functions, semaphores are used to sync between calling task and function
 *  @ref MmwDemo_DPC_ObjectDetection_reportFxn. So that it won't cause DPM crash because of running out of
 *  DPM local queues. The following diagram demonstrates the example calling flow for blocking @ref DPM_ioctl function call.
 *
 *  @image html dpm_ioctl_handling.png "DPM_ioctl calling flow"
 *
 *  There are DPM report functions on both MSS and DSS_CM4 for the same DPM_Report. However, the sequence is not guaranteed between the
 *  two cores.
 *
 *  @section memoryUsage Memory Usage
 *  @subsection memUsageSummary Memory usage summary
 *    For the binary running on the M4 core, two sections are reserved in DSS_L3 RAM called DSS_L3_REUSABLE and DSS_L3_BSS.
 *    DSS_L3_REUSABLE is used to stored functions which can be overwritten by the RADAR cube and scratch buffers during processing,
 *    i.e. which run only during the initialization phase. DSS_L3_BSS section stores the global DPC object if there isn't enough space in the M4 RAM.
 *    The user can check in the generated map file for the M4 core binary if this section is empty, in which case
 *    they can reduce it's size to zero in the platform M4 linker command files and update the macro DSS_L3_U_SIZE in the dss_cm4_main.c file.
 *    Please refer to the respective .map files in the demo folder for details on memory usage.
 *
 *  @section error Note on Error Codes
 *
 *    ------------------------------
 *    When demo runs into error conditions, an error code will be generated and printed out.
 *    Error code is defined as a negative integer. It comes from the following categories:
 *    - Drivers
 *    - Control modules
 *    - Data Processing Chain
 *    - Data Processing Unit
 *    - Demo
 *
 *    The error code is defined as (Module  error code base - Module specific error code). \n
 *    The base error code for the above modules can be found in @ref mmwave_error.h \n
 *    The base error code for DPC and DPU can be found in @ref dp_error.h
 *
 *    Module specific error code is specified in the module's header file.
 *    Examples:
 *       - UART driver error code is defined in uart.h
 *       - DPC error code is defined in the dpc used in demo
 *
 *  @subsection mmwave_error mmWave module Error Code
 *    Error code from mmWave module is encoded in the following manner:
 *
 *    Bits(31::16)  |  Bits(15::2)   | Bits (1::0)
 *    :-------------|:----------------|:-------------:
 *    mmwave  error  | Subsystem error   | error level
 *
 *    - mmwave error is defined in mmwave.h \n
 *    - Subsystem error is returned from sub-system such as mmwavelink and mailbox driver. \n
 *    - Error level is referred as WARNING level and ERROR level.
 *    - mmWave exposes an API - MMWave_decodeError() that can be used in demo to decode error code
 *
 *  @subsection mmwave_error_example Example
 *    - Here is an example on how to parse the error code - "-40111"\n
 *        -# The error code is from module with error base "-40000", which indicates it is DPC error.\n
 *        -# By referring to @ref dp_error.h, base "-40100" is from HWA based objectdetection DPC.\n
 *        -# Then find the error code in objectdetection.h for error(-11) as DPC_OBJECTDETECTION_ENOMEM__L3_RAM_RADAR_CUBE
 *
 *    - Another example is with mmWave control module: - "mmWave Config failed [Error code: -3109 Subsystem: 71]"\n
 *        -# The above message indicates the error is from module(-3100 ->mmwave) with error -9(MMWAVE_ECHIRPCFG)\n
 *        -# The subsystem(mmwavelink) error is 71(RL_RET_CODE_CHIRP_TX_ENA_1INVAL_IN) which can be found in mmwavelink.h
 *
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* MCU+SDK include files. */
#include <drivers/uart.h>
#include <kernel/dpl/CacheP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/CycleCounterP.h>
#include <kernel/dpl/AddrTranslateP.h>
#include <kernel/dpl/DebugP.h>
#include "FreeRTOS.h"
#include "task.h"

/* mmWave SDK Include Files: */
#include <ti/common/syscommon.h>
#include <ti/common/mmwavesdk_version.h>
#include <ti/control/mmwave/mmwave.h>

#include <ti/utils/cli/cli.h>
#include <ti/utils/mathutils/mathutils.h>
#include <ti/utils/testlogger/logger.h>

/* Demo Include Files */
#include <ti/demo/awr2x44P/mmw_ddm/include/mmw_config.h>
#include <ti/demo/utils/mmwdemo_rfparser.h>
#include <ti/demo/utils/mmwdemo_adcconfig.h>
#include <ti/demo/utils/mmwdemo_monitor.h>
#include <ti/demo/utils/enet_stream.h>
#include <ti/demo/awr2x44P/mmw_ddm/mmw_resDDM.h>
#include <ti/demo/awr2x44P/mmw_ddm/mmw_common.c>
#include <ti/demo/awr2x44P/mmw_ddm/mss/mmw_mss.h>
#include <ti/demo/awr2x44P/mmw_ddm/include/mmw_output.h>
#include <ti/board/antenna_geometry.h>
#include <ti/demo/utils/mmwdemo_flash.h>
/* Profiler Include Files */
#include <kernel/dpl/CycleCounterP.h>

/**
 * @brief Task Priority settings:
 * Mmwave task is at higher priority because of potential async messages from BSS
 * that need quick action in real-time.
 *
 * CLI task must be at a lower priority than object detection
 * dpm task priority because the dynamic CLI command handling in the objection detection
 * dpm task assumes CLI task is held back during this processing. The alternative
 * is to use a semaphore between the two tasks.
 */

/* Demo tasks should have priority higher than enet/lwip tasks */
#ifdef ENET_STREAM
#define MMWDEMO_CLI_TASK_PRIORITY                 3
#define MMWDEMO_UART_EXPORT_TASK_PRIORITY         4
#define MMWDEMO_MMWAVE_CTRL_TASK_PRIORITY         5
#define MMWDEMO_MMWAVE_ENET_TASK_PRIORITY         2
#else
#define MMWDEMO_CLI_TASK_PRIORITY                 3
#define MMWDEMO_UART_EXPORT_TASK_PRIORITY         4
#define MMWDEMO_MMWAVE_CTRL_TASK_PRIORITY         5
#endif
#define DPC_OBJDET_INSTANCEID       (0xFEEDFEED)
extern SemaphoreP_Object objDataSemaphoreHandle;

/* These address offsets are in bytes, when configure address offset in hardware,
   these values will be converted to number of 128bits
   Buffer at offset 0x0U is reserved by BSS, hence offset starts from 0x200
 */
#define MMW_DEMO_CQ_SIGIMG_ADDR_OFFSET          0x200U
#define MMW_DEMO_CQ_RXSAT_ADDR_OFFSET           0x400U

/* CQ data is at 16 bytes alignment for mulitple chirps */
#define MMW_DEMO_CQ_DATA_ALIGNMENT            16U


#define MAX_MOD_FREQ_DIVIDER_MANTISSA         127U

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/
/* FreeRTOS Task declarations. */
#define MMWDEMO_INIT_TASK_PRI         (1U)

#define MMWDEMO_INIT_TASK_STACK_SIZE  (4*1024U)
#define MMWDEMO_MMWAVE_CTRL_TASK_STACK_SIZE (3*1024U)
#define MMWDEMO_UART_DATA_EXPORT_TASK_STACK_SIZE (4*1024U)
#ifdef ENET_STREAM
#define MMWDEMO_MMWAVE_ENET_TASK_STACK_SIZE (4*1024U)
#endif

/* Applictaon task stack variables */
StackType_t gAppMainTskStack[MMWDEMO_INIT_TASK_STACK_SIZE] __attribute__((aligned(32)));
StackType_t gMmwCtrlTskStack[MMWDEMO_MMWAVE_CTRL_TASK_STACK_SIZE] __attribute__((aligned(32)));
StackType_t gUartTskStack[MMWDEMO_UART_DATA_EXPORT_TASK_STACK_SIZE] __attribute__((aligned(32)));
#ifdef ENET_STREAM
StackType_t gMmwEnetTskStack[MMWDEMO_MMWAVE_ENET_TASK_STACK_SIZE] __attribute__((aligned(32)));
/* Variable to store detected object data for ethernet streaming */
MmwDemo_enetStreamObjData gEnetStreamObjData __attribute__((aligned(128)));
#endif

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_MSS_MCB    gMmwMssMCB;

/* RF scale factor that can be used to translate
 * RF frequency related (start frequency, frequency slope, frequency constant etc)
 * configuration expressed in user-friendly units (like GHz/MHz) into units
 * that are required for mmwavelink (and therefore MMWave) APIs related to
 * such frequency configuration. It depends on whether the device is 60 GHz
 * or 77 GHz device.*/
#define MMWDEMO_RF_FREQ_SCALE_FACTOR              3.6f

/* Calibration Data Save/Restore defines */
#define MMWDEMO_CALIB_FLASH_SIZE	              4096
#define MMWDEMO_CALIB_STORE_MAGIC            (0x7CB28DF9U)

/* ADC Data Dithering MACROS */
/* 1 LSB in "RSS_CTRL::ADCBUFCFG1_EXTD_ADCBUFINTGENDLY" = 3 Clocks = 20 ns Delay (SYS_CLK = 150MHz)*/
/* 1 LSB in "RSS_CTRL::ADCBUFCFG1_EXTD_ADCBUFINTGENDLY" = 4 Clocks = 20 ns Delay (SYS_CLK = 200MHz)*/
#define MMWDEMO_DITHERING_MINDELAY            55U

MmwDemo_calibData gCalibDataStorage __attribute__((aligned(8)));

static void MmwDemo_checkEdmaErrors(void);

/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/

extern void MmwDemo_CLIInit(uint8_t taskPriority);
extern MmwDemo_RFParserHwAttr MmwDemo_RFParserHwCfg;
DPC_ObjectDetection_ElevEstCfg gElevEstCfg = {0};

/**************************************************************************
 ************************* Millimeter Wave Demo Functions prototype *************
 **************************************************************************/

/* MMW demo functions for datapath operation */
static int32_t MmwDemo_dataPathConfig (void);
static void MmwDemo_dataPathStart (void);
static void MmwDemo_dataPathStop (void);
void MmwDemo_handleObjectDetResult(void);
void MmwDemo_DPC_ObjectDetection_reportFxn(void *data,  uint16_t dataLen);

static void MmwDemo_transmitProcessedOutput
(
    UART_Handle     uartHandle,
    DPC_ObjectDetection_ExecuteResult   *result,
    MmwDemo_output_message_stats        *timingInfo
);

static void MmwDemo_measurementResultOutput(void* compRxChanCfg);

static int32_t MmwDemo_DPM_ioctl_blocking
(
    DPM_Handle handle,
    uint32_t cmd,
    void* arg,
    uint32_t argLen
);

/* Mmwave demo init functions */
static void MmwDemo_initTask(void* args);
static void MmwDemo_platformInit(MmwDemo_platformCfg *config);
static bool MmwDemo_BoardInit(void);

/* Mmwave control functions */
static void MmwDemo_mmWaveCtrlTask(void* args);
static int32_t MmwDemo_mmWaveCtrlStop (void);
static int32_t MmwDemo_eventCallbackFxn(uint8_t devIndex, uint16_t msgId, uint16_t sbId, uint16_t sbLen, uint8_t *payload);
int32_t MmwDemo_getNumEmptySubBands(uint32_t numTxAntennas);

/* CQ config function. */
static int32_t MmwDemo_configCQ(MmwDemo_SubFrameCfg *subFrameCfg,
                                           uint8_t numChirpsPerChirpEvent,
                                           uint8_t validProfileIdx);

/* Calibration save/restore APIs */
static int32_t MmwDemo_calibInit(void);
static int32_t MmwDemo_calibSave(MmwDemo_calibDataHeader *ptrCalibDataHdr, MmwDemo_calibData  *ptrCalibrationData);
static int32_t MmwDemo_calibRestore(MmwDemo_calibData  *calibrationData);

volatile uint32_t transmitStartTime =0;
/**************************************************************************
 ************************* Millimeter Wave Demo Functions **********************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *      Send assert information through CLI.
 */
void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line)
{
    if (!expression) {
        CLI_write ("Exception: %s, line %d.\n",file,line);
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to set the pending state of configuration.
 *
 *  @param[in] subFrameCfg Pointer to Sub-frame specific configuration
 *  @param[in] offset Configuration structure offset that uniquely identifies the
 *                    configuration to set to the pending state.
 *
 *  @retval None
 */
static void MmwDemo_setSubFramePendingState(MmwDemo_SubFrameCfg *subFrameCfg, uint32_t offset)
{
    switch (offset)
    {
        case MMWDEMO_GUIMONSEL_OFFSET:
            // Do nothing
        break;
        case MMWDEMO_CFARDOPPLERCFG_OFFSET:
            subFrameCfg->datapathStaticCfg.isCfarCfgPending = 1;
        break;
        case MMWDEMO_FOVAOA_OFFSET:
            subFrameCfg->datapathStaticCfg.isFovAoaCfgPending = 1;
        break;
        case MMWDEMO_CFARCFGRANGE_OFFSET:
            subFrameCfg->datapathStaticCfg.isRangeCfarCfgPending = 1;
        break;
        case MMWDEMO_COMPRESSIONCFG_OFFSET:
            subFrameCfg->datapathStaticCfg.isCompressionCfgPending = 1;
        break;
        case MMWDEMO_INTFMITIGCFG_OFFSET:
            subFrameCfg->datapathStaticCfg.isIntfStatsdBCfgPending = 1;
        break;
        case MMWDEMO_LOCALMAXCFG_OFFSET:
            subFrameCfg->datapathStaticCfg.isLocalMaxCfgPending = 1;
        break;
        case MMWDEMO_ADCBUFCFG_OFFSET:
            subFrameCfg->isAdcBufCfgPending = 1;
        break;
#ifdef LVDS_STREAM
        case MMWDEMO_LVDSSTREAMCFG_OFFSET:
            subFrameCfg->isLvdsStreamCfgPending = 1;
        break;
#endif
        default:
            MmwDemo_debugAssert(0);
        break;
    }
}

/**
 *  @b Description
 *  @n
 *      Resets (clears) all pending common configuration of Object Detection DPC
 *
 *  @param[in] cfg Object Detection DPC common configuration
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_resetDynObjDetCommonCfgPendingState(MmwDemo_DPC_ObjDet_CommonCfg *cfg)
{
    cfg->isAntennaCalibParamCfgPending = 0;
}

/**
 *  @b Description
 *  @n
 *      Utility function to apply configuration to specified sub-frame
 *
 *  @param[in] srcPtr Pointer to configuration
 *  @param[in] offset Offset of configuration within the parent structure
 *  @param[in] size   Size of configuration
 *  @param[in] subFrameNum Sub-frame Number (0 based) to apply to, broadcast to
 *                         all sub-frames if special code MMWDEMO_SUBFRAME_NUM_FRAME_LEVEL_CONFIG
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void MmwDemo_CfgUpdate(void *srcPtr, uint32_t offset, uint32_t size, int8_t subFrameNum)
{
    /* if subFrameNum undefined, broadcast to all sub-frames */
    if(subFrameNum == MMWDEMO_SUBFRAME_NUM_FRAME_LEVEL_CONFIG)
    {
        uint8_t  indx;
        for(indx = 0; indx < RL_MAX_SUBFRAMES; indx++)
        {
            memcpy((void *)((uint32_t) &gMmwMssMCB.subFrameCfg[indx] + offset), srcPtr, size);
            MmwDemo_setSubFramePendingState(&gMmwMssMCB.subFrameCfg[indx], offset);
        }
    }
    else
    {
        /* Apply configuration to specific subframe (or to position zero for the legacy case
           where there is no advanced frame config) */
        memcpy((void *)((uint32_t) &gMmwMssMCB.subFrameCfg[subFrameNum] + offset), srcPtr, size);
        MmwDemo_setSubFramePendingState(&gMmwMssMCB.subFrameCfg[subFrameNum], offset);
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to get temperature report from front end and
 *      save it in global structure.
 *
 *  @retval None
 */
void MmwDemo_getTemperatureReport()
{
    /* Get Temerature report */
    gMmwMssMCB.temperatureStats.tempReportValid = rlRfGetTemperatureReport(RL_DEVICE_MAP_INTERNAL_BSS,
                        (rlRfTempData_t*)&gMmwMssMCB.temperatureStats.temperatureReport);
}



/**************************************************************************
 ******************** Millimeter Wave Demo Results Transmit Functions *************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      Sends the calibration Range Bias (TDM) and Rx Channel Gain/Phase Measurement
 *      and Compensation info through CLI
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_measurementResultOutput(void *compRxChanCfg)
{
    int32_t i;
    Measure_compRxChannelBiasCfg *result = (Measure_compRxChannelBiasCfg*)compRxChanCfg;
    CLI_write ("compRxChanPhase (Im-Re) \n");
    CLI_write("range %.5f peakVal %d \n", result->targetRange, result->peakVal);

    /* Send the received calibration info through CLI */
    for (i = 0; i < SYS_COMMON_NUM_TX_ANTENNAS*SYS_COMMON_NUM_RX_CHANNEL; i++)
    {
        CLI_write ("%.5f ", (float)result->rxChPhaseComp[i].imag/16384.);
        CLI_write ("%.5f ", (float)result->rxChPhaseComp[i].real/16384.);
    }
    CLI_write ("\n");

}


/** @brief Transmits detection data over UART
*
*    The following data is transmitted:
*    1. Header (size = 32bytes), including "Magic word", (size = 8 bytes)
*       and including the number of TLV items
*    TLV Items:
*    2. If detectedObjects flag is 1 or 2, DPIF_PointCloudCartesian structure containing
*       X,Y,Z location and velocity for detected objects,
*       size = sizeof(DPIF_PointCloudCartesian) * number of detected objects
*    3. If detectedObjects flag is 1, DPIF_PointCloudSideInfo structure containing SNR
*       and noise for detected objects,
*       size = sizeof(DPIF_PointCloudCartesian) * number of detected objects
*    4. If logMagRange flag is set,  rangeProfile,
*       size = number of range bins * sizeof(uint16_t)
*    5. If noiseProfile flag is set,  noiseProfile,
*       size = number of range bins * sizeof(uint16_t)
*    6. If rangeAzimuthHeatMap flag is set, the zero Doppler column of the
*       range cubed matrix, size = number of Rx Azimuth virtual antennas *
*       number of chirps per frame * sizeof(uint32_t)
*    7. If rangeDopplerHeatMap flag is set, the log magnitude range-Doppler matrix,
*       size = number of range bins * number of Doppler bins * sizeof(uint16_t)
*    8. If statsInfo flag is set, the stats information
*   @param[in] uartHandle   UART driver handle
*   @param[in] result       Pointer to result from object detection DPC processing
*   @param[in] timingInfo   Pointer to timing information provided from core that runs data path
*/
static void MmwDemo_transmitProcessedOutput
(
    UART_Handle     uartHandle,
    DPC_ObjectDetection_ExecuteResult   *result,
    MmwDemo_output_message_stats        *timingInfo
)
{
    MmwDemo_output_message_header header;
    MmwDemo_GuiMonSel   *pGuiMonSel;
    MmwDemo_SubFrameCfg *subFrameCfg;
    uint32_t tlvIdx = 0;
    uint32_t index;
    uint32_t numPaddingBytes;
    uint32_t packetLen;
    uint8_t padding[MMWDEMO_OUTPUT_MSG_SEGMENT_LEN];
    MmwDemo_output_message_tl   tl[MMWDEMO_OUTPUT_MSG_MAX];
    uint16_t *detMatrix = (uint16_t *)result->detMatrix.data;
    DPIF_PointCloudCartesian *objOut;
    DPIF_PointCloudSideInfo *objOutSideInfo;
    DPC_ObjectDetection_Stats *stats;
    UART_Transaction trans;

    UART_Transaction_init(&trans);

    /* Get subframe configuration */
    subFrameCfg = &gMmwMssMCB.subFrameCfg[result->subFrameIdx];
    uint8_t txAntMask     = gMmwMssMCB.cfg.openCfg.chCfg.txChannelEn;
    uint8_t numTxAnt      = mathUtils_countSetBits(txAntMask);
    uint16_t numDopFFTSubBins = subFrameCfg->numDopplerBins / (numTxAnt + gMmwMssMCB.numEmptySubBands);

    /* Get Gui Monitor configuration */
    pGuiMonSel = &subFrameCfg->guiMonSel;

    /* Clear message header */
    memset((void *)&header, 0, sizeof(MmwDemo_output_message_header));

    /******************************************************************
       Send out data that is enabled, Since processing results are from DSP,
       address translation is needed for buffer pointers
    *******************************************************************/
    {
        detMatrix = (uint16_t *) AddrTranslateP_getLocalAddr((uint32_t)detMatrix);

        objOut = (DPIF_PointCloudCartesian *) AddrTranslateP_getLocalAddr((uint32_t)result->objOut);

        objOutSideInfo = (DPIF_PointCloudSideInfo *) AddrTranslateP_getLocalAddr((uint32_t)result->objOutSideInfo);

        stats = (DPC_ObjectDetection_Stats *) AddrTranslateP_getLocalAddr((uint32_t)result->stats);
    }

    /* Header: */
    header.platform =  0x2944;
    header.magicWord[0] = 0x0102;
    header.magicWord[1] = 0x0304;
    header.magicWord[2] = 0x0506;
    header.magicWord[3] = 0x0708;
    header.numDetectedObj = result->numObjOut;
    header.version =    MMWAVE_SDK_VERSION_BUILD |
                        (MMWAVE_SDK_VERSION_BUGFIX << 8) |
                        (MMWAVE_SDK_VERSION_MINOR << 16) |
                        (MMWAVE_SDK_VERSION_MAJOR << 24);

    packetLen = sizeof(MmwDemo_output_message_header);
    if (((pGuiMonSel->detectedObjects == 1) || (pGuiMonSel->detectedObjects == 2)) &&
         (result->numObjOut > 0))
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_DETECTED_POINTS;
        tl[tlvIdx].length = sizeof(DPIF_PointCloudCartesian) * result->numObjOut;
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    /* Side info */
    if ((pGuiMonSel->detectedObjects == 1) && (result->numObjOut > 0))
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO;
        tl[tlvIdx].length = sizeof(DPIF_PointCloudSideInfo) * result->numObjOut;
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    if (pGuiMonSel->logMagRange)
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_RANGE_PROFILE;
        tl[tlvIdx].length = sizeof(uint16_t) * subFrameCfg->numRangeBins;
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    if (pGuiMonSel->noiseProfile)
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_NOISE_PROFILE;
        tl[tlvIdx].length = sizeof(uint16_t) * subFrameCfg->numRangeBins;
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    if (pGuiMonSel->rangeDopplerHeatMap)
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP;
        tl[tlvIdx].length = subFrameCfg->numRangeBins * numDopFFTSubBins * sizeof(uint16_t);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }
    if (pGuiMonSel->statsInfo)
    {
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_STATS;
        tl[tlvIdx].length = sizeof(MmwDemo_output_message_stats);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;

        MmwDemo_getTemperatureReport();
        tl[tlvIdx].type = MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS;
        tl[tlvIdx].length = sizeof(MmwDemo_temperatureStats);
        packetLen += sizeof(MmwDemo_output_message_tl) + tl[tlvIdx].length;
        tlvIdx++;
    }

    header.numTLVs = tlvIdx;
    /* Round up packet length to multiple of MMWDEMO_OUTPUT_MSG_SEGMENT_LEN */
    header.totalPacketLen = MMWDEMO_OUTPUT_MSG_SEGMENT_LEN *
            ((packetLen + (MMWDEMO_OUTPUT_MSG_SEGMENT_LEN-1))/MMWDEMO_OUTPUT_MSG_SEGMENT_LEN);
    header.timeCpuCycles = 0; //Pmu_getCount(0);
    header.frameNumber = stats->frameStartIntCounter;
    header.subFrameNumber = result->subFrameIdx;

    DebugP_logInfo("Platform = %d, Version = %d, NumObj = %d, numTLVs = %d", header.platform, header.version, header.numDetectedObj, header.numTLVs);

    CacheP_wbInv((void *)&header, sizeof(MmwDemo_output_message_header), CacheP_TYPE_ALLD);
    UART_Transaction_init(&trans);
    trans.buf   = (uint8_t*)&header;
    trans.count = sizeof(MmwDemo_output_message_header);
    UART_write(uartHandle, &trans);

    tlvIdx = 0;
    /* Send detected Objects */
    if (((pGuiMonSel->detectedObjects == 1) || (pGuiMonSel->detectedObjects == 2)) &&
        (result->numObjOut > 0))
    {
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        /*Send array of objects */
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)objOut;
        trans.count = sizeof(DPIF_PointCloudCartesian) * result->numObjOut;
        UART_write(uartHandle, &trans);
        tlvIdx++;
    }

    /* Send detected Objects Side Info */
    if ((pGuiMonSel->detectedObjects == 1) && (result->numObjOut > 0))
    {

        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)objOutSideInfo;
        trans.count = sizeof(DPIF_PointCloudSideInfo) * result->numObjOut;
        UART_write(uartHandle, &trans);
        tlvIdx++;
    }

    /* Send Range profile */
    if (pGuiMonSel->logMagRange)
    {
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        for(index = 0; index < subFrameCfg->numRangeBins; index++)
        {
            UART_Transaction_init(&trans);
            trans.buf   = (uint8_t*)&detMatrix[index * numDopFFTSubBins];
            trans.count = sizeof(uint16_t);
            UART_write(uartHandle, &trans);
        }
        tlvIdx++;
    }

    /* Send noise profile */
    if (pGuiMonSel->noiseProfile)
    {
        uint32_t maxDopIdx = subFrameCfg->numDopplerBins/2 -1;
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        for(index = 0; index < subFrameCfg->numRangeBins; index++)
        {
            UART_Transaction_init(&trans);
            trans.buf   = (uint8_t*)&detMatrix[index*subFrameCfg->numDopplerBins + maxDopIdx];
            trans.count = sizeof(uint16_t);
            UART_write(uartHandle, &trans);
        }
        tlvIdx++;
    }

    /* Send data for range/Doppler heatmap */
    if (pGuiMonSel->rangeDopplerHeatMap == 1)
    {
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)detMatrix;
        trans.count = tl[tlvIdx].length;
        UART_write(uartHandle, &trans);
        tlvIdx++;
    }

    /* Send stats information */
    if (pGuiMonSel->statsInfo == 1)
    {
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);

        /* Address translation is done when buffer is received*/
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)timingInfo;
        trans.count = tl[tlvIdx].length;
        UART_write(uartHandle, &trans);
        tlvIdx++;
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&tl[tlvIdx];
        trans.count = sizeof(MmwDemo_output_message_tl);
        UART_write(uartHandle, &trans);
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)&gMmwMssMCB.temperatureStats;
        trans.count = tl[tlvIdx].length;
        UART_write(uartHandle, &trans);
        tlvIdx++;
    }

    /* Send padding bytes */
    numPaddingBytes = MMWDEMO_OUTPUT_MSG_SEGMENT_LEN - (packetLen & (MMWDEMO_OUTPUT_MSG_SEGMENT_LEN-1));
    if (numPaddingBytes<MMWDEMO_OUTPUT_MSG_SEGMENT_LEN)
    {
        UART_Transaction_init(&trans);
        trans.buf   = (uint8_t*)padding;
        trans.count = numPaddingBytes;
        UART_write(uartHandle, &trans);
    }

}

/**************************************************************************
 ******************** Millimeter Wave Demo control path Functions *****************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *      The function is used to trigger the Front end to stop generating chirps.
 *
 *  @retval
 *      Not Applicable.
 */
static int32_t MmwDemo_mmWaveCtrlStop (void)
{
    int32_t                 errCode = 0;

    DebugP_logInfo("App: Issuing MMWave_stop\n");

    /* Stop the mmWave module: */
    if (MMWave_stop (gMmwMssMCB.ctrlHandle, &errCode) < 0)
    {
        MMWave_ErrorLevel   errorLevel;
        int16_t             mmWaveErrorCode;
        int16_t             subsysErrorCode;

        /* Error/Warning: Unable to stop the mmWave module */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
        if (errorLevel == MMWave_ErrorLevel_ERROR)
        {
            /* Error: Display the error message: */
            DebugP_log ("Error: mmWave Stop failed [Error code: %d Subsystem: %d]\n",
                            mmWaveErrorCode, subsysErrorCode);

            /* Not expected */
            MmwDemo_debugAssert(0);
        }
        else
        {
            /* Warning: This is treated as a successful stop. */
            DebugP_log ("mmWave Stop error ignored [Error code: %d Subsystem: %d]\n",
                            mmWaveErrorCode, subsysErrorCode);
        }
    }

    return errCode;
}

/**
 *  @b Description
 *  @n
 *      The task is used to provide an execution context for the mmWave
 *      control task
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_mmWaveCtrlTask(void* args)
{
    int32_t errCode;

    while (1)
    {
        /* Execute the mmWave control module: */
        if (MMWave_execute (gMmwMssMCB.ctrlHandle, &errCode) < 0)
        {
            MmwDemo_debugAssert (0);
        }
    }
}

/**************************************************************************
 ******************** Millimeter Wave Demo data path Functions *******************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      Help function to make DPM_ioctl blocking until response is reported
 *
 *  @retval
 *      Success         -0
 *      Failed          <0
 */
static int32_t MmwDemo_DPM_ioctl_blocking
(
    DPM_Handle handle,
    uint32_t cmd,
    void* arg,
    uint32_t argLen
)
{
    int32_t retVal = 0;
    switch(cmd)
    {
        case DPC_OBJDET_IOCTL__STATIC_PRE_START_COMMON_CFG:
        case DPC_OBJDET_IOCTL__STATIC_PRE_START_CFG:
            retVal = DPM_send(handle, (void *)arg, (uint16_t)argLen, CSL_CORE_ID_M4SS0_1);
            break;
        case DPC_OBJDET_IOCTL__STATIC_PRE_START_ELEVEST_CFG:
            retVal = DPM_send(handle, (void *)arg, (uint16_t)argLen, CSL_CORE_ID_C66SS0);
            break;
    }
    if(retVal == 0)
    {
        /* Wait until ioctl completed */
        SemaphoreP_pend(&gMmwMssMCB.DPMioctlSemHandle, SystemP_WAIT_FOREVER);
    }

    return(retVal);
}

/**
 *  @b Description
 *  @n
 *      Perform Data path driver open
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dataPathOpen(void)
{
    gMmwMssMCB.adcBufHandle = MmwDemo_ADCBufOpen();
    if(gMmwMssMCB.adcBufHandle == NULL)
    {
        MmwDemo_debugAssert(0);
    }
}

/**
 *  @b Description
 *  @n
 *      Function to configure CQ.
 *
 *  @param[in] subFrameCfg Pointer to sub-frame config
 *  @param[in] numChirpsPerChirpEvent number of chirps per chirp event
 *  @param[in] validProfileIdx valid profile index
 *
 *  @retval
 *      0 if no error, else error (there will be system prints for these).
 */
static int32_t MmwDemo_configCQ(MmwDemo_SubFrameCfg *subFrameCfg,
                                uint8_t numChirpsPerChirpEvent,
                                uint8_t validProfileIdx)
{
    MmwDemo_AnaMonitorCfg*      ptrAnaMonitorCfg;
    ADCBuf_CQConf               cqConfig;
    rlRxSatMonConf_t*           ptrSatMonCfg;
    rlSigImgMonConf_t*          ptrSigImgMonCfg;
    int32_t                     retVal;
    uint16_t                    cqChirpSize;

    /* Get analog monitor configuration */
    ptrAnaMonitorCfg = &gMmwMssMCB.anaMonCfg;

    /* Config mmwaveLink to enable Saturation monitor - CQ2 */
    ptrSatMonCfg = &gMmwMssMCB.cqSatMonCfg[validProfileIdx];

    if (ptrAnaMonitorCfg->rxSatMonEn)
    {
        if (ptrSatMonCfg->profileIndx != validProfileIdx)
        {
            DebugP_log ("Error: Saturation monitoring (globally) enabled but not configured for profile(%d)\n",
                           validProfileIdx);
            MmwDemo_debugAssert(0);
        }

        retVal = mmwDemo_cfgRxSaturationMonitor(ptrSatMonCfg);
        if(retVal != 0)
        {
            DebugP_log ("Error: rlRfRxIfSatMonConfig returns error = %d for profile(%d)\n",
                           retVal, ptrSatMonCfg->profileIndx);
            goto exit;
        }
    }

    /* Config mmwaveLink to enable Saturation monitor - CQ1 */
    ptrSigImgMonCfg = &gMmwMssMCB.cqSigImgMonCfg[validProfileIdx];

    if (ptrAnaMonitorCfg->sigImgMonEn)
    {
        if (ptrSigImgMonCfg->profileIndx != validProfileIdx)
        {
            DebugP_log ("Error: Sig/Image monitoring (globally) enabled but not configured for profile(%d)\n",
                           validProfileIdx);
            MmwDemo_debugAssert(0);
        }

        retVal = mmwDemo_cfgRxSigImgMonitor(ptrSigImgMonCfg);
        if(retVal != 0)
        {
            DebugP_log ("Error: rlRfRxSigImgMonConfig returns error = %d for profile(%d)\n",
                           retVal, ptrSigImgMonCfg->profileIndx);
            goto exit;
        }
    }

    retVal = mmwDemo_cfgAnalogMonitor(ptrAnaMonitorCfg);
    if (retVal != 0)
    {
        DebugP_log ("Error: rlRfAnaMonConfig returns error = %d\n", retVal);
        goto exit;
    }

    if(ptrAnaMonitorCfg->rxSatMonEn || ptrAnaMonitorCfg->sigImgMonEn)
    {
        /* CQ driver config */
        memset((void *)&cqConfig, 0, sizeof(ADCBuf_CQConf));
        cqConfig.cqDataWidth = 0; /* 16bit for mmw demo */
        cqConfig.cq1AddrOffset = MMW_DEMO_CQ_SIGIMG_ADDR_OFFSET; /* CQ1 starts from the beginning of the buffer */
        cqConfig.cq2AddrOffset = MMW_DEMO_CQ_RXSAT_ADDR_OFFSET;  /* Address should be 16 bytes aligned */

        retVal = ADCBuf_control(gMmwMssMCB.adcBufHandle, ADCBufMMWave_CMD_CONF_CQ, (void *)&cqConfig);
        if (retVal < 0)
        {
            DebugP_log ("Error: MMWDemoDSS Unable to configure the CQ\n");
            MmwDemo_debugAssert(0);
        }
    }

    if (ptrAnaMonitorCfg->sigImgMonEn)
    {
        /* This is for 16bit format in mmw demo, signal/image band data has 2 bytes/slice
           For other format, please check DFP interface document
         */
        cqChirpSize = (ptrSigImgMonCfg->numSlices + 1) * sizeof(uint16_t);
        cqChirpSize = MATHUTILS_ROUND_UP_UNSIGNED(cqChirpSize, MMW_DEMO_CQ_DATA_ALIGNMENT);
        subFrameCfg->sigImgMonTotalSize = cqChirpSize * numChirpsPerChirpEvent;
    }

    if (ptrAnaMonitorCfg->rxSatMonEn)
    {
        /* This is for 16bit format in mmw demo, saturation data has one byte/slice
           For other format, please check DFP interface document
         */
        cqChirpSize = (ptrSatMonCfg->numSlices + 1) * sizeof(uint8_t);
        cqChirpSize = MATHUTILS_ROUND_UP_UNSIGNED(cqChirpSize, MMW_DEMO_CQ_DATA_ALIGNMENT);
        subFrameCfg->satMonTotalSize = cqChirpSize * numChirpsPerChirpEvent;
    }

exit:
    return(retVal);
}

/**
 *  @b Description
 *  @n
 *      Utility function to convert the CFAR threshold
 *      from a CLI encoded dB value to a linear value
 *      as expected by the Doppler DPU
 *
 *  @param[in] codedCfarVal CFAR SNR threshold in dB as encoded in the CLI
 *
 *  @retval
 *      CFAR threshold in linear format
 */
static uint16_t MmwDemo_convertCfarToLinear(uint16_t codedCfarVal)
{
    uint16_t linearVal;
    float    dbVal, linVal;

    /* dbVal is a float value from 0-100dB. It needs to
    be converted to linear scale..
    First, recover float dbVal that was encoded in CLI. */
    dbVal = (float)(codedCfarVal / MMWDEMO_CFAR_THRESHOLD_ENCODING_FACTOR);

    /* Now convert it to linear value */
    linVal = log2(pow(10, (dbVal / 20.0))) * (1 << 11) + 0.5;

    linearVal = (uint16_t) linVal;
    return (linearVal);
}

/**
 *  @b Description
 *  @n
 *      Utility function to convert the CFAR threshold
 *      from a CLI encoded dB value to a log2 value
 *      as expected by the Range CFAR DPU
 *
 *  @param[in] codedCfarVal CFAR SNR threshold in dB as encoded in the CLI
 *  @param[in] numBands Total number of subbands
 *
 *  @retval
 *      CFAR threshold in dB format
 */
static uint16_t MmwDemo_convertRangeCfarToThresh(uint16_t codedCfarVal, uint8_t numBands)
{
    uint16_t linearVal;
    float    dbVal, linVal;
    uint32_t defaultScaling = 1 << 11;
    float additionalScaling =  numBands / (float)(1 << mathUtils_ceilLog2(numBands));

    /* dbVal is a float value from 0-100dB. It needs to
    be converted to linear scale..
    First, recover float dbVal that was encoded in CLI. */
    dbVal = (float)(codedCfarVal / MMWDEMO_CFAR_THRESHOLD_ENCODING_FACTOR);

    /* Now convert it to linear value */
    linVal = (uint32_t)(log2f(pow(10, (float)dbVal/20.0)) * additionalScaling * defaultScaling + 0.5);

    linearVal = (uint16_t) linVal;
    return (linearVal);
}

/**
 *  @b Description
 *  @n
 *      Utility function to convert the CFAR threshold
 *      from a CLI encoded dB value to a log2 value
 *      as expected by the doppler CFAR DPU
 *
 *  @param[in] codedCfarVal CFAR SNR threshold in dB as encoded in the CLI
 *  @param[in] numBands Total number of subbands
 *
 *  @retval
 *      CFAR threshold in dB format
 */
#define CONST_LOG2_10  (3.3219)
static uint16_t MmwDemo_convertDopplerCfarToThresh(uint16_t codedCfarVal)
{
    uint16_t linearVal;
    float    dbVal, linVal;

    /* dbVal is a float value from 0-100dB. It needs to
    be converted to linear scale..
    First, recover float dbVal that was encoded in CLI. */
    dbVal = (float)(codedCfarVal / MMWDEMO_CFAR_THRESHOLD_ENCODING_FACTOR);

    /* Now convert it to linear value */
    linVal = (uint32_t)(dbVal/20.0 * CONST_LOG2_10 * 2048.0);

    linearVal = (uint16_t) linVal;
    return (linearVal);
}


/**
 *  @b Description
 *  @n
 *      The function is used to configure the data path based on the chirp profile.
 *      After this function is executed, the data path processing will ready to go
 *      when the ADC buffer starts receiving samples corresponding to the chirps.
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
static int32_t MmwDemo_dataPathConfig (void)
{
    int32_t                         errCode;
    MMWave_CtrlCfg                  *ptrCtrlCfg;
    MmwDemo_DPC_ObjDet_CommonCfg *objDetCommonCfg;
    MmwDemo_SubFrameCfg             *subFrameCfg;
    int8_t                          subFrameIndx;
    MmwDemo_RFParserOutParams       RFparserOutParams;
    DPC_ObjectDetection_PreStartCfg  objDetPreStartCfg;
    DPC_ObjectDetection_StaticCfg   *staticCfg;
    bool procChain = 1;
    /* Get data path object and control configuration */
    ptrCtrlCfg = &gMmwMssMCB.cfg.ctrlCfg;

    objDetCommonCfg = &gMmwMssMCB.objDetCommonCfg;
    staticCfg = &objDetPreStartCfg.staticCfg;

    gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames =
        MmwDemo_RFParser_getNumSubFrames(ptrCtrlCfg);

    {
        /* Calculating the maximum ADC Samples across all frames, useful for common array allocations */
        gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.maxAdcSamples = 0U;
        for(subFrameIndx = 0; subFrameIndx < gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames;
            subFrameIndx++)
        {
            subFrameCfg  = &gMmwMssMCB.subFrameCfg[subFrameIndx];
            if(subFrameCfg->numAdcSamples > gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.maxAdcSamples)
            {
                gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.maxAdcSamples = subFrameCfg->numAdcSamples;
            }
        }
    }

    DebugP_logInfo("App: Issuing Pre-start Common Config IOCTL\n");

    /* Get RF frequency scale factor */
    gMmwMssMCB.rfFreqScaleFactor = MMWDEMO_RF_FREQ_SCALE_FACTOR;

    /* DPC pre-start common config */
    errCode = MmwDemo_DPM_ioctl_blocking (gMmwMssMCB.objDetDpmHandle,
                         DPC_OBJDET_IOCTL__STATIC_PRE_START_COMMON_CFG,
                         &objDetCommonCfg->preStartCommonCfg,
                         sizeof (DPC_ObjectDetection_PreStartCommonCfg));

    if (errCode < 0)
    {
        DebugP_log ("Error: Unable to send DPC_OBJDET_IOCTL__STATIC_PRE_START_COMMON_CFG [Error:%d]\n", errCode);
        goto exit;
    }

    MmwDemo_resetDynObjDetCommonCfgPendingState(&gMmwMssMCB.objDetCommonCfg);

    /* Reason for reverse loop is that when sensor is started, the first sub-frame
     * will be active and the ADC configuration needs to be done for that sub-frame
     * before starting (ADC buf hardware does not have notion of sub-frame, it will
     * be reconfigured every sub-frame). This cannot be alternatively done by calling
     * the MmwDemo_ADCBufConfig function only for the first sub-frame because this is
     * a utility API that computes the rxChanOffset that is part of ADC dataProperty
     * which will be used by range DPU and therefore this computation is required for
     * all sub-frames.
     */
    for(subFrameIndx = gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames -1; subFrameIndx >= 0;
        subFrameIndx--)
    {
        subFrameCfg  = &gMmwMssMCB.subFrameCfg[subFrameIndx];

        /*****************************************************************************
         * Data path :: Algorithm Configuration
         *****************************************************************************/

        /* Parse the profile and chirp configs and get the valid number of TX Antennas */
        errCode = MmwDemo_RFParser_parseConfig(&RFparserOutParams, subFrameIndx,
                                         &gMmwMssMCB.cfg.openCfg, ptrCtrlCfg,
                                         &subFrameCfg->adcBufCfg,
                                         gMmwMssMCB.rfFreqScaleFactor,
                                         false, procChain);

        /* if number of doppler chirps is too low, interpolate to be able to detect
         * better with CFAR tuning. E.g. a 2-pt FFT will be problematic in terms
         * of distinguishing direction of motion */
        if (RFparserOutParams.numDopplerChirps <= 4)
        {
            RFparserOutParams.dopplerStep = RFparserOutParams.dopplerStep / (8 / RFparserOutParams.numDopplerBins);
            RFparserOutParams.numDopplerBins = 8;
        }

        if (errCode != 0)
        {
            DebugP_log ("Error: MmwDemo_RFParser_parseConfig [Error:%d]\n", errCode);
            goto exit;
        }

        subFrameCfg->numRangeBins = RFparserOutParams.numRangeBins;
        /* Workaround for range DPU limitation for FFT size 1024 and 12 virtual antennas case*/
        if ((RFparserOutParams.numVirtualAntennas == 12) && (RFparserOutParams.numRangeBins == 1024))
        {
            subFrameCfg->numRangeBins = 1022;
            RFparserOutParams.numRangeBins = 1022;
        }

        subFrameCfg->datapathStaticCfg.compressionCfg.numRxAntennaPerBlock = RFparserOutParams.numRxAntennas;
        if(subFrameCfg->datapathStaticCfg.compressionCfg.compressionMethod == 1)
        {   /* BFP Compression */
            subFrameCfg->datapathStaticCfg.compressionCfg.bfpCompExtraParamSets = 2*(RFparserOutParams.numRxAntennas -1);
        }
        else
        {
            subFrameCfg->datapathStaticCfg.compressionCfg.bfpCompExtraParamSets = 0;
        }

        subFrameCfg->numDopplerBins = RFparserOutParams.numDopplerBins;
        subFrameCfg->numChirpsPerChirpEvent = RFparserOutParams.numChirpsPerChirpEvent;
        subFrameCfg->adcBufChanDataSize = RFparserOutParams.adcBufChanDataSize;

        subFrameCfg->numAdcSamples = RFparserOutParams.numAdcSamples;
        subFrameCfg->numChirpsPerSubFrame = RFparserOutParams.numChirpsPerFrame;
        subFrameCfg->numVirtualAntennas = RFparserOutParams.numVirtualAntennas;

        /* Power Opt Config */
        staticCfg->powerOptCfg.isHwaDynamicClkGate    = gMmwMssMCB.powerOptCfg.isHwaDynamicClkGate;
        staticCfg->powerOptCfg.hwaStateAfterFrameProc = gMmwMssMCB.powerOptCfg.hwaStateAfterFrameProc;
        staticCfg->powerOptCfg.dspStateAfterFrameProc = gMmwMssMCB.powerOptCfg.dspStateAfterFrameProc;

        errCode = MmwDemo_ADCBufConfig(gMmwMssMCB.adcBufHandle,
                                 gMmwMssMCB.cfg.openCfg.chCfg.rxChannelEn,
                                 subFrameCfg->numChirpsPerChirpEvent,
                                 subFrameCfg->adcBufChanDataSize,
                                 &subFrameCfg->adcBufCfg,
                                 &staticCfg->ADCBufData.dataProperty.rxChanOffset[0]);
        if (errCode < 0)
        {
            DebugP_log("Error: ADCBuf config failed with error[%d]\n", errCode);
            MmwDemo_debugAssert (0);
        }

        errCode = MmwDemo_configCQ(subFrameCfg, subFrameCfg->numChirpsPerChirpEvent,
                                   RFparserOutParams.validProfileIdx);
        if (errCode < 0)
        {
            DebugP_log("Error: CQ config failed with error[%d]\n", errCode);
            MmwDemo_debugAssert(0);
        }

        /* DPC pre-start config */
        {
            int32_t idx;

            objDetPreStartCfg.subFrameNum = subFrameIndx;

            /* Fill static configuration */
            staticCfg->ADCBufData.data = (void *)CSL_RSS_ADCBUF_READ_U_BASE;
            staticCfg->ADCBufData.dataProperty.adcBits = ADCBUF_DATA_PROPERTY_ADCBITS_16BIT; /* 16-bit */

            /* only real format supported */
            MmwDemo_debugAssert(subFrameCfg->adcBufCfg.adcFmt == 1);

            staticCfg->ADCBufData.dataProperty.dataFmt = DPIF_DATAFORMAT_REAL16;

            if (subFrameCfg->adcBufCfg.chInterleave == 0)
            {
                staticCfg->ADCBufData.dataProperty.interleave = DPIF_RXCHAN_INTERLEAVE_MODE;
            }
            else
            {
                staticCfg->ADCBufData.dataProperty.interleave = DPIF_RXCHAN_NON_INTERLEAVE_MODE;
            }
            staticCfg->ADCBufData.dataProperty.numAdcSamples = RFparserOutParams.numAdcSamples;
            staticCfg->ADCBufData.dataProperty.numChirpsPerChirpEvent = RFparserOutParams.numChirpsPerChirpEvent;
            staticCfg->ADCBufData.dataProperty.numRxAntennas = RFparserOutParams.numRxAntennas;
            staticCfg->ADCBufData.dataSize = RFparserOutParams.numRxAntennas * RFparserOutParams.numAdcSamples * sizeof(cmplx16ImRe_t);
            staticCfg->dopplerStep = RFparserOutParams.dopplerStep;
            staticCfg->isValidProfileHasOneTxPerChirp = RFparserOutParams.validProfileHasOneTxPerChirp;
            staticCfg->numChirpsPerFrame = RFparserOutParams.numChirpsPerFrame;
            staticCfg->numDopplerBins = RFparserOutParams.numDopplerBins;

            staticCfg->ADCBufConfig.rxChannelEn = gMmwMssMCB.cfg.openCfg.chCfg.rxChannelEn;
            staticCfg->ADCBufConfig.adcBufChanDataSize = RFparserOutParams.adcBufChanDataSize;

            staticCfg->numChirps = RFparserOutParams.numDopplerChirps;
            /* Sum Tx must be enabled if range profile is to be sent out or
                 range CFAR is enabled */

            staticCfg->isSumTxEnabled = (subFrameCfg->datapathStaticCfg.rangeCfarCfg.cfg.isEnabled) ||
                                        (subFrameCfg->guiMonSel.logMagRange) ||
                                        (subFrameCfg->guiMonSel.noiseProfile) ||
                                        (subFrameCfg->guiMonSel.rangeDopplerHeatMap)||
                                        (gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.measureRxChannelBiasCfg.enabled);
            staticCfg->numRangeBins = RFparserOutParams.numRangeBins;
            /* Number of range bins are half the number of FFT Bins in case of real only chirp data */
            if(RFparserOutParams.adcDataFmtIsReal){
                staticCfg->numRangeFFTBins = (RFparserOutParams.numRangeBins)*(2);
            }
            else{
                staticCfg->numRangeFFTBins = (RFparserOutParams.numRangeBins);
            }
            staticCfg->numTxAntennas = RFparserOutParams.numTxAntennas;
            staticCfg->numVirtualAntAzim = RFparserOutParams.numVirtualAntAzim;
            staticCfg->numVirtualAntElev = RFparserOutParams.numVirtualAntElev;
            staticCfg->numVirtualAntennas = RFparserOutParams.numVirtualAntennas;
            staticCfg->rangeStep = RFparserOutParams.rangeStep;
            staticCfg->numBandsTotal = staticCfg->numTxAntennas + MmwDemo_getNumEmptySubBands(staticCfg->numTxAntennas);

            if(staticCfg->numRangeFFTBins  > 1024 ){
                staticCfg->rangeFFTtuning.fftOutputDivShift = 0;
                staticCfg->rangeFFTtuning.numLastButterflyStagesToScale = 3; /* scale only 3 stages */
            }
            else if (staticCfg->numRangeFFTBins  >= 1022)
            {
                staticCfg->rangeFFTtuning.fftOutputDivShift = 0;
                staticCfg->rangeFFTtuning.numLastButterflyStagesToScale = 2; /* scale only 2 stages */
            } else if (staticCfg->numRangeFFTBins  == 512)
            {
                staticCfg->rangeFFTtuning.fftOutputDivShift = 1;
                staticCfg->rangeFFTtuning.numLastButterflyStagesToScale = 1; /* scale last stage */
            } else
            {
                staticCfg->rangeFFTtuning.fftOutputDivShift = 2;
                staticCfg->rangeFFTtuning.numLastButterflyStagesToScale = 0; /* no scaling needed as ADC data is 16-bit and we have 8 bits to grow */
            }

            for (idx = 0; idx < RFparserOutParams.numRxAntennas; idx++)
            {
                staticCfg->rxAntOrder[idx] = RFparserOutParams.rxAntOrder[idx];
            }
            for (idx = 0; idx < RFparserOutParams.numTxAntennas; idx++)
            {
                staticCfg->txAntOrder[idx] = RFparserOutParams.txAntOrder[idx];
            }

            DebugP_logInfo("App: Issuing Pre-start Config IOCTL (subFrameIndx = %d)\n", subFrameIndx);

            /* Copy out the DPC Static cfg params */

            staticCfg->cfarCfg.subFrameNum = subFrameIndx;

            subFrameCfg->datapathStaticCfg.cfarCfg.cfg.thresholdScale =
                MmwDemo_convertDopplerCfarToThresh(subFrameCfg->datapathStaticCfg.cfarCfg.cfg.thresholdScale); //staticCfg->numVirtualAntennas);
            memcpy(&staticCfg->cfarCfg.cfg, &subFrameCfg->datapathStaticCfg.cfarCfg.cfg, sizeof(DPU_DopplerProc_CfarCfg));
            memcpy(&staticCfg->compressionCfg, &subFrameCfg->datapathStaticCfg.compressionCfg, sizeof(DPU_RangeProcHWA_CompressionCfg));
            memcpy(&staticCfg->localMaxCfg, &subFrameCfg->datapathStaticCfg.localMaxCfg, sizeof(DPU_DopplerProc_LocalMaxCfg));
            memcpy(&staticCfg->intfStatsdBCfg, &subFrameCfg->datapathStaticCfg.intfStatsdBCfg, sizeof(DPU_RangeProcHWADDMA_intfStatsdBCfg));
            memcpy(&staticCfg->aoaFovCfg, &subFrameCfg->datapathStaticCfg.aoaFovCfg, sizeof(DPC_ObjectDetection_FovAoaCfg));

            staticCfg->rangeCfarCfg.subFrameNum = subFrameIndx;
            subFrameCfg->datapathStaticCfg.rangeCfarCfg.cfg.thresholdScale =
                MmwDemo_convertRangeCfarToThresh(subFrameCfg->datapathStaticCfg.rangeCfarCfg.cfg.thresholdScale, staticCfg->numBandsTotal);
            memcpy(&staticCfg->rangeCfarCfg.cfg, &subFrameCfg->datapathStaticCfg.rangeCfarCfg.cfg, sizeof(DPU_CFARProc_CfarCfg));

            /* send pre-start config */
            errCode = MmwDemo_DPM_ioctl_blocking (gMmwMssMCB.objDetDpmHandle,
                                 DPC_OBJDET_IOCTL__STATIC_PRE_START_CFG,
                                 &objDetPreStartCfg,
                                 sizeof (DPC_ObjectDetection_PreStartCfg));
            if (errCode < 0)
            {
                DebugP_log ("Error: Unable to send DPC_OBJDET_IOCTL__STATIC_PRE_START_CFG [Error:%d]\n", errCode);
                goto exit;
            }
        }
    }
#if !MSS_AOA_ENABLED
    /* send elevation estimation config to DSS if AoA proc is run on DSS */
    errCode = MmwDemo_DPM_ioctl_blocking (gMmwMssMCB.objDetDpmHandle,
                        DPC_OBJDET_IOCTL__STATIC_PRE_START_ELEVEST_CFG,
                        &gElevEstCfg, sizeof (DPC_ObjectDetection_ElevEstCfg));

    if (errCode < 0)
    {
        DebugP_log ("Error: Unable to send DPC_OBJDET_IOCTL__STATIC_PRE_START_ELEVEST_CFG [Error:%d]\n", errCode);
        goto exit;
    }
#endif
exit:
    return errCode;
}

/**
 *  @b Description
 *  @n
 *      The function is used to Start data path to handle chirps from front end.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dataPathStart (void)
{
    DebugP_logInfo("App: Issuing DPM_start\n");
#ifdef LVDS_STREAM
    /* Configure HW LVDS stream for the first sub-frame that will start upon
     * start of frame */
    if (gMmwMssMCB.subFrameCfg[0].lvdsStreamCfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED)
    {
        MmwDemo_configLVDSHwData(0);
    }
#endif
}

/**
 *  @b Description
 *  @n
 *      The function is used to stop data path.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_dataPathStop (void)
{
    DebugP_logInfo("App: Issuing DPM_stop\n");

    DPC_ObjectDetection_ExecuteResult* result;

    result = (DPC_ObjectDetection_ExecuteResult *)gElevEstCfg.commonCfg.result;

    /* Write the FFT clip status on CLI. */
    if(result->FFTClipCount[0]>0U)
    {
        CLI_write("Warning! FFT clipping happened for %d times in Range FFT Stage. \n", result->FFTClipCount[0]);
    }
    if(result->FFTClipCount[1]>0U)
    {
        CLI_write("Warning! FFT clipping happened for %d times in Doppler or Azimuth FFT Stage. \n", result->FFTClipCount[1]);
    }

    result->stats->frameStartIntCounter = 0;
    result->stats->subframeStartIntCounter = 0;
    memset((void*)&result->FFTClipCount[0], 0, sizeof(result->FFTClipCount));

    SemaphoreP_post(&gMmwMssMCB.DPMstopSemHandle);
}

/**
 *  @b Description
 *  @n
 *      Registered event function to mmwave which is invoked when an event from the
 *      BSS is received.
 *
 *  @param[in]  devIndex
 *      Device Index
 *  @param[in]  msgId
 *      Message Identifier
 *  @param[in]  sbId
 *      Subblock identifier
 *  @param[in]  sbLen
 *      Length of the subblock
 *  @param[in]  payload
 *      Pointer to the payload buffer
 *
 *  @retval
 *      Always return 0
 */
static int32_t MmwDemo_eventCallbackFxn(uint8_t devIndex, uint16_t msgId, uint16_t sbId, uint16_t sbLen, uint8_t *payload)
{
    uint16_t asyncSB = RL_GET_SBID_FROM_UNIQ_SBID(sbId);

    /* SEEKER PATCH: log every BSS async event to CLI UART so host can
     * see exactly which event fires before LVDS halt / sensorStart -1.
     * Goal: narrow root cause of multi-minute LVDS streaming halt. */
    CLI_write("BSSEV t=%u dev=%u msgId=0x%04x sb=0x%04x len=%u\r\n",
              (uint32_t)ClockP_getTimeUsec(), devIndex, msgId, asyncSB, sbLen);

    /* Process the received message: */
    switch (msgId)
    {
        case RL_RF_ASYNC_EVENT_MSG:
        {
            /* Received Asychronous Message: */
            switch (asyncSB)
            {
                case RL_RF_AE_CPUFAULT_SB:
                {
                    CLI_write("BSSEV !! RF_CPUFAULT (BSS firmware crash)\r\n");
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_ESMFAULT_SB:
                {
                    uint32_t esm0 = ((uint32_t*)payload)[0];
                    uint32_t esm1 = ((uint32_t*)payload)[1];
                    CLI_write("BSSEV !! RF_ESMFAULT esm0=0x%08x esm1=0x%08x\r\n", esm0, esm1);
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_ANALOG_FAULT_SB:
                {
                    CLI_write("BSSEV !! RF_ANALOG_FAULT\r\n");
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_RF_AE_INITCALIBSTATUS_SB:
                {
                    rlRfInitComplete_t*  ptrRFInitCompleteMessage;
                    uint32_t            calibrationStatus;

                    /* Get the RF-Init completion message: */
                    ptrRFInitCompleteMessage = (rlRfInitComplete_t*)payload;
                    calibrationStatus = ptrRFInitCompleteMessage->calibStatus & 0x1FFFU;

                    /* Display the calibration status: */
                    CLI_write ("Debug: Init Calibration Status = 0x%x\r\n", calibrationStatus);
                    break;
                }
                case RL_RF_AE_FRAME_TRIGGER_RDY_SB:
                {
                    gMmwMssMCB.stats.frameTriggerReady++;
                    break;
                }
                case RL_RF_AE_MON_TIMING_FAIL_REPORT_SB:
                {
                    gMmwMssMCB.stats.failedTimingReports++;
                    CLI_write("BSSEV MON_TIMING_FAIL n=%u\r\n",
                              gMmwMssMCB.stats.failedTimingReports);
                    break;
                }
                case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
                {
                    gMmwMssMCB.stats.calibrationReports++;
                    {
                        rlRfRunTimeCalibReport_t *r = (rlRfRunTimeCalibReport_t*)payload;
                        CLI_write("BSSEV RUN_CALIB done=0x%x err=0x%x temp=%d\r\n",
                                  r->calibUpdateStatus, r->calibErrorFlag, r->temperature);
                    }
                    break;
                }
                case RL_RF_AE_FRAME_END_SB:
                {
                    gMmwMssMCB.stats.sensorStopped++;
                    /* SEEKER PATCH 2026-05-05: do NOT call MmwDemo_dataPathStop()
                     * here. The companion openCfg.disableFrameStopAsyncEvent=true
                     * setting normally suppresses delivery of this event entirely;
                     * if it slips through anyway (different SDK build, runtime
                     * race), tearing down the data path is what causes the LVDS
                     * stream to halt indefinitely without recovery. Just log and
                     * continue -- if BSS truly stopped, the host-side LVDS UDP
                     * watchdog will catch the gap and surface it. */
                    CLI_write("BSSEV !! FRAME_END (n=%u) -- suppressing dataPathStop, streaming continues\r\n",
                              gMmwMssMCB.stats.sensorStopped);
                    DebugP_logInfo("App: BSS FRAME_END received (suppressed)\n");
                    break;
                }
                default:
                {
                    CLI_write("BSSEV RF unhandled sb=0x%04x\r\n", asyncSB);
                    DebugP_log ("Error: Asynchronous Event SB Id %d not handled\n", asyncSB);
                    break;
                }
            }
            break;
        }
        /* Async Event from MMWL */
        case RL_MMWL_ASYNC_EVENT_MSG:
        {
            switch (asyncSB)
            {
                case RL_MMWL_AE_MISMATCH_REPORT:
                {
                    /* link reports protocol error in the async report from BSS */
                    MmwDemo_debugAssert(0);
                    break;
                }
                case RL_MMWL_AE_INTERNALERR_REPORT:
                {
                    /* link reports internal error during BSS communication */
                    MmwDemo_debugAssert(0);
                    break;
                }
            }
            break;
        }
        /* Async Event from MSS */
        case RL_DEV_ASYNC_EVENT_MSG:
        {
            switch (asyncSB)
            {
                case RL_DEV_AE_MSSPOWERUPDONE_SB:
                {
                    DebugP_log("Received RL_DEV_AE_MSSPOWERUPDONE_SB\n");

                }
                break;
                default:
                {
                    DebugP_log("Unhandled Async Event msgId: 0x%x, asyncSB:0x%x  \n\n", msgId, asyncSB);
                    break;
                }
            }
            break;

        }
        default:
        {
            DebugP_log ("Error: Asynchronous message %d is NOT handled\n", msgId);
            break;
        }
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      DPM Registered Report Handler. The DPM Module uses this registered function to notify
 *      the application about DPM reports.
 *
 *  @param[in]  data
 *      Pointer to data
 *  @param[in]  dataLen
 *      Length of the data
 *
 *  @retval
 *      Not Applicable.
 */

void MmwDemo_DPC_ObjectDetection_reportFxn(void *data, uint16_t dataLen)
{
    static uint32_t subFrameCnt = RL_MAX_SUBFRAMES;

    if(dataLen == sizeof(DPC_ObjectDetection_ElevEstCommonCfg))
    {
        gElevEstCfg.commonCfg = *(DPC_ObjectDetection_ElevEstCommonCfg*)data;
        gElevEstCfg.commonCfg.dspStateAfterFrameProc = gMmwMssMCB.powerOptCfg.dspStateAfterFrameProc;
        subFrameCnt = gElevEstCfg.commonCfg.numSubFrames;
        SemaphoreP_post(&gMmwMssMCB.DPMioctlSemHandle);
    }
    else if (dataLen == sizeof(DPC_ObjectDetection_ElevEstSubframeCfg))
    {
        gElevEstCfg.subframeCfg[--subFrameCnt] = *(DPC_ObjectDetection_ElevEstSubframeCfg *)(data);
        SemaphoreP_post(&gMmwMssMCB.DPMioctlSemHandle);
    }
    else
    {
        uint32_t msg = *((uint32_t*)data);
        switch (msg)
        {
            case MMWDEMO_DPC_COMMONCFG:
            case MMWDEMO_DPC_ELEVEST_CFG:
                SemaphoreP_post(&gMmwMssMCB.DPMioctlSemHandle);
                break;
            case MMWDEMO_DPC_RESULT:
            {
                /*****************************************************************
                 * datapath has finished frame processing, results are reported
                 *****************************************************************/
                /* SEEKER PATCH 2026-05-05: removed the periodic-CBUFF-cycle
                 * call. Empirical sweep at v3 (5s)/v4 (2.5s)/v5 (10s)/v6
                 * (off) showed faster cycling REGRESSED the burst length
                 * (v4 dropped from ~85s to ~30s), proving the chip-side
                 * CBUFF wraparound is NOT the failure mode. The halt is
                 * monitor-timing related (see openCfg patch above) so the
                 * cycle hook is removed entirely. The function definition
                 * stays in mmw_lvds_stream.c for ABI compat. */

                if(gMmwMssMCB.stats.isLastFrameDataProcessed)
                {
                    /* reset Frame data processed flag, set after full obj data is actually streamed out */
                    gMmwMssMCB.stats.isLastFrameDataProcessed = false;
                    /* signal the UART task to transmit the data */
                    SemaphoreP_post(&gMmwMssMCB.UartExportSemHandle);
                }
                else{
                    volatile uint32_t transmitTime = (CycleCounterP_getCount32() - transmitStartTime)/(SOC_getSelfCpuClk()/1000000U);
                    DebugP_log ("UART processing not completed: numObjOut %d Time %d\n", ((DPC_ObjectDetection_ExecuteResult *)gElevEstCfg.commonCfg.result)->numObjOut, transmitTime);
                    MmwDemo_debugAssert(0);
                }
                break;
            }
        }
    }
}

/**
 *  @b Description
 *  @n
 *      Utility function to get next sub-frame index
 *
 *  @param[in] currentIndx Current sub-frame index
 *  @param[in] numSubFrames Number of sub-frames
 *
 *  @retval
 *      Index of next sub-frame.
 */
static uint8_t MmwDemo_getNextSubFrameIndx(uint8_t currentIndx, uint8_t numSubFrames)
{
    uint8_t nextIndx;

    if (currentIndx == (numSubFrames - 1))
    {
        nextIndx = 0;
    }
    else
    {
        nextIndx = currentIndx + 1;
    }
    return(nextIndx);
}

/**
 *  @b Description
 *  @n
 *      Utility function to get previous sub-frame index
 *
 *  @param[in] currentIndx Current sub-frame index
 *  @param[in] numSubFrames Number of sub-frames
 *
 *  @retval
 *      Index of previous sub-frame
 */
static uint8_t MmwDemo_getPrevSubFrameIndx(uint8_t currentIndx, uint8_t numSubFrames)
{
    uint8_t prevIndx;

    if (currentIndx == 0)
    {
        prevIndx = numSubFrames - 1;
    }
    else
    {
        prevIndx = currentIndx - 1;
    }
    return(prevIndx);
}

/* Workaround for Errata ANA#46: Spurs caused due to data transfer activity */
static void MmwDPC_chirpAvailISR(void* arg)
{
    uint32_t delay = 0;
    CSL_rss_ctrlRegs *ptr_rss_ctrl_regs = (CSL_rss_ctrlRegs*)CSL_RSS_CTRL_U_BASE;

    if(gMmwMssMCB.adcDataDithDelayCfg.isDitherEn == 1U)
    {
        /* configure the variable amount of delay */
        delay = rand() % gMmwMssMCB.adcDataDithDelayCfg.ditherVal;

        ptr_rss_ctrl_regs->ADCBUFCFG1_EXTD =  delay + MMWDEMO_DITHERING_MINDELAY;
    }
}

/* Workaround for Errata ANA#46: Spurs caused due to data transfer activity */
static int32_t MmwDemo_registerChirpStartInterrupt(void)
{
    int32_t retVal = 0;
    int32_t status = SystemP_SUCCESS;
    HwiP_Params hwiPrms;

    /* Register interrupt */
    HwiP_Params_init(&hwiPrms);
    hwiPrms.intNum = CSL_MSS_INTR_RSS_ADC_CAPTURE_COMPLETE_DITH;
    hwiPrms.callback = &MmwDPC_chirpAvailISR;
    hwiPrms.priority = 2;
    status = HwiP_construct(&gMmwMssMCB.adcDataDithDelayCfg.chirpAvailHwiObject, &hwiPrms);
    if (SystemP_SUCCESS != status)
    {
        retVal = SystemP_FAILURE;
    }
    else
    {
        /* Keep the interrupt disable: Interrupt should be enabled once dithering enable request is received from CLI */
        HwiP_disableInt((uint32_t)CSL_MSS_INTR_RSS_ADC_CAPTURE_COMPLETE_DITH);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      Function to handle frame processing results from DPC
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_handleObjectDetResult(void)
{
    DPC_ObjectDetection_ExecuteResult        *dpcResults;
    volatile uint32_t                        startTime;
    uint8_t                                  nextSubFrameIdx;
    uint8_t                                  numSubFrames;
    uint8_t                                  currSubFrameIdx;
    MmwDemo_SubFrameStats                    currSubFrameStats;

    static uint32_t prevInterFrameEndTimeStamp = 0U;

    /*****************************************************************
     * datapath has finished frame processing, results are reported
     *****************************************************************/
    dpcResults = gElevEstCfg.commonCfg.result;

    numSubFrames = gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames;
    currSubFrameIdx = dpcResults->subFrameIdx;

    /*****************************************************************
     * Transmit results
     *****************************************************************/
    startTime = CycleCounterP_getCount32();


    /* Send out of CLI the range bias and phase config measurement if it was enabled. */
    if (gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.measureRxChannelBiasCfg.enabled == 1)
    {
        if(dpcResults->compRxChanBiasMeasurement != NULL)
        {
            dpcResults->compRxChanBiasMeasurement = (Measure_compRxChannelBiasCfg *) AddrTranslateP_getLocalAddr((uint32_t)dpcResults->compRxChanBiasMeasurement);
            MmwDemo_measurementResultOutput((void*)dpcResults->compRxChanBiasMeasurement);
        }
        else
        {
            /* DPC is not ready to ship the measurement results */
        }
    }

    /* DSS_CM4 CPU load is not computed as it is a nortos application. */
    currSubFrameStats.outputStats.interFrameCPULoad = 0;
    currSubFrameStats.outputStats.activeFrameCPULoad= 0;
    /* Update current frame stats */
    currSubFrameStats.outputStats.interChirpProcessingMargin = dpcResults->stats->interChirpProcessingMargin;
    currSubFrameStats.outputStats.interFrameProcessingTime = (dpcResults->stats->interFrameEndTimeStamp - dpcResults->stats->interFrameStartTimeStamp)/CM4_CLOCK_MHZ;
    currSubFrameStats.outputStats.interFrameProcessingMargin = (dpcResults->stats->frameStartTimeStamp - prevInterFrameEndTimeStamp)/CM4_CLOCK_MHZ;
    prevInterFrameEndTimeStamp = dpcResults->stats->interFrameEndTimeStamp;

#ifdef LVDS_STREAM
    if (gMmwMssMCB.subFrameCfg[currSubFrameIdx].lvdsStreamCfg.dataFmt !=
             MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED)
    {
        /* check Edma errors (which are considered fatal) for the current sub-frame's
         * h/w session that is expected to be completed by now */
        MmwDemo_checkEdmaErrors();

        /* Pend for completion of h/w session, generally this will not wait
         * because of time spent doing inter-frame processing is expected to
         * be bigger than the transmission of the h/w session */
        SemaphoreP_pend(&gMmwMssMCB.lvdsStream.hwFrameDoneSemHandle, SystemP_WAIT_FOREVER);
    }
#endif

    /* Transmit processing results for the frame */
    transmitStartTime = CycleCounterP_getCount32();
    MmwDemo_transmitProcessedOutput(gMmwMssMCB.loggingUartHandle,
                                    dpcResults,
                                    &currSubFrameStats.outputStats);

    /* Update current frame transmit time */
    currSubFrameStats.outputStats.transmitOutputTime = (CycleCounterP_getCount32() - startTime)/(SOC_getSelfCpuClk()/1000000U); /* In micro seconds */

    nextSubFrameIdx = MmwDemo_getNextSubFrameIndx(currSubFrameIdx,   numSubFrames);

    /*****************************************************************
     * Prepare for subFrame switch
     *****************************************************************/
    if(numSubFrames > 1)
    {
        MmwDemo_SubFrameCfg  *nextSubFrameCfg;

        nextSubFrameCfg = &gMmwMssMCB.subFrameCfg[nextSubFrameIdx];

#ifdef LVDS_STREAM
        /* Configure HW LVDS stream for this subframe? */
        if(nextSubFrameCfg->lvdsStreamCfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED)
        {
            /* check Edma errors (which are considered fatal) for any previous session, even
             * though we have checked for a previous s/w session, if s/w session weren't
             * enabled, then this will check for previous h/w session related Edma errors */
            MmwDemo_checkEdmaErrors();

            MmwDemo_configLVDSHwData(nextSubFrameIdx);
        }
        currSubFrameStats.subFramePreparationTime = (CycleCounterP_getCount32() - startTime)/(SOC_getSelfCpuClk()/1000000U);
#endif
    }
    else
    {
        currSubFrameStats.subFramePreparationTime = 0;
    }

    /* set the Frame data processed flag to indicate that obj data is out successfully */
    gMmwMssMCB.stats.isLastFrameDataProcessed = true;
}

/**
 *  @b Description
 *  @n
 *      UART Data Export Task which exports the detected objects and other processing chain outputs on UART.
 *
 *  @retval
 *      Not Applicable.
 */
void mmwDemo_mssUartDataExportTask(void* args)
{
    int32_t retVal;
    uint32_t msg = MMWDEMO_DPC_EXECUTE;

    /* Enable save/restore of Floating Point Registers */
    portTASK_USES_FLOATING_POINT();

    while (1)
    {
        /* Export the Data: */
        SemaphoreP_pend(&gMmwMssMCB.UartExportSemHandle, SystemP_WAIT_FOREVER);

#if MSS_AOA_ENABLED
        /* Perform the elevation estimation */
        DPC_ObjectDetection_ExecuteResult *result = gElevEstCfg.commonCfg.result;
        DetObjParams *detObjList = (DetObjParams*)SOC_phyToVirt((uint32_t)result->detObjList);
        DPC_ObjDet_estimateXYZ(&gElevEstCfg, detObjList, result->objOut,
                                    result->dopNumObjOut,
                                    &result->numObjOut);

        /* This is required to flush out L2. */
        CacheP_wbInvAll(CacheP_TYPE_ALLD);
#endif

        /* This is to ensure that AoA happens only in parallel with Range Proc and do not exceed after that. */
        retVal = DPM_send(gMmwMssMCB.objDetDpmHandle, (void*)(&msg), sizeof(uint32_t), CSL_CORE_ID_M4SS0_1);
        DebugP_assert(0U == retVal);

#ifdef ENET_STREAM
        if(gMmwMssMCB.enetCfg.streamEnable) {

            gEnetStreamObjData.numObj = gElevEstCfg.commonCfg.result->numObjOut;
            gEnetStreamObjData.dummy  = 0x0U;

            gEnetStreamObjData.objData = (uint8_t*)(gElevEstCfg.commonCfg.result->objOut);
            SemaphoreP_post(&objDataSemaphoreHandle);

            gMmwMssMCB.stats.isLastFrameDataProcessed = true;
        }
#else
        MmwDemo_handleObjectDetResult();
#endif
    }
}

#ifdef ENET_STREAM
/**
 *  @b Description
 *  @n
 *      Function to declare when Ethernet Configuration is complete
 *
 *  @retval
 *      0  - Success.
 *      <0 - Failed with errors
 */
int32_t MmwDemo_mssEnetCfgDone(void)
{

    /* Post EnetCfgDone Semaphore to signal that the IP
    has been configured and connection can now be made */
    SemaphoreP_post(&gMmwMssMCB.enetCfg.EnetCfgDoneSemHandle);

    return 0;
}
#endif

/**************************************************************************
 ******************** Millimeter Wave Demo sensor management Functions **********
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      mmw demo helper Function to do one time sensor initialization.
 *      User need to fill gMmwMssMCB.cfg.openCfg before calling this function
 *
 *  @param[in]  isFirstTimeOpen     If true then issues MMwave_open
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_openSensor(bool isFirstTimeOpen)
{
    int32_t             errCode;
    MMWave_ErrorLevel   errorLevel;
    int16_t             mmWaveErrorCode;
    int16_t             subsysErrorCode;
    int32_t             retVal;
    MMWave_CalibrationData     calibrationDataCfg;
    MMWave_CalibrationData     *ptrCalibrationDataCfg;

    /*  Open mmWave module, this is only done once */
    if (isFirstTimeOpen == true)
    {

        /**********************************************************
         **********************************************************/

        /* Open mmWave module, this is only done once */
        /* Setup the calibration frequency:*/
        gMmwMssMCB.cfg.openCfg.freqLimitLow  = 760U;
        gMmwMssMCB.cfg.openCfg.freqLimitHigh = 810U;

        /* start/stop async events */
        gMmwMssMCB.cfg.openCfg.disableFrameStartAsyncEvent = false;
        /* SEEKER PATCH 2026-05-05: suppress FRAME_END async event so a
         * transient timing-monitor trip in BSS doesn't tear down the data
         * path. With this true, the BSS still emits FRAME_END internally if
         * a real failure occurs, but it no longer causes MMWave layer to
         * call MmwDemo_dataPathStop() (see MmwDemo_eventCallbackFxn
         * @ RL_RF_AE_FRAME_END_SB). Companion to the dataPathStop suppress
         * patch ~30 lines further inside the FRAME_END handler. */
        gMmwMssMCB.cfg.openCfg.disableFrameStopAsyncEvent  = true;

        /* No custom calibration: */
        gMmwMssMCB.cfg.openCfg.useCustomCalibration        = false;
        gMmwMssMCB.cfg.openCfg.customCalibrationEnableMask = 0x0;

        /* calibration monitoring base time unit
         * setting it to one frame duration as the demo doesnt support any
         * monitoring related functionality
         */
        gMmwMssMCB.cfg.openCfg.calibMonTimeUnit            = 1;

        if( (gMmwMssMCB.calibCfg.saveEnable != 0) &&
		(gMmwMssMCB.calibCfg.restoreEnable != 0) )
        {
            /* Error: only one can be enabled at at time */
            DebugP_log ("Error: MmwDemo failed with both save and restore enabled.\n");
            return -1;
        }

        if(gMmwMssMCB.calibCfg.restoreEnable != 0)
        {
            if(MmwDemo_calibRestore(&gCalibDataStorage) < 0)
            {
                DebugP_log ("Error: MmwDemo failed restoring calibration data from flash.\n");
                return -1;
            }

            /*  Boot calibration during restore: Disable calibration for:
                 - Rx gain,
                 - Rx IQMM,
                 - Tx phase shifer,
                 - Tx Power

                 The above calibration data will be restored from flash. Since they are calibrated in a control
                 way to avoid interfaerence and spec violations.
                 In this demo, other bit fields(except the above) are enabled as indicated in customCalibrationEnableMask to perform boot time
                 calibration. The boot time calibration will overwrite the restored calibration data from flash.
                 However other bit fields can be disabled and calibration data can be restored from flash as well.

                 Note: In this demo, calibration masks are enabled for all bit fields when "saving" the data.
            */
            gMmwMssMCB.cfg.openCfg.useCustomCalibration        = true;
            gMmwMssMCB.cfg.openCfg.customCalibrationEnableMask = 0x1F0U;

            calibrationDataCfg.ptrCalibData = &gCalibDataStorage.calibData;
            calibrationDataCfg.ptrPhaseShiftCalibData = &gCalibDataStorage.phaseShiftCalibData;
            ptrCalibrationDataCfg = &calibrationDataCfg;
        }
        else
        {
            ptrCalibrationDataCfg = NULL;
        }


        /* Open the mmWave module: */
        if (MMWave_open (gMmwMssMCB.ctrlHandle, &gMmwMssMCB.cfg.openCfg, ptrCalibrationDataCfg, &errCode) < 0)
        {
            /* Error: decode and Report the error */
            MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
            DebugP_log ("Error: mmWave Open failed [Error code: %d Subsystem: %d]\n",
                            mmWaveErrorCode, subsysErrorCode);
            return -1;
        }

        /* Save calibration data in flash */
        if(gMmwMssMCB.calibCfg.saveEnable != 0)
        {
            retVal = rlRfCalibDataStore(RL_DEVICE_MAP_INTERNAL_BSS, &gCalibDataStorage.calibData);
            if(retVal != RL_RET_CODE_OK)
            {
                /* Error: Calibration data restore failed */
                 DebugP_log("MSS demo failed rlRfCalibDataStore with Error[%d]\n", retVal);
                return -1;
            }

            /* update txIndex in all chunks to get data from all Tx.
            This should be done regardless of num TX channels enabled in MMWave_OpenCfg_t::chCfg or number of Tx
            application is interested in. Data for all existing Tx channels should be retrieved
            from RadarSS and in the order as shown below.
            RadarSS will return non-zero phase shift values for all the channels enabled via
            MMWave_OpenCfg_t::chCfg and zero phase shift values for channels disabled in MMWave_OpenCfg_t::chCfg */
            gCalibDataStorage.phaseShiftCalibData.PhShiftcalibChunk[0].txIndex = 0;
            gCalibDataStorage.phaseShiftCalibData.PhShiftcalibChunk[1].txIndex = 1;
            gCalibDataStorage.phaseShiftCalibData.PhShiftcalibChunk[2].txIndex = 2;

            /* Basic validation passed: Restore the phase shift calibration data */
            retVal = rlRfPhShiftCalibDataStore(RL_DEVICE_MAP_INTERNAL_BSS, &(gCalibDataStorage.phaseShiftCalibData));
            if (retVal != RL_RET_CODE_OK)
            {
                /* Error: Phase shift Calibration data restore failed */
                DebugP_log("MSS demo failed rlRfPhShiftCalibDataStore with Error[%d]\n", retVal);
                return retVal;
            }

            /* Save data in flash */
            retVal = MmwDemo_calibSave(&gMmwMssMCB.calibCfg.calibDataHdr, &gCalibDataStorage);
            if(retVal < 0)
            {
                return retVal;
            }
        }

        /* Open the datapath modules that runs on MSS */
        MmwDemo_dataPathOpen();
    }
    return 0;
}

/**
 *  @b Description
 *  @n
 *      MMW demo helper Function to configure phase shifter chirps for the DDMA processing chain.
 *      For the xth Tx antenna and the kth chirp, the phase shifter value is given by
 *      (k - 1) * (x - 1) / (numTxTotal + 1)
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_configPhaseShifterChirps(void){

    int32_t     errCode = 0;
    uint16_t    chirpIdx, chirpStartIdx, chirpEndIdx, numTxAntAzim, numTxTotalDivisor,
                numTxAntElev, txAntMask, txOrderIdx, activeTxCnt, chirpPhaseMultiplier;
    /* xth value of this array corresponds to the phase shift multiplier for the xth Tx antenna */
    uint16_t       phaseShiftMultiplier[SYS_COMMON_NUM_TX_ANTENNAS];
    rlRfPhaseShiftCfg_t phaseShiftCfg;

    memset ((void *)&phaseShiftCfg, 0, sizeof(phaseShiftCfg));

    txAntMask     = gMmwMssMCB.cfg.openCfg.chCfg.txChannelEn;
    numTxAntAzim = mathUtils_countSetBits(txAntMask & MmwDemo_RFParserHwCfg.azimTxAntMask);
    numTxAntElev = mathUtils_countSetBits(txAntMask & MmwDemo_RFParserHwCfg.elevTxAntMask);

    gMmwMssMCB.numEmptySubBands = MmwDemo_getNumEmptySubBands(numTxAntAzim + numTxAntElev);
    numTxTotalDivisor =  numTxAntAzim + numTxAntElev + gMmwMssMCB.numEmptySubBands;

    activeTxCnt = 0;
    /* Get the phase multiplier factor for each Tx antenna */
    /* Loop over Tx Antenna Phase order index */
    for(txOrderIdx = 0; txOrderIdx < SYS_COMMON_NUM_TX_ANTENNAS; txOrderIdx++){
        /* Check if the Tx antenna corresponding to the xth index in the phase order is enabled */
        if(1 << gMmwMssMCB.ddmPhaseShiftOrder[txOrderIdx] & txAntMask){
            /* Antenna is enabled, hence compute the phase shift value */
            phaseShiftMultiplier[gMmwMssMCB.ddmPhaseShiftOrder[txOrderIdx]] = activeTxCnt;
            activeTxCnt++;
        }
        else{
            /* Antenna is disabled */
            phaseShiftMultiplier[gMmwMssMCB.ddmPhaseShiftOrder[txOrderIdx]] = 0;
        }
    }

    /* Configure Phase Shifter Chirps */
    if(gMmwMssMCB.cfg.ctrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_FRAME){

        chirpStartIdx = gMmwMssMCB.cfg.ctrlCfg.u.frameCfg[0].frameCfg.chirpStartIdx;
        chirpEndIdx   = gMmwMssMCB.cfg.ctrlCfg.u.frameCfg[0].frameCfg.chirpEndIdx;

        /* Phase multipliers have been computed; phase shift for xth chirp = (x-1) * phase multipler */
        for (chirpIdx = chirpStartIdx; chirpIdx <= chirpEndIdx; chirpIdx++){

            chirpPhaseMultiplier = (chirpIdx - chirpStartIdx);

            /* Populate the chirp configuration: */
            phaseShiftCfg.chirpStartIdx   = (chirpEndIdx + 1 - chirpPhaseMultiplier) % numTxTotalDivisor;
            phaseShiftCfg.chirpEndIdx     = (chirpEndIdx + 1 - chirpPhaseMultiplier) % numTxTotalDivisor;

            /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
            phaseShiftCfg.tx0PhaseShift    =
                        ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[0]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

            /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
            phaseShiftCfg.tx1PhaseShift    =
                        ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[1]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

            /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
            phaseShiftCfg.tx2PhaseShift    =
                        ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[2]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

            /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
            phaseShiftCfg.tx3PhaseShift    =
                        ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[3]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

            /* Add the chirp to the profile */
            if (MMWave_addPhaseShiftChirp (gMmwMssMCB.ctrlHandle, &phaseShiftCfg, &errCode) == NULL)
            {
                /* Error: Unable to add the phase shifter chirp. Return the error code. */
                CLI_write ("Error: Unable to add the phase shifter chirp.\n");
                return errCode;
            }
        }
    }

    else if(gMmwMssMCB.cfg.ctrlCfg.dfeDataOutputMode == MMWave_DFEDataOutputMode_ADVANCED_FRAME){
        uint8_t numOfSubFrames = gMmwMssMCB.cfg.ctrlCfg.u.advancedFrameCfg[0].frameCfg.frameSeq.numOfSubFrames;
        uint8_t subFrameIdx = 0;
        for(subFrameIdx =0; subFrameIdx < numOfSubFrames; subFrameIdx++ ){
            chirpStartIdx = gMmwMssMCB.cfg.ctrlCfg.u.advancedFrameCfg[0].frameCfg.frameSeq.subFrameCfg[subFrameIdx].chirpStartIdx;
            chirpEndIdx   = gMmwMssMCB.cfg.ctrlCfg.u.advancedFrameCfg[0].frameCfg.frameSeq.subFrameCfg[subFrameIdx].numOfChirps + chirpStartIdx - 1;

            /* Phase multipliers have been computed; phase shift for xth chirp = (x-1) * phase multipler */
            for (chirpIdx = chirpStartIdx; chirpIdx <= chirpEndIdx; chirpIdx++){

                // printf("Chirp %d\n", chirpIdx);
                chirpPhaseMultiplier = (chirpIdx - chirpStartIdx);

                /* Populate the chirp configuration: */
                phaseShiftCfg.chirpStartIdx   = (chirpEndIdx + 1 - chirpPhaseMultiplier) % numTxTotalDivisor + chirpStartIdx;
                phaseShiftCfg.chirpEndIdx     = (chirpEndIdx + 1 - chirpPhaseMultiplier) % numTxTotalDivisor + chirpStartIdx;

                // printf("phaseShiftCfg.chirpEndIdx: %d\n", phaseShiftCfg.chirpEndIdx);

                /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
                phaseShiftCfg.tx0PhaseShift   =
                            ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[0]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

                /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
                phaseShiftCfg.tx1PhaseShift    =
                            ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[1]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

                /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
                phaseShiftCfg.tx2PhaseShift    =
                            ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[2]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

                /* 1 LSB of phaseShiftCfg.txPhaseShift = 360/2^6 = 5.625 degrees Valid range: 0 to 63 */
                phaseShiftCfg.tx3PhaseShift    =
                            ((uint32_t) MATHUTILS_ROUND_FLOAT(((float) ((chirpPhaseMultiplier * phaseShiftMultiplier[3]) % numTxTotalDivisor) / numTxTotalDivisor) * (1U << 6))) << 2;

                /* Add the chirp to the profile */
                if (MMWave_addPhaseShiftChirp (gMmwMssMCB.ctrlHandle, &phaseShiftCfg, &errCode) == NULL)
                {
                /* Error: Unable to add the phase shifter chirp. Return the error code. */
                    CLI_write ("Error: Unable to add the phase shifter chirp.\n");
                    return errCode;
                }
                // printf("Phase shift values- chirp %d\n", chirpIdx);
                // printf("%d\n%d\n%d\n%d\n\n", phaseShiftCfg.tx0PhaseShift/4, phaseShiftCfg.tx1PhaseShift/4, phaseShiftCfg.tx2PhaseShift/4, phaseShiftCfg.tx3PhaseShift/4);
            }
        }
    }

    return errCode;
}

/**
 *  @b Description
 *  @n
 *      Returns number of empty subbands
 *
 *  @retval
 *      Success     - Number of empty subbands
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_getNumEmptySubBands(uint32_t numTxAntennas){

    int32_t numBandsEmpty;
    /* Empty subbands */
    switch (numTxAntennas)
    {
        case 2:
            numBandsEmpty = 1;
            break;
        case 3:
            numBandsEmpty = 1;
            break;
        case 4:
            numBandsEmpty = 2;
            break;
        default:
            numBandsEmpty = -1;
            goto exit;
    }

exit:
    return numBandsEmpty;
}

/**
 *  @b Description
 *  @n
 *      MMW demo helper Function to configure sensor. User need to fill gMmwMssMCB.cfg.ctrlCfg and
 *      add profiles/chirp to mmWave before calling this function
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_configSensor(void)
{
    int32_t     errCode = 0;

    errCode = MmwDemo_configPhaseShifterChirps();
    if(errCode != 0){
        goto exit;
    }

    /* Configure the mmWave module: */
    if (MMWave_config (gMmwMssMCB.ctrlHandle, &gMmwMssMCB.cfg.ctrlCfg, &errCode) < 0)
    {
        MMWave_ErrorLevel   errorLevel;
        int16_t             mmWaveErrorCode;
        int16_t             subsysErrorCode;

        /* Error: Report the error */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
        DebugP_log ("Error: mmWave Config failed [Error code: %d Subsystem: %d]\n",
                        mmWaveErrorCode, subsysErrorCode);
        goto exit;
    }
    else
    {
        errCode = MmwDemo_dataPathConfig();
    }

exit:
    return errCode;
}

/**
 *  @b Description
 *  @n
 *      mmw demo helper Function to start sensor.
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_startSensor(void)
{
    int32_t     errCode;
    MMWave_CalibrationCfg   calibrationCfg;

    /*****************************************************************************
     * Data path :: start data path first - this will pend for DPC to ack
     *****************************************************************************/
    MmwDemo_dataPathStart();

    /*****************************************************************************
     * RF :: now start the RF and the real time ticking
     *****************************************************************************/
    /* Initialize the calibration configuration: */
    memset ((void *)&calibrationCfg, 0, sizeof(MMWave_CalibrationCfg));
    /* Populate the calibration configuration: */
    calibrationCfg.dfeDataOutputMode = gMmwMssMCB.cfg.ctrlCfg.dfeDataOutputMode;
    calibrationCfg.u.chirpCalibrationCfg.enableCalibration    = true;
    /* SEEKER PATCH 2026-04-30: disable periodic RF calibration. TI ships
     * this enabled by default which interrupts LVDS streaming periodically
     * and accumulates BSS state until the LVDS port halts (observed
     * halt @ ~700-1300 frames / ~35-37 s of continuous streaming).
     * One-time calibration at sensor start is retained. */
    calibrationCfg.u.chirpCalibrationCfg.enablePeriodicity    = false;
    calibrationCfg.u.chirpCalibrationCfg.periodicTimeInFrames = 10U;
    calibrationCfg.u.chirpCalibrationCfg.reportEn             = 1;

    DebugP_logInfo("App: MMWave_start Issued\n");

    DebugP_log("Starting Sensor (issuing MMWave_start)\n");

    /* Start the mmWave module: The configuration has been applied successfully. */
    if (MMWave_start(gMmwMssMCB.ctrlHandle, &calibrationCfg, &errCode) < 0)
    {
        MMWave_ErrorLevel   errorLevel;
        int16_t             mmWaveErrorCode;
        int16_t             subsysErrorCode;

        /* Error/Warning: Unable to start the mmWave module */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);
        DebugP_log ("Error: mmWave Start failed [mmWave Error: %d Subsys: %d]\n", mmWaveErrorCode, subsysErrorCode);
        /* datapath has already been moved to start state; so either we initiate a cleanup of start sequence or
           assert here and re-start from the beginning. For now, choosing the latter path */
        MmwDemo_debugAssert(0);
        return -1;
    }

    gMmwMssMCB.sensorStartCount++;
    return 0;
}

/**
 *  @b Description
 *  @n
 *      Epilog processing after sensor has stopped
 *
 *  @retval None
 */
static void MmwDemo_sensorStopEpilog(void)
{
    /* Note data path has completely stopped due to
     * end of frame, so we can do non-real time processing like prints on
     * console */
    DebugP_log("Data Path Stopped (last frame processing done)\n");
}


/**
 *  @b Description
 *  @n
 *      Stops the RF and datapath for the sensor. Blocks until both operation are completed.
 *      Prints epilog at the end.
 *
 *  @retval  None
 */
void MmwDemo_stopSensor(void)
{
    int32_t errCode;

    /* Stop sensor RF, data path will be stopped after RF stop is completed */
    MmwDemo_mmWaveCtrlStop();

    /* Wait until DPM_stop is completed */
    SemaphoreP_pend(&gMmwMssMCB.DPMstopSemHandle, SystemP_WAIT_FOREVER);

#ifdef LVDS_STREAM
    /* Delete any active streaming session */
    if(gMmwMssMCB.lvdsStream.hwSessionHandle != NULL)
    {
        /* Evaluate need to deactivate h/w session:
         * One sub-frame case:
         *   if h/w only enabled, deactivation never happened, hence need to deactivate
         *   if h/w and s/w both enabled, then s/w would leave h/w activated when it is done
         *   so need to deactivate
         *   (only s/w enabled cannot be the case here because we are checking for non-null h/w session)
         * Multi sub-frame case:
         *   Given stop, we must have re-configured the next sub-frame by now which is next of the
         *   last sub-frame i.e we must have re-configured sub-frame 0. So if sub-frame 0 had
         *   h/w enabled, then it is left in active state and need to deactivate. For all
         *   other cases, h/w was already deactivated when done.
         */
        if ((gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames == 1) ||
            ((gMmwMssMCB.objDetCommonCfg.preStartCommonCfg.numSubFrames > 1) &&
             (gMmwMssMCB.subFrameCfg[0].lvdsStreamCfg.dataFmt != MMW_DEMO_LVDS_STREAM_CFG_DATAFMT_DISABLED))
           )
        {
            if (CBUFF_deactivateSession(gMmwMssMCB.lvdsStream.hwSessionHandle, &errCode) < 0)
            {
                DebugP_log("CBUFF_deactivateSession failed with errorCode = %d\n", errCode);
                MmwDemo_debugAssert(0);
            }
        }
        MmwDemo_LVDSStreamDeleteHwSession();
    }

    /* Delete s/w session if it exists. S/w session never needs to be deactivated in stop because
     * it always (unconditionally) deactivates itself upon completion.
     */
    if(gMmwMssMCB.lvdsStream.swSessionHandle != NULL)
    {
        MmwDemo_LVDSStreamDeleteSwSession();
    }
#endif

    /* Print epilog */
    MmwDemo_sensorStopEpilog();

    gMmwMssMCB.sensorStopCount++;

    /* print for user */
    DebugP_log("Sensor has been stopped: startCount: %d stopCount %d\n",
                  gMmwMssMCB.sensorStartCount,gMmwMssMCB.sensorStopCount);
}

/**************************************************************************
 ******************** Millimeter Wave Demo init Functions ************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      Platform specific hardware initialization.
 *
 *  @param[in]  config     Platform initialization configuraiton
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_platformInit(MmwDemo_platformCfg *config)
{
    /* Initialize the DEMO configuration: */
    config->sysClockFrequency   = MSS_SYS_VCLK;
    config->loggingBaudRate     = 892857;
    config->commandBaudRate     = 115200;

}

#ifdef LVDS_STREAM
/**
 *  @b Description
 *  @n
 *      Checks for EDMA errors, used on devices where error interrupts are not connected
 *      to the CPU. Current use case is for LVDS, note it is not very useful to
 *      check for edma errors within the CBUFF session completion interrupts
 *      because they will not happen if edma had errors. So this API should be
 *      called at opportune places in the application code, typically at some time
 *      later than the triggering of the session when it is roughly expected that
 *      the session would have completed by that time.
 */
static void MmwDemo_checkEdmaErrors(void)
{
    bool        isAnyError = false;
    uint32_t    baseAddr = 0U;

    baseAddr = EDMA_getBaseAddr(gMmwMssMCB.edmaHandle);
    DebugP_assert(baseAddr != 0);

    isAnyError = ((EDMA_getErrIntrStatus(baseAddr) != 0U) ||
                   (EDMA_errIntrHighStatusGet(baseAddr) != 0U));

    if (isAnyError == true)
    {
        DebugP_log("EDMA channel controller has errors, see gMmwMssMCB.EDMA_errorInfo\n");
        MmwDemo_debugAssert(0);
    }
}
#endif

/**
 *  @b Description
 *  @n
 *      Calibration save/restore initialization
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_calibInit(void)
{
    int32_t        retVal = 0;
    rlVersion_t    verArgs;

    /* Initialize verArgs */
    memset((void *)&verArgs, 0, sizeof(rlVersion_t));

    /* Calibration save/restore init */
    gMmwMssMCB.calibCfg.sizeOfCalibDataStorage = sizeof(MmwDemo_calibData);
    gMmwMssMCB.calibCfg.calibDataHdr.magic = MMWDEMO_CALIB_STORE_MAGIC;
    memcpy((void *)& gMmwMssMCB.calibCfg.calibDataHdr.linkVer, (void *)&verArgs.mmWaveLink, sizeof(rlSwVersionParam_t));
    memcpy((void *)& gMmwMssMCB.calibCfg.calibDataHdr.radarSSVer, (void *)&verArgs.rf, sizeof(rlFwVersionParam_t));

    /* Check if Calibration data is over the Reserved storage */
    if(gMmwMssMCB.calibCfg.sizeOfCalibDataStorage   <= MMWDEMO_CALIB_FLASH_SIZE)
    {
        gMmwMssMCB.calibCfg.calibDataHdr.hdrLen = sizeof(MmwDemo_calibDataHeader);
        gMmwMssMCB.calibCfg.calibDataHdr.dataLen= sizeof(MmwDemo_calibData) - sizeof(MmwDemo_calibDataHeader);

        /* Resets calibration data */
        memset((void *)&gCalibDataStorage, 0, sizeof(MmwDemo_calibData));

        retVal = mmwDemo_flashInit();
    }
    else
    {
        DebugP_log ("Error: Calibration data size is bigger than reserved size\n");
        retVal = -1;
    }

    return retVal;

}

/**
 *  @b Description
 *  @n
 *      This function retrieves the calibration data from front end and saves it in flash.
 *
 *  @param[in]  ptrCalibDataHdr     	Pointer to Calibration data header
 *  @param[in]  ptrCalibrationData      Pointer to Calibration data
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_calibSave(MmwDemo_calibDataHeader *ptrCalibDataHdr, MmwDemo_calibData  *ptrCalibrationData)
{
    uint32_t         flashOffset;
    int32_t          retVal = 0;

    /* Calculate the read size in bytes */
    flashOffset = gMmwMssMCB.calibCfg.flashOffset;

    /* Copy header  */
    memcpy((void *)&(ptrCalibrationData->calibDataHdr), ptrCalibDataHdr, sizeof(MmwDemo_calibDataHeader));

    /* Flash calibration data */
    retVal = mmwDemo_flashWrite(flashOffset, (uint8_t *)ptrCalibrationData, sizeof(MmwDemo_calibData));
    if(retVal < 0)
    {
        /* Flash Header failed */
        DebugP_log ("Error: MmwDemo failed flashing calibration data with error[%d].\n", retVal);
    }
    return(retVal);
}


/**
 *  @b Description
 *  @n
 *      This function reads calibration data from flash and send it to front end through MMWave_open()
 *
 *  @param[in]  ptrCalibData     	Pointer to Calibration data
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
static int32_t MmwDemo_calibRestore(MmwDemo_calibData  *ptrCalibData)
{
    MmwDemo_calibDataHeader    *pDataHdr;
    int32_t                     retVal = 0;
    uint32_t                    flashOffset;

    pDataHdr = &(ptrCalibData->calibDataHdr);

    /* Calculate the read size in bytes */
    flashOffset = gMmwMssMCB.calibCfg.flashOffset;

    /* Read calibration data header */
    if(mmwDemo_flashRead(flashOffset, (uint8_t *)pDataHdr, sizeof(MmwDemo_calibData)) < 0)
    {
        /* Error: only one can be enable at at time */
        DebugP_log ("Error: MmwDemo failed when reading calibration data from flash.\n");
        return -1;
    }

    /* Validate data header */
    if( (pDataHdr->magic != MMWDEMO_CALIB_STORE_MAGIC) ||
         (pDataHdr->hdrLen != gMmwMssMCB.calibCfg.calibDataHdr.hdrLen) ||
         (pDataHdr->dataLen != gMmwMssMCB.calibCfg.calibDataHdr.dataLen) )
    {
        /* Header validation failed */
        DebugP_log ("Error: MmwDemo calibration data header validation failed.\n");
        retVal = -1;
    }
    /* Matching mmwLink version:
         In this demo, we would like to save/restore with the matching mmwLink and RF FW version.
         However, this logic can be changed to use data saved from previous mmwLink and FW releases,
         as long as the data format of the calibration data matches.
     */
    else if(memcmp((void *)&pDataHdr->linkVer, (void *)&gMmwMssMCB.calibCfg.calibDataHdr.linkVer, sizeof(rlSwVersionParam_t)) != 0)
    {
        DebugP_log ("Error: MmwDemo failed mmwLink version validation when restoring calibration data.\n");
        retVal = -1;
    }
    else if(memcmp((void *)&pDataHdr->radarSSVer, (void *)&gMmwMssMCB.calibCfg.calibDataHdr.radarSSVer, sizeof(rlFwVersionParam_t)) != 0)
    {
        DebugP_log ("Error: MmwDemo failed RF FW version validation when restoring calibration data.\n");
        retVal = -1;
    }
    return(retVal);
}


/**
 *  @b Description
 *  @n
 *      System Initialization Task which initializes the various
 *      components in the system.
 *
 *  @retval
 *      Not Applicable.
 */

static void MmwDemo_initTask(void* args)
{
    int32_t             errCode;
    MMWave_InitCfg      initCfg;
    DPM_InitCfg         dpmInitCfg;
    int32_t syncStatus;
#if defined(SOC_AWR2x44LC)
    int32_t syncStatusRefVal = CSL_FMKR(DPM_DSS_CM4_BOOT_INFO_BIT_POS, DPM_MSS_BOOT_INFO_BIT_POS, 3);
#else
    int32_t syncStatusRefVal = CSL_FMKR(DPM_DSS_BOOT_INFO_BIT_POS, DPM_MSS_BOOT_INFO_BIT_POS, 7);
#endif

    int32_t             i;
    MMWave_ErrorLevel       errorLevel;
    int16_t                 mmWaveErrorCode;
    int16_t                 subsysErrorCode;

    Drivers_open();
    Board_driversOpen();

    /* ============================================================
     * SEEKER PATCH 2026-05-05: disable external nRESET pad.
     *
     * Root cause of the 5-sec reset loop: the FT4232H on AWR's J10
     * USB drives nRESET (Port C bit 6) and SOP[2:0] (Port D bits 2/3/4)
     * directly via wiring on the AWR2944PEVM (per SPRUJ22C §2.8/2.10
     * + schematic PROC194A_FTDI.SchDoc). When the host has no app
     * holding the FTDI MPSSE channel open, the FTDI pins drift,
     * glitching nRESET every ~5 sec.
     *
     * Fix: MSS_TOPRCM.WARM_RESET_CONFIG.PAD_BYPASS = 0 disables the
     * external nRESET pad entirely. After this write, FTDI glitches
     * on nRESET have zero effect on the running chip. SW_RST and
     * WDOG_RST_EN paths remain active for legitimate SDK-driven
     * resets. SOP pins are already latched at POR, so FTDI drift on
     * Port D is harmless after boot.
     *
     * Field encoding: WARM_RESET_CONFIG bits [2:0] = PAD_BYPASS,
     * 3-bit triple-vote, default 0x7. Setting to 0x0 disables.
     * SDK uses the same unlock/RMW pattern in
     * drivers/watchdog/v0/soc/awr2x44p/watchdog_soc.c:74.
     *
     * Verified working in mmw_studio_cli build over multi-hour
     * uptime (heartbeat #3001+ in the May-5 live test).
     * ============================================================ */
    {
        volatile uint32_t *warmCfg = (volatile uint32_t *)(0x02140100U);

        SOC_controlModuleUnlockMMR(SOC_DOMAIN_ID_MSS_TOP_RCM, 0);
        uint32_t before = *warmCfg;
        *warmCfg = before & ~(0x7U << 0);   /* clear PAD_BYPASS[2:0] */
        uint32_t after = *warmCfg;
        SOC_controlModuleLockMMR(SOC_DOMAIN_ID_MSS_TOP_RCM, 0);

        DebugP_log("[BOOT] PAD_BYPASS disable: WARM_RESET_CONFIG 0x%08X -> 0x%08X\r\n",
                   (unsigned)before, (unsigned)after);
    }

    MmwDemo_BoardInit();

    /* Debug Message: */
    DebugP_log ("**********************************************\n");
    DebugP_log ("Debug: Launching the MMW Demo on MSS\n");
    DebugP_log ("**********************************************\n");

    /* Debug Message: */
    DebugP_log("Debug: Launched the Initialization Task\n");

    /*****************************************************************************
     * Initialize the mmWave SDK components:
     *****************************************************************************/

    /* Initialize Last Frame data Set flag for first frame. */
    gMmwMssMCB.stats.isLastFrameDataProcessed = true;

#ifdef LVDS_STREAM
    gMmwMssMCB.edmaHandle = gEdmaHandle[CONFIG_EDMA0];

    /* Initialize LVDS streaming components */
    if ((errCode = MmwDemo_LVDSStreamInit()) < 0 )
    {
        DebugP_log ("Error: MMWDemoDSS LVDS stream init failed with Error[%d]\n",errCode);
        return;
    }

    /* Configure Pad registers for LVDS. */
    HW_WR_REG32(CSL_MSS_TOPRCM_U_BASE + CSL_MSS_TOPRCM_LVDS_PAD_CTRL0 , 0x0);
    HW_WR_REG32(CSL_MSS_TOPRCM_U_BASE + CSL_MSS_TOPRCM_LVDS_PAD_CTRL1 , 0x02000000);

    /*The delay below is needed only if the DCA1000EVM is being used to capture the data traces.
      This is needed because the DCA1000EVM FPGA needs the delay to lock to the
      bit clock before they can start capturing the data correctly. */
    ClockP_usleep(12 * 1000);
#endif

    /* initialize cq configs to invalid profile index to be able to detect
     * unconfigured state of these when monitors for them are enabled.
     */
    for(i = 0; i < RL_MAX_PROFILES_CNT; i++)
    {
        gMmwMssMCB.cqSatMonCfg[i].profileIndx    = (RL_MAX_PROFILES_CNT + 1);
        gMmwMssMCB.cqSigImgMonCfg[i].profileIndx = (RL_MAX_PROFILES_CNT + 1);
    }

    /* Platform specific configuration */
    MmwDemo_platformInit(&gMmwMssMCB.cfg.platformCfg);

    /* Workaround for Errata ANA#46: Spurs caused due to data transfer activity */
    if (MmwDemo_registerChirpStartInterrupt() != 0)
    {
        CLI_write("Error: Failed to register chirp start interrupts\r\n");
        DebugP_assert(0);
    }

    /* Open the UART Instance */
    gMmwMssMCB.commandUartHandle = gUartHandle[CONFIG_UART0];
    if (gMmwMssMCB.commandUartHandle == NULL)
    {
        MmwDemo_debugAssert (0);
        return;
    }

    /* Open the Logging UART Instance: */
    gMmwMssMCB.loggingUartHandle = gUartHandle[CONFIG_UART1];
    if (gMmwMssMCB.loggingUartHandle == NULL)
    {
        DebugP_log("Error: Unable to open the Logging UART Instance\n");
        MmwDemo_debugAssert (0);
        return;
    }

    DebugP_logInfo("Both UART instances opened");

    /* Create binary semaphores which is used to signal DPM_start/DPM_stop/DPM_ioctl is done
     * to the sensor management task. The signalling (SemaphoreP_post) will be done
     * from DPM registered report function (which will execute in the DPM execute task context). */
    SemaphoreP_constructBinary(&gMmwMssMCB.DPMstartSemHandle, 0);
    SemaphoreP_constructBinary(&gMmwMssMCB.DPMstopSemHandle, 0);
    SemaphoreP_constructBinary(&gMmwMssMCB.DPMioctlSemHandle, 0);
    SemaphoreP_constructBinary(&gMmwMssMCB.UartExportSemHandle, 0);

    /* Create binary semaphore to pend Main task, */
    SemaphoreP_constructBinary(&gMmwMssMCB.demoInitTaskCompleteSemHandle, 0);

    /*****************************************************************************
     * mmWave: Initialization of the high level module
     *****************************************************************************/

    /* Initialize the mmWave control init configuration */
    memset ((void*)&initCfg, 0 , sizeof(MMWave_InitCfg));

    /* Populate the init configuration: */
    initCfg.domain                  = MMWave_Domain_MSS;
    initCfg.eventFxn                = MmwDemo_eventCallbackFxn;
    initCfg.linkCRCCfg.crcBaseAddr  = (uint32_t) AddrTranslateP_getLocalAddr(CONFIG_CRC0_BASE_ADDR);
    initCfg.linkCRCCfg.useCRCDriver = 1U;
    initCfg.linkCRCCfg.crcChannel   = CRC_CHANNEL_1;
    initCfg.cfgMode                 = MMWave_ConfigurationMode_FULL;

    /* Initialize and setup the mmWave Control module */
    gMmwMssMCB.ctrlHandle = MMWave_init (&initCfg, &errCode);
    if (gMmwMssMCB.ctrlHandle == NULL)
    {
         /* Error: Unable to initialize the mmWave control module */
        MMWave_decodeError (errCode, &errorLevel, &mmWaveErrorCode, &subsysErrorCode);

        /* Error: Unable to initialize the mmWave control module */
        DebugP_log ("Error: mmWave Control Initialization failed [Error code %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return;
    }
    DebugP_log ("Debug: mmWave Control Initialization was successful\n");

    /* Synchronization: This will synchronize the execution of the control module
     * between the domains. This is a prerequiste and always needs to be invoked. */
    if (MMWave_sync(gMmwMssMCB.ctrlHandle, &errCode) < 0)
    {
        /* Error: Unable to synchronize the mmWave control module */
        DebugP_log ("Error: mmWave Control Synchronization failed [Error code %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return;
    }
    DebugP_log ("Debug: mmWave Control Synchronization was successful\n");

    /*****************************************************************************
     * Launch the mmWave control execution task
     * - This should have a higher priroity than any other task which uses the
     *   mmWave control API
     *****************************************************************************/
    gMmwMssMCB.taskHandles.mmwCtrlTask = xTaskCreateStatic( MmwDemo_mmWaveCtrlTask,
                                      "mmwdemo_ctrl_task",
                                      MMWDEMO_MMWAVE_CTRL_TASK_STACK_SIZE,
                                      NULL,
                                      MMWDEMO_MMWAVE_CTRL_TASK_PRIORITY,
                                      gMmwCtrlTskStack,
                                      &gMmwMssMCB.taskHandles.mmwCtrlTaskObj );

    configASSERT(gMmwMssMCB.taskHandles.mmwCtrlTask != NULL);

#ifdef ENET_STREAM
   /*****************************************************************************
     * Launch the mmWave enet task
     *****************************************************************************/
    /* Create Enet configuration done semaphore */
    SemaphoreP_constructBinary(&gMmwMssMCB.enetCfg.EnetCfgDoneSemHandle, 0);

    gMmwMssMCB.taskHandles.enetTask = xTaskCreateStatic( enetTask,
                                      "enet_task",
                                      MMWDEMO_MMWAVE_ENET_TASK_STACK_SIZE,
                                      NULL,
                                      MMWDEMO_MMWAVE_ENET_TASK_PRIORITY,
                                      gMmwEnetTskStack,
                                      &gMmwMssMCB.taskHandles.enetTaskObj );

    configASSERT(gMmwMssMCB.taskHandles.enetTask != NULL);
#endif


    /* Initialization of the DPM Module */
    memset ((void *)&dpmInitCfg, 0, sizeof(DPM_InitCfg));

    /* Setup the configuration: */
    dpmInitCfg.localEndPt  = gRemoteCoreEndPt[CSL_CORE_ID_R5FSS0_0];
    dpmInitCfg.reportFxn   = MmwDemo_DPC_ObjectDetection_reportFxn;
    dpmInitCfg.setBitPos   = DPM_MSS_BOOT_INFO_BIT_POS;

    /* Initialize the DPM Module: */
    gMmwMssMCB.objDetDpmHandle = DPM_init (&dpmInitCfg, &errCode);
    if (gMmwMssMCB.objDetDpmHandle == NULL)
    {
        DebugP_log ("Error: Unable to initialize the DPM Module [Error: %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return;
    }

    /* Synchronization: This will synchronize the execution of the datapath module
     * between the domains. This is a prerequiste and always needs to be invoked. */
    while (1)
    {

        /* Get the synchronization status: */
        syncStatus = DPM_synch (gMmwMssMCB.objDetDpmHandle, &errCode);
        if (syncStatus < 0)
        {
            /* Error: Unable to synchronize the framework */
            DebugP_log ("Error: DPM Synchronization failed [Error code %d]\n", errCode);
            MmwDemo_debugAssert (0);
            return;
        }
        if (syncStatus == syncStatusRefVal)
        {
            /* Synchronization acheived: */
            break;
        }
        /* Sleep and poll again: */
        ClockP_usleep(1 * 1000U);
    }

    /* Calibration save/restore initialization */
    if(MmwDemo_calibInit()<0)
    {
        DebugP_log("Error: Calibration data initialization failed \n");
        MmwDemo_debugAssert (0);
    }

    /* Launch the UART Data Export Task */
    gMmwMssMCB.taskHandles.uartDataExportTask = xTaskCreateStatic( mmwDemo_mssUartDataExportTask,
                                           "mmwdemo_uart_task",
                                           MMWDEMO_UART_DATA_EXPORT_TASK_STACK_SIZE,
                                           NULL,
                                           MMWDEMO_UART_EXPORT_TASK_PRIORITY,
                                           gUartTskStack,
                                           &gMmwMssMCB.taskHandles.uartDataExportTaskObj );

    configASSERT(gMmwMssMCB.taskHandles.uartDataExportTask != NULL);

    /*****************************************************************************
     * Initialize the Profiler
     *****************************************************************************/
    CycleCounterP_reset();

    /*****************************************************************************
     * Initialize the CLI Module:
     *****************************************************************************/
    MmwDemo_CLIInit(MMWDEMO_CLI_TASK_PRIORITY);

    /* Never return for this task. */
    SemaphoreP_pend(&gMmwMssMCB.demoInitTaskCompleteSemHandle, SystemP_WAIT_FOREVER);

    /* The following line should never be reached. */
    DebugP_assertNoLog(0);
}

/**
 *  @b Description
 *  @n
 *      Performs Board Initialization
 *
 *  @retval
 *      Success -   true
 *  @retval
 *      Error   -   false
 */
static bool MmwDemo_BoardInit(void)
{
    /* Configure HSI Clock. */
    HW_WR_REG32(CSL_MSS_TOPRCM_U_BASE + CSL_MSS_TOPRCM_HSI_CLK_SRC_SEL, 0x333);

    return true;
}

/**
 *  @b Description
 *  @n
 *      This function computes Modulation frequency divider
 *      (7 bit mantissa and 3 bit exponent) and Modulation depth
 *      (3 bit integer and 18 bit fraction) based on user provided inputs
 *      Modulation rate, Modulation depth in %
 *
 *  @param[in] refClk Pre-divided reference clock input for the ADPLL
 *  @param[in] dpllM ADPLL Feedback Multiplier
 *  @param[in] ptrdpllCfg Pointer to SSC Config (MmwDemo_spreadSpectrumConfig)
 *
 *  @retval
 *      FRACCTRL Register value
 *
 */

static uint32_t computeSscFactCtrlVal(const uint32_t refClk, const uint16_t dpllM, MmwDemo_spreadSpectrumConfig *ptrdpllCfg)
{
    float modRateSel = 0.0f;
    float deltaMStep = 0.0f;
    uint32_t delatMStepInt = 0U;
    float deltaMStepFrac = 0.0f;
    uint32_t deltaMStepFracInt = 0U;
    uint32_t modFreqDivExponent = 0U;
    uint32_t modFreqDivMantissa = 0U;

    modRateSel = (refClk * 1000.0f) / (4.0f * ptrdpllCfg->modRate);

    modFreqDivExponent = (uint32_t) (floor(modRateSel/MAX_MOD_FREQ_DIVIDER_MANTISSA));

    modFreqDivMantissa = (uint32_t) (floor(modRateSel / pow(2.0f, modFreqDivExponent)));

    ptrdpllCfg->modRate = refClk * 1000 / (4 * modFreqDivMantissa * (pow(2, modFreqDivExponent)));

    if(modFreqDivExponent <= 3U)
    {
        deltaMStep = (ptrdpllCfg->modDepth * dpllM) / (100.0f * modFreqDivMantissa * pow(2.0f, modFreqDivExponent));
    }
    else
    {
        deltaMStep = (ptrdpllCfg->modDepth * dpllM) / (100.0f * modFreqDivMantissa * 8.0f);
    }

    delatMStepInt = (uint32_t) (deltaMStep + 0.5f);

    deltaMStepFrac = deltaMStep - delatMStepInt;

    deltaMStepFracInt = (uint32_t) ceilf(deltaMStepFrac * (1U << 18U));

    ptrdpllCfg->modDepth = 100 * ( (deltaMStepFrac/ (1 << 18) ) * modFreqDivMantissa * ( 1U << modFreqDivExponent) / dpllM);

    return (deltaMStepFracInt + (delatMStepInt * (1U << 18U)) +
            (modFreqDivMantissa * (1U << 21U)) + (modFreqDivExponent * (1U << 28U)) +
            (ptrdpllCfg->downSpread * (1U << 31U)));
}


/**
 *  @b Description
 *  @n
 *      Performs Spread Spectrum Configuration (SSC)
 *      SSC reduces the Electromagnetic Interference by spreading it out across frequencies instead of concentrating at a single frequency
 *
 *  @retval
 *      none
 */
void MMWDemo_configSSC(void)
{
    uint16_t dpllM = 0U;
    uint16_t dpllN = 0U;
    uint32_t finp = 0U; /* should be XTAL in MHz */
    uint32_t refClk = 0U;
    uint8_t xtalSopVal = 0U;

    CSL_mss_toprcmRegs *ptrMssTopRcmRegs = (CSL_mss_toprcmRegs *)CSL_MSS_TOPRCM_U_BASE;

    /*  Check XTAL frequency based on SOP register SOP_MODE_LAT_4_0 [Bit4 and Bit3]:
        Bit4, Bit3 : R2 (SOP4_MSS_UARTA_TX), P3 (SOP3_MSS_UARTB_TX)
        00 (0x0) -> 40MHz (0x28)
        11 (0x3) -> 50MHz (0x32)
        other values are not supported
    */
    xtalSopVal = CSL_FEXTR(ptrMssTopRcmRegs->ANA_REG_WU_MODE_REG_LOWV, 6U, 5U);

    if( xtalSopVal == 0x0U)
    {
        finp = 40U;
    }
    else if (xtalSopVal == 0x3U)
    {
        finp = 50U;
    }
    else
    {
        DebugP_assert(FALSE);
    }

    if(gMmwMssMCB.coreAdpllSscCfg.isEnable)
    {
        dpllM = CSL_FEXT(ptrMssTopRcmRegs->PLL_CORE_MN2DIV,
                         MSS_TOPRCM_PLL_CORE_MN2DIV_PLL_CORE_MN2DIV_M);

        dpllN = CSL_FEXT(ptrMssTopRcmRegs->PLL_CORE_M2NDIV,
                         MSS_TOPRCM_PLL_CORE_M2NDIV_PLL_CORE_M2NDIV_N);

        /** refClk - Pre-divided reference clock input for the ADPLL
         * CLKINP/(N+1) where N is Pre divider and CLKINP is OSC Clock
         */
        refClk = finp/(dpllN + 1);

        ptrMssTopRcmRegs->PLL_CORE_FRACCTRL = computeSscFactCtrlVal(refClk, dpllM, &gMmwMssMCB.coreAdpllSscCfg);

        CSL_FINS(ptrMssTopRcmRegs->PLL_CORE_CLKCTRL, MSS_TOPRCM_PLL_CORE_CLKCTRL_PLL_CORE_CLKCTRL_ENSSC, 1U);

    }

    if(gMmwMssMCB.dspAdpllSscCfg.isEnable)
    {
        dpllM = CSL_FEXT(ptrMssTopRcmRegs->PLL_DSP_MN2DIV,
                         MSS_TOPRCM_PLL_DSP_MN2DIV_PLL_DSP_MN2DIV_M);

        dpllN = CSL_FEXT(ptrMssTopRcmRegs->PLL_DSP_M2NDIV,
                         MSS_TOPRCM_PLL_DSP_M2NDIV_PLL_DSP_M2NDIV_N);

        refClk = finp/(dpllN + 1);

        ptrMssTopRcmRegs->PLL_DSP_FRACCTRL = computeSscFactCtrlVal(refClk, dpllM, &gMmwMssMCB.dspAdpllSscCfg);

        CSL_FINS(ptrMssTopRcmRegs->PLL_DSP_CLKCTRL, MSS_TOPRCM_PLL_DSP_CLKCTRL_PLL_DSP_CLKCTRL_ENSSC, 1U);

    }

    if(gMmwMssMCB.perAdpllSscCfg.isEnable)
    {
        dpllM = CSL_FEXT(ptrMssTopRcmRegs->PLL_PER_MN2DIV,
                         MSS_TOPRCM_PLL_PER_MN2DIV_PLL_PER_MN2DIV_M);

        dpllN = CSL_FEXT(ptrMssTopRcmRegs->PLL_PER_M2NDIV,
                         MSS_TOPRCM_PLL_PER_M2NDIV_PLL_PER_M2NDIV_N);

        refClk = finp/(dpllN + 1);

        ptrMssTopRcmRegs->PLL_PER_FRACCTRL = computeSscFactCtrlVal(refClk, dpllM, &gMmwMssMCB.perAdpllSscCfg);

        CSL_FINS(ptrMssTopRcmRegs->PLL_PER_CLKCTRL, MSS_TOPRCM_PLL_PER_CLKCTRL_PLL_PER_CLKCTRL_ENSSC, 1U);
    }

    return;
}

/**
 *  @b Description
 *  @n
 *      Entry point into the Millimeter Wave Demo
 *
 *  @retval
 *      Not Applicable.
 */
int32_t main (void)
{
    /* init SOC specific modules */
    System_init();
    Board_init();

    gMmwMssMCB.taskHandles.initTask = xTaskCreateStatic( MmwDemo_initTask,
                                  "mmwdemo_init_task",
                                  MMWDEMO_INIT_TASK_STACK_SIZE,
                                  NULL,
                                  MMWDEMO_INIT_TASK_PRI,
                                  gAppMainTskStack,
                                  &gMmwMssMCB.taskHandles.initTaskObj );
    configASSERT(gMmwMssMCB.taskHandles.initTask != NULL);

    /* Start the scheduler to start the tasks executing. */
    vTaskStartScheduler();

    /* The following line should never be reached because vTaskStartScheduler()
    will only return if there was not enough FreeRTOS heap memory available to
    create the Idle and (if configured) Timer tasks.  Heap management, and
    techniques for trapping heap exhaustion, are described in the book text. */
    DebugP_assertNoLog(0);

    return 0;
}
