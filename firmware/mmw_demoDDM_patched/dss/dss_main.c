/**
 *   @file  dss_main.c
 *
 *   @brief
 *      This is the main file which implements the millimeter wave Demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020 - 2025 Texas Instruments, Inc.
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

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* MCU PLUS SDK Include Files */
#include <kernel/dpl/CycleCounterP.h>
#include <kernel/dpl/SemaphoreP.h>
#include <ti_drivers_config.h>
#include <ti_board_config.h>
#include <ti_drivers_open_close.h>
#include <ti_board_open_close.h>
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/TaskP.h>
#include "FreeRTOS.h"
#include "task.h"

/* mmWave SDK Include Files: */
#include <ti/common/syscommon.h>
#include <ti/common/mmwavesdk_version.h>
#include <ti/datapath/dpc/objectdetection/objdethwaDDMA/objectdetection.h>
#include <ti/utils/mathutils/mathutils.h>
#include <ti/demo/awr2x44P/mmw_ddm/mmw_common.c>
#include <kernel/nortos/dpl/c66/Context_c66.h>
#include <kernel/freertos/dpl/common/ClockP_freertos_priv.h>

/* Task declarations */
#define MMWDEMO_DPC_TASK_PRI              (5U)
#define MMWDEMO_DPC_TASK_STACK_SIZE       (1U * 1024u)

/* DSP Power Gate, Underclock configurations */
#define MMWDEMO_DSP_CLK_SRC_XTAL                 (0x111U)
#define MMWDEMO_DSP_UC_ENABLE                    (0x2U)
#define MMWDEMO_DSP_PG_ENABLE                    (0x1U)
#define DSP_ICFG_PDCCMD                   (0x01810000U)
#define DSP_ICFG_PDCCMD_GEMPD             (16U)

static HwiP_Object      gDspSleepHwiObject;

/* Stack for tasks */
StackType_t gMmwDemo_dpcTaskStack[MMWDEMO_DPC_TASK_STACK_SIZE] __attribute__((aligned(64)));

/*! @brief   Semaphore Object to trigger Elevation Estimation Execution */
SemaphoreP_Object gDPCExecSemHandle;

/**
 * @brief
 *  Millimeter Wave Demo MCB
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Millimeter Wave demo
 */
typedef struct MmwDemo_DSS_MCB_t
{
    /*! @brief     DPM Handle */
    TaskHandle_t                 objDetDpcTaskHandle;

    /*! @brief     DPM task object */
    StaticTask_t                 objDetDpcTaskObj;

    /*! @brief   Semaphore Object to pend main task */
    SemaphoreP_Object             dpcTaskCompleteSemHandle;

    /*! @brief dpm Handle */
    DPM_Handle          objDetDpmHandle;

} MmwDemo_DSS_MCB;

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_DSS_MCB    gMmwDssMCB;

DPC_ObjectDetection_ElevEstCfg gElevEstCfg = {0};

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
    if(dataLen == sizeof(DPC_ObjectDetection_ElevEstCfg))
    {
        (void)memcpy((void*)&gElevEstCfg, (void*)data, dataLen);

        SemaphoreP_post(&gDPCExecSemHandle);
    }
    else
    {
        uint32_t msg = *((uint32_t*)data);
        switch (msg)
        {
            case MMWDEMO_DPC_RESULT:
            {
                SemaphoreP_post(&gDPCExecSemHandle);
                break;
            }
            default:
                break;
        }
    }
}

/**
 * @b Description
 * @n
 *      Function to call an idle instruction
 * 
 *  @retval
 *      Not Applicable.
 */
static void MMWDemo_dspCallIdleInstruction(void)
{
    asm(" idle ");
}

/**
 *  @b Description
 *  @n
 *      Callback function for PDC Interrupt
 *
 *  @retval
 *      Not Applicable.
 */
static void MMWDemo_dspPDCIntrCallBack (void *arg)
{
    *(volatile uint32_t *)DSP_ICFG_PDCCMD |= (uint32_t)0x1 << DSP_ICFG_PDCCMD_GEMPD;
    MMWDemo_dspCallIdleInstruction();

    return;
}

/**
 *  @b Description
 *  @n
 *      Register PDC Interrupt for DSP Sleep (118)
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_registerDSPPowerDownIntr(void)
{
    HwiP_Params      hwiPrms;
    int32_t         status = SystemP_SUCCESS;
    HwiP_Params_init(&hwiPrms);
    hwiPrms.intNum      = (uint32_t) CSL_DSS_INTR_PDC_INT;
    hwiPrms.callback    = &MMWDemo_dspPDCIntrCallBack;
    status              = HwiP_construct(&gDspSleepHwiObject, &hwiPrms);
    DebugP_assert(status == SystemP_SUCCESS);

    return;
}

/**
 *  @b Description
 *  @n
 *     Hook function called before Saving context for DSP Power Down
 *
 *  @retval
 *      Not Applicable.
 */
void MMWDemo_dspPreSave(void)
{
    CSL_dss_rcmRegs *ptrDssRcmRegs = (CSL_dss_rcmRegs *)CSL_DSS_RCM_U_BASE;
    TimerP_stop(gClockCtrl.timerBaseAddr);
    ptrDssRcmRegs->DSP_PD_CTRL |= 0x1U;
}

/**
 *  @b Description
 *  @n
 *     Hook function called after restoring context once DSP is Powered Up
 *
 *  @retval
 *      Not Applicable.
 */
void MMWDemo_dspPGPostRestore(void)
{
    CSL_dss_rcmRegs *ptrDssRcmRegs = (CSL_dss_rcmRegs *)CSL_DSS_RCM_U_BASE;
    contextSave.altPc = 0;
    ptrDssRcmRegs->DSP_PD_CTRL &= 0xFFFFFFFEU;
    TimerP_start(gClockCtrl.timerBaseAddr); // Restart RTIA Timer to resume task execution
}

/**
 *  @b Description
 *  @n
 *     Initialization for context saving
 *
 *  @retval
 *      Not Applicable.
 */
static void MMWDemo_dspPowerGateInit (void)
{
    contextSave.PreSaveFxn = MMWDemo_dspPreSave;
    contextSave.PostRestoreFxn = MMWDemo_dspPGPostRestore;
    contextSave.altPc = 0;
    contextSave.minTime = 50000; //In ns
    contextSave.ptrDssRcmRegs = (void *)CSL_DSS_RCM_U_BASE;
    contextSave.wakeupSrc = 16; /* DSP_PD_TRIGGER_WAKUP */
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
static void MmwDemo_dssInitTask(void* args)
{
#if !MSS_AOA_ENABLED
    uint32_t msg = 0;
    CSL_dss_rcmRegs *ptrDssRcmRegs = (CSL_dss_rcmRegs *)CSL_DSS_RCM_U_BASE;
#endif
    int32_t             errCode;
    DPM_InitCfg         dpmInitCfg;

    /* Register DSP Power Down Interrupt */
    MmwDemo_registerDSPPowerDownIntr();

    /* Initialize Power Gating configurations */
    MMWDemo_dspPowerGateInit();

    CycleCounterP_reset();

    /* Create binary semaphore to pend Main task, */
    (void)SemaphoreP_constructBinary(&gMmwDssMCB.dpcTaskCompleteSemHandle, 0);

    /*! @brief   Semaphore Object to handle DPC state */
    (void)SemaphoreP_constructBinary(&gDPCExecSemHandle, 0);

    /*****************************************************************************
     * Initialization of the DPM Module:
     *****************************************************************************/
    (void)memset ((void *)&dpmInitCfg, 0, sizeof(DPM_InitCfg));

    /* Setup the configuration: */
    dpmInitCfg.localEndPt  = gRemoteCoreEndPt[CSL_CORE_ID_C66SS0];
    dpmInitCfg.reportFxn   = MmwDemo_DPC_ObjectDetection_reportFxn;
    dpmInitCfg.setBitPos   = DPM_DSS_BOOT_INFO_BIT_POS;

    /* Initialize the DPM Module: */
    gMmwDssMCB.objDetDpmHandle = DPM_init (&dpmInitCfg, &errCode);
    if (gMmwDssMCB.objDetDpmHandle == NULL)
    {
        DebugP_log("Error: Unable to initialize the DPM Module [Error: %d]\n", errCode);
        DebugP_assert (false);
        return;
    }

    /* Synchronization: This will synchronize the execution of the control module
     * between the domains. This is a prerequiste and always needs to be invoked. */
    while (true)
    {
        int32_t syncStatus;

        /* Get the synchronization status: */
        syncStatus = DPM_synch (gMmwDssMCB.objDetDpmHandle, &errCode);
        if (syncStatus < 0)
        {
            /* Error: Unable to synchronize the framework */
            DebugP_log("Error: DPM Synchronization failed [Error code %d]\n", errCode);
            DebugP_assert (false);
            return;
        }
        if ((uint32_t)syncStatus == CSL_FMKR(DPM_DSS_BOOT_INFO_BIT_POS, DPM_MSS_BOOT_INFO_BIT_POS, 7U))
        {
            /* Synchronization acheived: */
            break;
        }
        /* Sleep and poll again: */
        ClockP_usleep(1U * 1000U);
    }

    /* wait for the configs to be available */
    /* if AoA proc runs on MSS, no config will be sent from MSS and execution will be stalled here */
    (void)SemaphoreP_pend(&gDPCExecSemHandle, SystemP_WAIT_FOREVER);

#if !MSS_AOA_ENABLED
    /* send the ack for config received */
    msg = MMWDEMO_DPC_ELEVEST_CFG;
    errCode = DPM_send(gMmwDssMCB.objDetDpmHandle, (void*)(&msg), sizeof(uint32_t), CSL_CORE_ID_R5FSS0_0);
    DebugP_assert(0U == errCode);

    while(1)
    {
        SemaphoreP_pend(&gDPCExecSemHandle, SystemP_WAIT_FOREVER);
        DPC_ObjectDetection_ExecuteResult *result = gElevEstCfg.commonCfg.result;

        DPC_ObjDet_estimateXYZ(&gElevEstCfg, 
                                    (DetObjParams*)SOC_phyToVirt((uint32_t)result->detObjList), /* Translate the detObjList address (DSS_L2) from M4 core view to DSS core view. */
                                    result->objOut,
                                    result->dopNumObjOut,
                                    &result->numObjOut);

        /* Notify the MSS that results are ready in L3 RAM for export. */
        msg = MMWDEMO_DPC_RESULT;
        errCode = DPM_send(gMmwDssMCB.objDetDpmHandle, (void*)(&msg), sizeof(uint32_t), CSL_CORE_ID_R5FSS0_0);
        DebugP_assert(0U == errCode);
        if ( gElevEstCfg.commonCfg.dspStateAfterFrameProc == MMWDEMO_DSP_UC_ENABLE)
        {
            /* Underclock DSP core (XTAL) */
            ptrDssRcmRegs->DSS_DSP_CLK_SRC_SEL = MMWDEMO_DSP_CLK_SRC_XTAL;
        }
        else if (gElevEstCfg.commonCfg.dspStateAfterFrameProc == MMWDEMO_DSP_PG_ENABLE)
        {
            /* Power Gate DSP */
            contextSave.altPc = 0x12345678;
        }
    }
#endif
}

/**
 *  @b Description
 *  @n
 *      Entry point into the Millimeter Wave Demo
 *
 *  @retval
 *      Not Applicable.
 */
int main (void)
{

    /* init SOC specific modules */
    System_init();
    Board_init();

    /* Initialize and populate the demo MCB */
    (void)memset ((void*)&gMmwDssMCB, 0, sizeof(gMmwDssMCB));

    /* It is the only task running on this core at highest priority */
    gMmwDssMCB.objDetDpcTaskHandle = xTaskCreateStatic( MmwDemo_dssInitTask,
                                  "MmwDemo_dssInitTask",
                                  MMWDEMO_DPC_TASK_STACK_SIZE,
                                  NULL,
                                  MMWDEMO_DPC_TASK_PRI,
                                  gMmwDemo_dpcTaskStack,
                                  &gMmwDssMCB.objDetDpcTaskObj );
    configASSERT(gMmwDssMCB.objDetDpcTaskHandle != NULL);

    /* Start the scheduler to start the tasks executing. */
    vTaskStartScheduler();

    /* The following line should never be reached because vTaskStartScheduler()
    will only return if there was not enough FreeRTOS heap memory available to
    create the Idle and (if configured) Timer tasks.  Heap management, and
    techniques for trapping heap exhaustion, are described in the book text. */
    DebugP_assertNoLog(0);

    return 0;
}
