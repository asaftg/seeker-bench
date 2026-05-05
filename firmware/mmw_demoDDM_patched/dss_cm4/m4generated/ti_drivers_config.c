/*
 *  Copyright (C) 2021 Texas Instruments Incorporated
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
/*
 * Auto generated file
 */

#include "ti_drivers_config.h"

/*
 * EDMA
 */
/* EDMA atrributes */
static EDMA_Attrs gEdmaAttrs[CONFIG_EDMA_NUM_INSTANCES] =
{
    {

        .baseAddr           = CSL_CM4_DSS_TPCC_A_U_BASE,
        .tcBaseAddr[0]    = CSL_CM4_DSS_TPTC_A0_U_BASE,
        .tcBaseAddr[1]    = CSL_CM4_DSS_TPTC_A1_U_BASE,
        .numTptc            = 2,
        .compIntrNumber     = CSL_CM4_INTR_DSS_TPCC_A_INTAGG,
        .compIntrNumberDirMap      = 0,
        .isErrIntrAvailable   = 0,
        .intrAggEnableAddr  = CSL_CM4_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_TPCC_A_INTAGG_MASK,
        .intrAggEnableMask  = 0x1FF & (~(2U << 0)),
        .intrAggStatusAddr  = CSL_CM4_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_TPCC_A_INTAGG_STATUS,
        .intrAggClearMask   = (2U << 0),
        .initPrms           =
        {
            .regionId     = 0,
            .queNum       = 0,
            .initParamSet = FALSE,
            .ownResource    =
            {
                .qdmaCh      = 0xFFU,
                .dmaCh[0]    = 0xFFFFFFFFU,
                .dmaCh[1]    = 0xFFFFFFFFU,
                .tcc[0]      = 0xFFFFFFFFU,
                .tcc[1]      = 0xFFFFFFFFU,
                .paramSet[0] = 0xFFFFFFFFU,
                .paramSet[1] = 0xFFFFFFFFU,
                .paramSet[2] = 0xFFFFFFFFU,
                .paramSet[3] = 0xFFFFFFFFU,
            },
            .reservedDmaCh[0]    = 0x00000001U,
            .reservedDmaCh[1]    = 0x00000000U,
        },
    },
};

/* EDMA objects - initialized by the driver */
static EDMA_Object gEdmaObjects[CONFIG_EDMA_NUM_INSTANCES];
/* EDMA driver configuration */
EDMA_Config gEdmaConfig[CONFIG_EDMA_NUM_INSTANCES] =
{
    {
        &gEdmaAttrs[CONFIG_EDMA0],
        &gEdmaObjects[CONFIG_EDMA0],
    },
};

uint32_t gEdmaConfigNum = CONFIG_EDMA_NUM_INSTANCES;

/*
 * HWA
 */
/* HWA atrributes */
HWA_Attrs gHwaAttrs[CONFIG_HWA_NUM_INSTANCES] =
{
    {
        .instanceNum                = 0U,
        .ctrlBaseAddr               = CSL_CM4_DSS_HWA_CFG_U_BASE,
        .paramBaseAddr              = CSL_CM4_DSS_HWA_PARAM_U_BASE,
        .ramBaseAddr                = CSL_CM4_DSS_HWA_WINDOW_RAM_U_BASE,
        .dssBaseAddr                = CSL_CM4_DSS_CTRL_U_BASE,
        .numHwaParamSets            = SOC_HWA_NUM_PARAM_SETS,
        .intNum1ParamSet            = CSL_CM4_INTR_DSS_HWA_PARAM_DONE_INTR1,
        .intNum2ParamSet            = CSL_CM4_INTR_DSS_HWA_PARAM_DONE_INTR2,
        .intNumDone                 = CSL_CM4_INTR_DSS_HWA_THREAD1_LOOP_INT,
        .intNumDoneALT              = CSL_CM4_INTR_DSS_HWA_THREAD2_LOOP_INT,
        .intNumLocalRamErr          = CSL_CM4_INTR_DSS_HWA_LOCAL_RAM_ERR,
        .intNum1ParamSetDirMap      = 0,
        .intNum2ParamSetDirMap      = 0,
        .intNumDoneDirMap           = 0,
        .intNumDoneALTDirMap        = 0,
        .intNumLocalRamErrDirMap    = 0,
        .numDmaChannels             = SOC_HWA_NUM_DMA_CHANNEL,
        .accelMemBaseAddr           = CSL_DSS_HWA_DMA0_U_BASE,
        .accelMemSize               = SOC_HWA_MEM_SIZE,
        .isConcurrentAccessAllowed  = true,
    },
};

/* HWA RAM atrributes */
HWA_RAMAttrs gHwaRamCfg[HWA_NUM_RAMS] =
{
    {CSL_CM4_DSS_HWA_WINDOW_RAM_U_BASE, CSL_CM4_DSS_HWA_WINDOW_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_MULT_RAM_U_BASE, CSL_CM4_DSS_HWA_MULT_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_DEROT_RAM_U_BASE, CSL_CM4_DSS_HWA_DEROT_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_SHUFFLE_RAM_U_BASE, CSL_CM4_DSS_HWA_SHUFFLE_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_HIST_THRESH_RAM_U_BASE, CSL_CM4_DSS_HWA_HIST_THRESH_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_2DSTAT_ITER_VAL_RAM_U_BASE, CSL_CM4_DSS_HWA_2DSTAT_ITER_VAL_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_2DSTAT_ITER_IDX_RAM_U_BASE, CSL_CM4_DSS_HWA_2DSTAT_ITER_IDX_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_2DSTAT_SMPL_VAL_RAM_U_BASE, CSL_CM4_DSS_HWA_2DSTAT_SMPL_VAL_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_2DSTAT_SMPL_IDX_RAM_U_BASE, CSL_CM4_DSS_HWA_2DSTAT_SMPL_IDX_RAM_U_SIZE},
    {CSL_CM4_DSS_HWA_HIST_RAM_U_BASE, CSL_CM4_DSS_HWA_HIST_RAM_U_SIZE}
};

/* HWA objects - initialized by the driver */
HWA_Object gHwaObject[CONFIG_HWA_NUM_INSTANCES];
/* HWA objects - storage for HWA driver object handles */
HWA_Object *gHwaObjectPtr[CONFIG_HWA_NUM_INSTANCES] = { NULL };
/* HWA objects count */
uint32_t gHwaConfigNum = CONFIG_HWA_NUM_INSTANCES;

/*
 * IPC Notify
 */
#include <drivers/ipc_notify.h>
#include <drivers/ipc_notify/v1/ipc_notify_v1.h>

/* Dedicated mailbox memories address and size */
#define MSS_MBOX_MEM                (CSL_MSS_MBOX_U_BASE)
#define MSS_MBOX_MEM_SIZE           (8U*1024U)

/*
* SW queue between each pair of CPUs
*
* place SW queues at the bottom of the dedicated mailbox memories.
* Driver assume this memory is init to zero in bootloader as it's ECC protected and
* needs to be intialized only once and to ensure that only one core has done the
* mailbox ram initialization before ipc_init. If SBL is not used then Gel does the initialization.
* We need 2 SW Q's for the R5F to send messages to C66SS0 and DSS_M4, i.e 64 B
* we need 2 SW Q's for C66SS0 to send messages to R5F and DSS_CM4, i.e 64 B
* and we need 2 SW Q's for DSS_CM4 to send messages to R5F and C66SS0, i.e 64 B.
*
* Rest of the mailbox memory can be used for ipc_rpmessage or custom message passing.
*/
#define M4SS0_1_TO_C66SS0_SW_QUEUE         (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*6U))
#define C66SS0_TO_M4SS0_1_SW_QUEUE         (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*5U))
#define M4SS0_1_TO_R5FSS0_0_SW_QUEUE       (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*4U))
#define R5FSS0_0_TO_M4SS0_1_SW_QUEUE       (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*3U))
#define C66SS0_TO_R5FSS0_0_SW_QUEUE        (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*2U))
#define R5FSS0_0_TO_C66SS0_SW_QUEUE        (IpcNotify_SwQueue*)((MSS_MBOX_MEM + MSS_MBOX_MEM_SIZE) - (MAILBOX_MAX_SW_QUEUE_SIZE*1U))

/*
 * IPC RP Message
 */
#include <drivers/ipc_rpmsg.h>

/* Number of CPUs that are enabled for IPC RPMessage */
#define IPC_RPMESSAGE_NUM_CORES           (3U)
/* Number of VRINGs for the numner of CPUs that are enabled for IPC */
#define IPC_RPMESSAGE_NUM_VRINGS          (IPC_RPMESSAGE_NUM_CORES*(IPC_RPMESSAGE_NUM_CORES-1))
/* Number of a buffers in a VRING, i.e depth of VRING queue */
#define IPC_RPMESSAGE_NUM_VRING_BUF       (1U)
/* Max size of a buffer in a VRING */
#define IPC_RPMESSAGE_MAX_VRING_BUF_SIZE  (1152U)
/* Size of each VRING is
 *     number of buffers x ( size of each buffer + space for data structures of one buffer (32B) )
 */
#define IPC_RPMESSAGE_VRING_SIZE          RPMESSAGE_VRING_SIZE(IPC_RPMESSAGE_NUM_VRING_BUF, IPC_RPMESSAGE_MAX_VRING_BUF_SIZE)

/* Total Shared memory size used for IPC */
#define IPC_SHARED_MEM_SIZE               (7296U)

/* Shared Memory Used for IPC.
*
* IMPORTANT: Make sure of below,
* - The section defined below should be placed at the exact same location in memory for all the CPUs
* - The memory should be marked as non-cached for all the CPUs
* - The section should be marked as NOLOAD in all the CPUs linker command file
*/
uint8_t gIpcSharedMem[IPC_SHARED_MEM_SIZE] __attribute__((aligned(128), section(".bss.ipc_vring_mem")));

/* This function is called within IpcNotify_init, this function returns core specific IPC config */
void IpcNotify_getConfig(IpcNotify_InterruptConfig **interruptConfig, uint32_t *interruptConfigNum)
{
    /* extern globals that are specific to this core */
    extern IpcNotify_InterruptConfig gIpcNotifyInterruptConfig_m4ss0_1[];
    extern uint32_t gIpcNotifyInterruptConfigNum_m4ss0_1;

    *interruptConfig = &gIpcNotifyInterruptConfig_m4ss0_1[0];
    *interruptConfigNum = gIpcNotifyInterruptConfigNum_m4ss0_1;
}

/* This function is called within IpcNotify_init, this function allocates SW queue */
void IpcNotify_allocSwQueue(IpcNotify_MailboxConfig *mailboxConfig)
{
    IpcNotify_MailboxConfig (*mailboxConfigPtr)[CSL_CORE_ID_MAX] = (void *)mailboxConfig;

    mailboxConfigPtr[CSL_CORE_ID_M4SS0_1][CSL_CORE_ID_R5FSS0_0].swQ = M4SS0_1_TO_R5FSS0_0_SW_QUEUE;
    mailboxConfigPtr[CSL_CORE_ID_M4SS0_1][CSL_CORE_ID_C66SS0].swQ = M4SS0_1_TO_C66SS0_SW_QUEUE;
    mailboxConfigPtr[CSL_CORE_ID_R5FSS0_0][CSL_CORE_ID_M4SS0_1].swQ = R5FSS0_0_TO_M4SS0_1_SW_QUEUE;
    mailboxConfigPtr[CSL_CORE_ID_C66SS0][CSL_CORE_ID_M4SS0_1].swQ = C66SS0_TO_M4SS0_1_SW_QUEUE;
}



void Pinmux_init(void);
void PowerClock_init(void);
void PowerClock_deinit(void);

/*
 * Common Functions
 */

uint64_t SOC_virtToPhy(void *virtAddr)
{
    uintptr_t   temp = (uintptr_t) virtAddr;
    uint64_t    phyAddr = (uint64_t) temp;
#if (__ARM_ARCH == 7) && (__ARM_ARCH_PROFILE == 'M')

    if ((temp >=  CSL_CM4_MSS_SPIA_RAM_U_BASE) &&
        (temp < CSL_CM4_DSS_ESM_U_BASE))
    {
        phyAddr -= ARM_M4_VIRT_TO_PHY_OFFSET;
    }
    else if ((temp >= CSL_CM4_HWA_M4_RAM_U_BASE) &&
            (temp < (CSL_CM4_HWA_M4_RAM_U_BASE + CSL_DSS_CM4_RAM_U_SIZE)))
    {
        phyAddr -= CSL_CM4_HWA_M4_RAM_U_BASE;
        phyAddr += CSL_DSS_CM4_RAM_U_BASE;
    }

#endif
    return (phyAddr);
}

uint8_t SOC_rcmGetEfusePGVer(void)
{
    CSL_top_ctrlRegs* ptrTopCtrlRegs = (CSL_top_ctrlRegs*)CSL_CM4_TOP_CTRL_U_BASE;

    return (uint8_t)((ptrTopCtrlRegs->EFUSE1_ROW_14) & 0xFU);
}

uint32_t SOC_rcmGetHwaIPVer(void)
{
    uint32_t regVal = 0U;

    regVal = CSL_REG32_RD(CSL_CM4_DSS_HWA_CFG_U_BASE);

    return regVal;
}

const char *SOC_getCoreName(uint16_t coreId)
{
    static char *coreIdNames[CSL_CORE_ID_MAX + 1] =
    {
        "r5f0-0",
        "c66ss0",
        "m4ss0-1",
        "unknown"
    };
    const char *name;

    if(coreId < CSL_CORE_ID_MAX)
    {
        name = coreIdNames[coreId];
    }
    else
    {
        name = coreIdNames[CSL_CORE_ID_MAX];
    }
    return name;
}



void System_init(void)
{
    /* DPL init sets up address transalation unit, on some CPUs this is needed
     * to access SCICLIENT services, hence this needs to happen first
     */
    Dpl_init();

    

    PowerClock_init();
    /* Now we can do pinmux */
    Pinmux_init();
    /* finally we initialize all peripheral drivers */
    EDMA_init();
    HWA_init();


    /* IPC Notify */
    {
        IpcNotify_Params notifyParams;
        int32_t status;

        /* initialize parameters to default */
        IpcNotify_Params_init(&notifyParams);

        /* specify the priority of IPC Notify interrupt */
        notifyParams.intrPriority = 5U;

        /* specify the core on which this API is called */
        notifyParams.selfCoreId = CSL_CORE_ID_M4SS0_1;

        /* list the cores that will do IPC Notify with this core
        * Make sure to NOT list 'self' core in the list below
        */
        notifyParams.numCores = 2;
        notifyParams.coreIdList[0] = CSL_CORE_ID_R5FSS0_0;
        notifyParams.coreIdList[1] = CSL_CORE_ID_C66SS0;

        notifyParams.isMailboxIpcEnabled = 0;

        notifyParams.isCrcEnabled = 0;


        notifyParams.isCustomIpcConfigEnabled = 0;

        /* initialize the IPC Notify module */
        status = IpcNotify_init(&notifyParams);
        DebugP_assert(status==SystemP_SUCCESS);

    }
    /* IPC RPMessage */
    {
        RPMessage_Params rpmsgParams;
        int32_t status;

        /* initialize parameters to default */
        RPMessage_Params_init(&rpmsgParams);

        /* TX VRINGs */
        rpmsgParams.vringTxBaseAddr[CSL_CORE_ID_R5FSS0_0] = (uintptr_t)(&gIpcSharedMem[4864]);
        rpmsgParams.vringTxBaseAddr[CSL_CORE_ID_C66SS0] = (uintptr_t)(&gIpcSharedMem[6080]);
        /* RX VRINGs */
        rpmsgParams.vringRxBaseAddr[CSL_CORE_ID_R5FSS0_0] = (uintptr_t)(&gIpcSharedMem[1216]);
        rpmsgParams.vringRxBaseAddr[CSL_CORE_ID_C66SS0] = (uintptr_t)(&gIpcSharedMem[3648]);
        /* Other VRING properties */
        rpmsgParams.vringSize = IPC_RPMESSAGE_VRING_SIZE;
        rpmsgParams.vringNumBuf = IPC_RPMESSAGE_NUM_VRING_BUF;
        rpmsgParams.vringMsgSize = IPC_RPMESSAGE_MAX_VRING_BUF_SIZE;
        rpmsgParams.isCrcEnabled = 0;

        /* initialize the IPC RP Message module */
        status = RPMessage_init(&rpmsgParams);
        DebugP_assert(status==SystemP_SUCCESS);
    }

}

void System_deinit(void)
{
    EDMA_deinit();
    HWA_deinit();
    RPMessage_deInit();
    IpcNotify_deInit();

    PowerClock_deinit();
    Dpl_deinit();
}
