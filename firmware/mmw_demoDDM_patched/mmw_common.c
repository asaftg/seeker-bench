/**
 *   @file  mmw_common.c
 *
 *   @brief
 *      Defines DPM functions for inter-core communication.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2024-25 Texas Instruments, Inc.
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

#include <drivers/ipc_rpmsg.h>
#include <ti/common/mmwave_error.h>

#define DPM_MSS_BOOT_INFO_BIT_POS     (0U)
#define DPM_DSS_CM4_BOOT_INFO_BIT_POS (1U)
#define DPM_DSS_BOOT_INFO_BIT_POS     (2U)

#define DPM_EINVAL                  (MMWAVE_ERRNO_DPM_BASE-1)

#define DPM_BUFFER_MAX_SIZE            (512U)

#define MMWDEMO_DPC_COMMONCFG     0x11111111U
#define MMWDEMO_DPC_CFG           0x22222222U
#define MMWDEMO_DPC_ELEVEST_CFG   0x33333333U
#define MMWDEMO_DPC_START         0x44444444U
#define MMWDEMO_DPC_EXECUTE       0x55555555U
#define MMWDEMO_DPC_RESULT        0x66666666U

/**
 * @brief
 *  DPM Handle
 */
typedef void*   DPM_Handle;

typedef void (*DPM_ReportFxn)(void *data, uint16_t dataLen);

typedef struct DPM_InitCfg_t
{
    uint16_t localEndPt;

    DPM_ReportFxn reportFxn;

    uint8_t setBitPos;
}DPM_InitCfg;

/**
 * @brief
 *  DPM Master control block
 *
 * @details
 *  The structure is used to hold all the relevant information required to
 *  execute the DPM module
 */
typedef struct DPM_MCB_t
{
    /* dpm init config */
    DPM_InitCfg dpmInitCfg;
    /**
     * @brief   Remote IPC Object: This is used by the IPC module to communicate
     * with the remote peer to exchange DPM IPC messages.
     */
    RPMessage_Object ipcMailboxObj;

}DPM_MCB;

DPM_MCB gDPMObj;
uint32_t gCbHitCnt = 0;

uint16_t gRemoteCoreEndPt[CSL_CORE_ID_MAX] =
    {   (uint16_t)RPMESSAGE_MAX_LOCAL_ENDPT - 1U,
        (uint16_t)RPMESSAGE_MAX_LOCAL_ENDPT - 2U,
        (uint16_t)RPMESSAGE_MAX_LOCAL_ENDPT - 3U,
        (uint16_t)RPMESSAGE_MAX_LOCAL_ENDPT - 4U
    };

CSL_mss_ctrlRegs* CSL_MSS_CTRL_getBaseAddress (void)
{

#ifdef SUBSYS_M4
    return (CSL_mss_ctrlRegs*)CSL_CM4_MSS_CTRL_U_BASE;
#else
    return (CSL_mss_ctrlRegs*)CSL_MSS_CTRL_U_BASE;
#endif

}


static void mboxRemoteMailboxCallbackFxn (RPMessage_Object *obj,
            void *arg, void *data, uint16_t dataLen, int32_t crcStatus,
            uint16_t remoteCoreId, uint16_t remoteEndPt)
{
   gCbHitCnt++;
    /* This is assumed that only a single core will generate the interrupt at a time. */
    /* Scenarios when callbackFxn is hit, before data is consumed are not taken care.
     * and data is subject to get overwritten with the new data. */
    gDPMObj.dpmInitCfg.reportFxn((void*)data, dataLen);
}


int32_t Mmw_setLinkState(uint8_t bitPos, uint8_t val)
{
    int32_t retVal = SystemP_SUCCESS;

    CSL_mss_ctrlRegs *mssCtrl = CSL_MSS_CTRL_getBaseAddress();

    if (mssCtrl != NULL)
    {
        CSL_FINSR(mssCtrl->MSS_BOOT_INFO_REG0,
                  (uint32_t)bitPos,
                  (uint32_t)bitPos,
                  (uint32_t)val);
    }
    else
    {
        retVal = -1;
    }
    return retVal;
}

DPM_Handle DPM_init(DPM_InitCfg* ptrInitCfg, int32_t* errCode)
{
    DPM_MCB*            ptrDPM = NULL;
    RPMessage_CreateParams createParams;

    /* Sanity Check */
    if((ptrInitCfg == NULL) || (errCode == NULL))
    {
        return NULL;
    }

    RPMessage_CreateParams_init(&createParams);
    createParams.localEndPt = ptrInitCfg->localEndPt;
    /* Setup IPC callback configuration: */
    createParams.recvCallback = mboxRemoteMailboxCallbackFxn;

    /* Initialize the DPM Master control block: */
    ptrDPM = (DPM_MCB*)&gDPMObj;
    (void)memset ((void*)ptrDPM, 0, sizeof(DPM_MCB));

    (void)memcpy ((void*)&ptrDPM->dpmInitCfg, (void*)ptrInitCfg, sizeof(DPM_InitCfg));

    *errCode = RPMessage_construct(&ptrDPM->ipcMailboxObj, &createParams);

    *errCode = Mmw_setLinkState(ptrInitCfg->setBitPos, (uint8_t)1U);

    return (DPM_Handle)ptrDPM;
}

/* mmw_synch all cores */
int32_t DPM_synch(DPM_Handle handle, int32_t* errCode)
{
    *errCode = SystemP_SUCCESS;
    CSL_mss_ctrlRegs *mssCtrl = CSL_MSS_CTRL_getBaseAddress();
    int32_t retVal = SystemP_FAILURE;
    uint32_t status;

    /* check if other cores are enable */
    if (mssCtrl != NULL)
    {
        /* Get the operational status for the CM4 */
        status = CSL_FEXTR (mssCtrl->MSS_BOOT_INFO_REG0, 2U, 0U);
        retVal = (int32_t) status;
    }
    return retVal;
}

int32_t DPM_send(DPM_Handle handle,  void* arg, uint32_t argLen, uint16_t remoteCoreID)
{
    int32_t retVal;
    DPM_MCB*    ptrDPM;

    ptrDPM = (DPM_MCB*)handle;
    if (ptrDPM == NULL)
    {
        /* Error: Invalid argument */
        retVal = DPM_EINVAL;
        return retVal;
    }

    retVal = RPMessage_send((void *)arg, (uint16_t)argLen, remoteCoreID,
                            gRemoteCoreEndPt[remoteCoreID], ptrDPM->dpmInitCfg.localEndPt,
                            SystemP_WAIT_FOREVER);

    return retVal;
}
