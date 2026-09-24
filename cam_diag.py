"""Camera diagnostic script v2 - grab frame with BayerRG8 and test SDK conversion."""
import sys, ctypes, time, numpy as np
IKapLib_dir = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary\Examples\Python\IKapLib"
if IKapLib_dir not in sys.path:
    sys.path.append(IKapLib_dir)

import IKapC
import IKapCDef

print("=" * 60)
print("  IKapC Camera Diagnostic Tool v2")
print("=" * 60)

res = IKapC.ItkManInitialize()
res, numCameras = IKapC.ItkManGetDeviceCount()
res, devInfo = IKapC.ItkManGetDeviceInfo(0)
fn = devInfo.FullName.decode('utf-8', errors='ignore') if devInfo.FullName else "?"
print(f"\nCAMERA MODEL: {fn}")

res, hDev = IKapC.ItkDevOpen(0, IKapCDef.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE)
if res != IKapCDef.ITKSTATUS_OK:
    print(f"FAILED to open: {res}")
    sys.exit(1)

# Set BayerRG8 (the only color format this camera supports)
IKapC.ItkDevFromString(hDev, b"PixelFormat", b"BayerRG8")
res, pf = IKapC.ItkDevToString(hDev, b"PixelFormat")
print(f"PIXEL FORMAT: {pf}")

res, w_str = IKapC.ItkDevToString(hDev, b"Width")
res, h_str = IKapC.ItkDevToString(hDev, b"Height")
res, ps_str = IKapC.ItkDevToString(hDev, b"PayloadSize")
w = int(w_str.decode() if isinstance(w_str, bytes) else w_str)
h = int(h_str.decode() if isinstance(h_str, bytes) else h_str)
ps = int(ps_str.decode() if isinstance(ps_str, bytes) else ps_str)
print(f"WIDTH: {w}")
print(f"HEIGHT: {h}")
print(f"PAYLOAD SIZE: {ps}")
print(f"EXPECTED MONO BYTES: {w * h}")
print(f"EXPECTED COLOR BYTES: {w * h * 3}")

# Build format map
fmt_map = {}
for attr in dir(IKapCDef):
    if attr.startswith("ITKBUFFER_VAL_FORMAT_"):
        fmt_map[getattr(IKapCDef, attr)] = attr

# Allocate stream
res, hStream = IKapC.ItkDevAllocStreamEx(hDev, 0, 3)
print(f"\nAllocStream: {res}")

# Create BGR conversion buffer at correct dimensions
res, hBufferConvert = IKapC.ItkBufferNew(w, h, IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888)
print(f"ItkBufferNew(BGR888, {w}x{h}): result={res}")

xferMode = ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_TRANSFER_MODE_SYNCHRONOUS_WITH_PROTECT)
startMode = ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_START_MODE_NON_BLOCK)
IKapC.ItkStreamSetPrm(hStream, IKapCDef.ITKSTREAM_PRM_START_MODE, startMode)
IKapC.ItkStreamSetPrm(hStream, IKapCDef.ITKSTREAM_PRM_TRANSFER_MODE, xferMode)

captured = {"raw": None, "raw_info": None, "converted": None}

def on_frame(pParam):
    try:
        res2, hBuf = IKapC.ItkStreamGetCurrentBuffer(hStream)
        if res2 != IKapCDef.ITKSTATUS_OK:
            return
        res2, bInfo = IKapC.ItkBufferGetInfo(hBuf)
        if bInfo.State not in (IKapCDef.ITKBUFFER_VAL_STATE_FULL, IKapCDef.ITKBUFFER_VAL_STATE_UNCOMPLETED):
            return

        # Raw buffer
        res2, npRaw = IKapC.ItkBufferToNumPy(hBuf)
        if res2 == IKapCDef.ITKSTATUS_OK and npRaw is not None:
            captured["raw"] = npRaw.copy()
            captured["raw_info"] = {
                'ImageWidth': bInfo.ImageWidth,
                'ImageHeight': bInfo.ImageHeight,
                'ImageSize': bInfo.ImageSize,
                'TotalSize': bInfo.TotalSize,
                'PixelFormat': bInfo.PixelFormat,
                'ImagePixelDepth': bInfo.ImagePixelDepth,
            }

        # Try SDK conversion to BGR888
        cres = IKapC.ItkBufferConvert(hBuf, hBufferConvert,
                                       IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888,
                                       IKapCDef.ITKBUFFER_VAL_CONVERT_OPTION_AUTO_FORMAT)
        if cres == IKapCDef.ITKSTATUS_OK:
            res3, npConv = IKapC.ItkBufferToNumPy(hBufferConvert)
            if res3 == IKapCDef.ITKSTATUS_OK and npConv is not None:
                captured["converted"] = npConv.copy()
    except Exception as ex:
        print(f"  Frame callback error: {ex}")

cbFunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(on_frame)
IKapC.ItkStreamRegisterCallback(hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME, cbFunc, ctypes.c_void_p(None))

res = IKapC.ItkStreamStart(hStream, 0)
print(f"StreamStart: {res}")

for _ in range(50):
    time.sleep(0.1)
    if captured["raw"] is not None:
        break

IKapC.ItkStreamStop(hStream)
IKapC.ItkStreamUnregisterCallback(hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME)
IKapC.ItkDevFreeStream(hStream)
IKapC.ItkBufferFree(hBufferConvert)
IKapC.ItkDevClose(hDev)

import cv2

print(f"\n{'='*60}")
print(f"  FRAME CAPTURE RESULTS")
print(f"{'='*60}")

if captured["raw"] is not None:
    raw = captured["raw"]
    info = captured["raw_info"]
    print(f"  RAW NUMPY SHAPE:  {raw.shape}")
    print(f"  RAW NUMPY DTYPE:  {raw.dtype}")
    print(f"  RAW NUMPY SIZE:   {raw.size}")
    print(f"  RAW NUMPY NBYTES: {raw.nbytes}")
    print(f"  BUFFER ImageWidth:  {info['ImageWidth']}")
    print(f"  BUFFER ImageHeight: {info['ImageHeight']}")
    print(f"  BUFFER ImageSize:   {info['ImageSize']}")
    print(f"  BUFFER TotalSize:   {info['TotalSize']}")
    print(f"  BUFFER PixelFormat: {info['PixelFormat']} = {fmt_map.get(info['PixelFormat'], 'UNKNOWN')}")
    print(f"  BUFFER PixelDepth:  {info['ImagePixelDepth']}")
    
    # Save raw numpy
    np.save("diag_raw.npy", raw)
    
    # Reshape if 1D
    if len(raw.shape) == 1:
        raw2d = raw.reshape((info['ImageHeight'], info['ImageWidth']))
    else:
        raw2d = raw
    print(f"  RAW RESHAPED: {raw2d.shape}")
    cv2.imwrite("diag_raw_mono.png", raw2d)
    print(f"  Saved raw as mono: diag_raw_mono.png")
    
    # OpenCV BayerRG demosaic
    bgr_cv = cv2.cvtColor(raw2d, cv2.COLOR_BayerRG2BGR)
    cv2.imwrite("diag_opencv_bayer_rg.png", bgr_cv)
    print(f"  Saved OpenCV BayerRG->BGR: diag_opencv_bayer_rg.png")
    print(f"  OpenCV result shape: {bgr_cv.shape}")
else:
    print("  NO RAW FRAME CAPTURED!")

if captured["converted"] is not None:
    conv = captured["converted"]
    print(f"\n  SDK CONVERTED SHAPE: {conv.shape}")
    print(f"  SDK CONVERTED DTYPE: {conv.dtype}")
    print(f"  SDK CONVERTED SIZE:  {conv.size}")
    if len(conv.shape) == 1:
        try:
            conv3d = conv.reshape((h, w, 3))
            cv2.imwrite("diag_sdk_bgr.png", conv3d)
            print(f"  Saved SDK BGR888 converted: diag_sdk_bgr.png")
        except Exception as e:
            print(f"  Failed to reshape SDK converted: {e}")
    elif len(conv.shape) == 3:
        cv2.imwrite("diag_sdk_bgr.png", conv)
        print(f"  Saved SDK BGR888 converted: diag_sdk_bgr.png")
else:
    print("\n  SDK BGR CONVERSION: NOT AVAILABLE or FAILED")

print(f"\n{'='*60}")
print(f"  DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
