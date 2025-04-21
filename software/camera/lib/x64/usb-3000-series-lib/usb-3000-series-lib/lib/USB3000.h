#ifndef USB3000_H
#define USB3000_H

int _stdcall FindUSB3DAQ();
int _stdcall USB3OpenDevice(int DevIndex);
int _stdcall USB3CloseDevice(int DevIndex);
int _stdcall USB3GetDeviceSN(int DevIndex, char *SN);
int _stdcall USB3ReadDevcieSN(int DevIndex);
int _stdcall USB3GetDeviceModel(int DevIndex, char *Model);
int _stdcall USB3ReadDevcieModel(int DevIndex);

//--------------------------------------------------------------------------
// Ananlog Input Configuration

int _stdcall SetUSB3AiSampleRate(int DevIndex, unsigned int SamplePeriod);
int _stdcall SetUSB3AiSampleMode(int DevIndex, unsigned char AiSampleMode);
int _stdcall SetUSB3AiConnectType(int DevIndex, unsigned char AiConnectType);
int _stdcall SetUSB3AiRange(int DevIndex, unsigned char Chan, float AiRange);
int _stdcall SetUSB3AiChanSel(int DevIndex, unsigned char Chan, unsigned char Sel);
int _stdcall SetUSB3AiTrigSource(int DevIndex, unsigned char AiTrigSource);
int _stdcall SetUSB3AiConvSource(int DevIndex, unsigned char AiConvSource);
int _stdcall SetUSB3AiPreTrigPoints(int DevIndex, unsigned int AiPreTrigPoints);
int _stdcall SetUSB3AiOneShotPoints(int DevIndex, unsigned int AiOneShotPoints);
int _stdcall SetUSB3ClrAiFifo(int DevIndex);

//--------------------------------------------------------------------------
// Digital I/O Configuration

int _stdcall SetUSB3DiSampleRate(int DevIndex, unsigned int SamplePeriod);
int _stdcall SetUSB3DiSampleMode(int DevIndex, unsigned char DiSampleMode);
int _stdcall SetUSB3DiTrigSource(int DevIndex, unsigned char DiTrigSource);
int _stdcall SetUSB3DiConvSource(int DevIndex, unsigned char DiConvSource);
int _stdcall SetUSB3DiPreTrigPoints(int DevIndex, unsigned int DiPreTrigPoints);
int _stdcall SetUSB3DiOneShotPoints(int DevIndex, unsigned int DiOneShotPoints);
int _stdcall SetUSB3ClrDiFifo(int DevIndex);

int _stdcall SetUSB3DoSampleRate(int DevIndex, unsigned int SamplePeriod);
int _stdcall SetUSB3DoSampleMode(int DevIndex, unsigned char DoSampleMode);
int _stdcall SetUSB3DoTrigSource(int DevIndex, unsigned char DoTrigSource);
int _stdcall SetUSB3DoConvSource(int DevIndex, unsigned char DoConvSource);
int _stdcall SetUSB3DoCycle(int DevIndex, unsigned int DoCycle);
int _stdcall SetUSB3DoDataFifo(int DevIndex, unsigned int Value[], unsigned int Len);
int _stdcall SetUSB3ClrDoFifo(int DevIndex);
int _stdcall SetUSB3DoWaveCtrl(int DevIndex, unsigned int Chan);
int _stdcall SetUSB3DoImmediately(int DevIndex, unsigned int Chan, unsigned int Value);

//--------------------------------------------------------------------------
// Ananlog Output Configuration

int _stdcall SetUSB3AoSampleRate(int DevIndex, unsigned char Chan, unsigned int SamplePeriod);
int _stdcall SetUSB3AoSampleMode(int DevIndex, unsigned char Chan, unsigned char AoSampleMode);
int _stdcall SetUSB3AoTrigSource(int DevIndex, unsigned char Chan, unsigned char AoTrigSource);
int _stdcall SetUSB3AoConvSource(int DevIndex, unsigned char Chan, unsigned char AoConvSource);
int _stdcall SetUSB3AoCycle(int DevIndex, unsigned char Chan, unsigned int AoCycle);
int _stdcall SetUSB3AoDataFifo(int DevIndex, unsigned char Chan, float *Voltage, unsigned int Len);
int _stdcall SetUSB3AoDataFifo4Chans(int DevIndex, float *Volt0, float *Volt1, float *Volt2, float *Volt3, unsigned int Len);
int _stdcall SetUSB3ClrAoFifo(int DevIndex, unsigned char Chan);
int _stdcall SetUSB3AoSync(int DevIndex, unsigned char Chans);
int _stdcall SetUSB3AoImmediately(int DevIndex, unsigned char Chan, float Voltage);
int _stdcall SetUSB3AoImmediately4Chans(int DevIndex, float Volt0, float Volt1, float Volt2, float Volt3);
int _stdcall SetUSB3AoWaveKB(int DevIndex, unsigned char Chan, float k, float b);

//--------------------------------------------------------------------------
// Counter Configuration

int _stdcall SetCtSampleRate(int DevIndex, unsigned char Chan, unsigned int SamplePeriod);
int _stdcall SetCtSampleMode(int DevIndex, unsigned char Chan, unsigned char CtSampleMode);
int _stdcall SetCtRunMode(int DevIndex, unsigned char Chan, unsigned char CtRunMode);
int _stdcall SetCtTrigSource(int DevIndex, unsigned char Chan, unsigned char CtTrigSource);
int _stdcall SetCtConvSource(int DevIndex, unsigned char Chan, unsigned char CtConvSource);
int _stdcall SetCtCountEdge(int DevIndex, unsigned char Chan, unsigned char CtCountEdge);
int _stdcall SetCtValue(int DevIndex, unsigned char Chan, unsigned int Value);
int _stdcall SetCtOverflowValueEdgeCount(int DevIndex, unsigned char Chan, unsigned int OverflowValue);
int _stdcall SetCtFrontPartValueEdgeCount(int DevIndex, unsigned char Chan, unsigned int FrontPartValue);
int _stdcall SetCtReloadValueEdgeCount(int DevIndex, unsigned char Chan, unsigned int ReloadValue);
int _stdcall SetCtDirEdgeCount(int DevIndex, unsigned char Chan, unsigned char Dir);
int _stdcall SetCtPulseWidthStartEdge(int DevIndex, unsigned char Chan, unsigned char StartEdge);
int _stdcall SetCtInternalSrcPeriod(int DevIndex, unsigned char Chan, unsigned int Period);
int _stdcall SetCtSrcSel(int DevIndex, unsigned char Chan, unsigned char SrcSel);
int _stdcall SetCtEncoderMode(int DevIndex, unsigned char Chan, unsigned char EncoderMode);
int _stdcall SetCtZPhase(int DevIndex, unsigned char Chan, unsigned char ALev, unsigned char BLev);
int _stdcall SetCtZValue(int DevIndex, unsigned char Chan, unsigned int Value);
int _stdcall SetCtZEnable(int DevIndex, unsigned char Chan, unsigned char Enable);

int _stdcall SetCtSoftTrig(int DevIndex, unsigned char Chan);
int _stdcall SetCtClrFifo(int DevIndex, unsigned char Chan);

//--------------------------------------------------------------------------
// Trig Control

int _stdcall SetUSB3AiSoftTrig(int DevIndex);
int _stdcall SetUSB3DiSoftTrig(int DevIndex);
int _stdcall SetUSB3DoSoftTrig(int DevIndex);
int _stdcall SetUSB3AoSoftTrig(int DevIndex, unsigned char Chan);
int _stdcall SetUSB3GlobalSoftTrig(int DevIndex);

int _stdcall SetUSB3ClrTrigger(int DevIndex);
int _stdcall SetUSB3ClrAiTrigger(int DevIndex);
int _stdcall SetUSB3ClrDiTrigger(int DevIndex);
int _stdcall SetUSB3ClrDoTrigger(int DevIndex);
int _stdcall SetUSB3ClrAoTrigger(int DevIndex, unsigned char Chan);
int _stdcall SetUSB3ClrGlobalSoftTrig(int DevIndex);

//--------------------------------------------------------------------------
// Sync Configuration

int _stdcall SetUSB3ExtTrigOutSource(int DevIndex,unsigned char Source);
int _stdcall SetUSB3ExtConvOutSource(int DevIndex,unsigned char Source);

//--------------------------------------------------------------------------
// Get Data Acquired

int _stdcall USB3GetAi(int DevIndex, unsigned long Points, float *Ai, long TimeOut);
int _stdcall USB3GetDi(int DevIndex, unsigned long Points, unsigned char *Di, long TimeOut);
int _stdcall USB3GetCt(int DevIndex, unsigned char Chan, unsigned long Points, unsigned int *Ct, long TimeOut);
int _stdcall GetLatestCtValue(int DevIndex, unsigned char Chan, unsigned int *Value, long TimeOut);

//--------------------------------------------------------------------------
// 校准时使用的函数

int _stdcall GoToCalibrate(int DevIndex);
int _stdcall WriteFlash(int DevIndex, short Addr, unsigned char *data);

//--------------------------------------------------------------------------
// ERROR CODE

const int NO_USBDAQ = -1;
const int DevIndex_Overflow = -2;
const int Bad_Firmware = -3;
const int USBDAQ_Closed = -4;
const int Transfer_Data_Fail = -5;
const int NO_Enough_Memory = -6;
const int Time_Out = -7;
const int Not_Reading = -8;
const int ChanIndex_Overflow = -9;
const int Undefined_AiRange = -10;
const int Undefined_SamplePeriod = -11;
const int Undefined_AiConnectType = -12;
const int Undefined_AiSampleMode = -13;
const int Undefined_WaveLen = -14;
const int Undefined_Paramter = -15;

#endif