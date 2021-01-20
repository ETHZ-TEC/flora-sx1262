#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:49:20 2018

@author: rtrueb

The purpose of the LoraConfig class is to store one specific LoRa PHY
configuration and calculate the time-on-air for that specific LoRa
configuration.

Calculations are based on the data sheet for SX126x LoRa tranceiver chips.

"""

import numpy as np
from enum import IntEnum, unique
import unittest

minPlOverhead = 13 # bytes
adrReqOvh = 4 # bytes
adrAnsOvh = 1 # bytes

class LoraConfig(object):
    def __init__(self, bw=None, sf=None, phyPl=None, cr=1, ih=True, lowDataRate=False, crc=True, nPreambleSyms=None):
        """
        Args:
            sf: spreading factor (5 to 12)
            bw: bandwdith in Hertz (e.g. 125000)
            phyPl: physical layer payload in bytes (1 to 255)
            numPreambleSyms: number of preamble symbols (6 to 65535)
            cr: coding rate value (1 to 4) (cr=1 coding rate of 4/5; cr=4 coding rate of 4/8)
            ih: implicit header (ih=False header enabled; ih=True header disabled)
            crc: CRC enabled (crc=True CRC enabled; crc=False CRC disabled)
            lowDataRate: low data rate optimization (de=True ON; de=False OFF)
        """
        self.bw = bw                        # bandwidth (BW) [in Hz]
        self.sf = sf                        # spreading factor (SF)
        self.phyPl = phyPl                  # physical layer payload in bytes
        self.cr = cr                        # coding rate (CR) [1, 2, 3, 4]
        self.ih = ih                        # implicit header
        self.lowDataRate = lowDataRate      # low data rate enabled
        self.crc = crc                      # CRC enabled
        self.nPreambleSyms = nPreambleSyms  # number of symbols in the preamble

        self._bwList = [  7810,
                         10420,
                         15630,
                         20830,
                         31250,
                         41670,
                         62500,
                        125000,
                        250000,
                        500000]
        self._sfList = (5, 6, 7, 8, 9, 10, 11, 12)
        self._crList = (1, 2, 3, 4)
        self._phyPlRange = (0, 255)
        self._nPreambleSymsRange = (0, 65535) # datasheet says (8, 65535) but less is possible in reality

    @property
    def timeOnAir(self):
        """Calculates the on-air-time of a transmission using LoRa modulation.
        Returns:
            Time-on-air of a single packet in seconds.
        """
        assert self.bw in self._bwList
        assert self.sf in self._sfList
        assert type(self.ih) == bool or type(self.ih) == int
        assert type(self.lowDataRate) == bool
        assert self.cr in self._crList
        assert self._nPreambleSymsRange[0] <= self.nPreambleSyms <= self._nPreambleSymsRange[1]
        assert self._phyPlRange[0] <= self.phyPl <= self._phyPlRange[1]

        sub = 2 if ( (self.sf not in (5, 6)) and self.lowDataRate) else 0
        syncSyms = 6.25 if (self.sf in (5, 6)) else 4.25
        nBitCrc = 16 if self.crc else 0
        nBitHeader = 0 if self.ih else 20
        constVal = 0 if (self.sf in (5, 6)) else 8

        arg1 = 8*self.phyPl + nBitCrc - 4*self.sf + nBitHeader + constVal
        ceilPart = np.ceil(max(arg1, 0)/(4*(self.sf - sub)))
        nSymbol = self.nPreambleSyms + syncSyms + 8 + ceilPart*(self.cr + 4)
        toa = (2**(self.sf) / self.bw) * nSymbol

        return toa

    @property
    def sensitivity(self):
        """Get the radio receive sensitivity for the current radio configuration.
        Returns:
            Radio sensitivity in dBm
        """
        return getSensitivity(mod='lora', datarate=self.sf)


class FskConfig(object):
    def __init__(self, bitrate=None, nPreambleBits=None, nSyncwordBytes=None, nLengthBytes=None, nAddressBytes=None, phyPl=None, nCrcBytes=None):
        """
        Args:
            bitrate: bit rate in bits/sec
            nPreambleBits: number of preamble bits (8 to 65535)
            nSyncwordBytes: number of sync word bytes (0 to 8)
            nLengthBytes: number of length bytes (0 or 1)
            nAddressBytes: number of address bytes (0 or 1)
            phyPl: physical layer payload length in bytes (1 to 255)
            nCrcBytes: number of CRC bytes (0, 1, or 2)
        """
        self.bitrate = bitrate
        self.nPreambleBits = nPreambleBits
        self.nSyncwordBytes = nSyncwordBytes
        self.nLengthBytes = nLengthBytes
        self.nAddressBytes = nAddressBytes
        self.phyPl = phyPl
        self.nCrcBytes = nCrcBytes

        self._bitrateRange = (600, 300000)
        self._nPreambleBitsRange = (8, 65535)
        self._nSyncwordBytesList = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        self._nLengthBytesList = (0, 1)
        self._nAddressBytesList = (0, 1)
        self._phyPlRange = (0, 255)
        self._nCrcBytesList = (0, 1, 2)
        self._bwList = [    4800,
                            5800,
                            7300,
                            9700,
                           11700,
                           14600,
                           19500,
                           23400,
                           29300,
                           39000,
                           46900,
                           58600,
                           78200,
                           93800,
                          117300,
                          156200,
                          187200,
                          234300,
                          312000,
                          373600,
                          467000,]
    @property
    def timeOnAir(self):
        """Calculates the on-air-time of a transmission using FSK modulation.
        Returns:
            Time-on-air of a single packet in seconds.
        """
        assert self._bitrateRange[0] <= self.bitrate <= self._bitrateRange[1]
        assert self._nPreambleBitsRange[0] <= self.nPreambleBits <= self._nPreambleBitsRange[1]
        assert self.nSyncwordBytes in self._nSyncwordBytesList
        assert self.nLengthBytes in self._nLengthBytesList
        assert self.nAddressBytes in self._nAddressBytesList
        assert self._phyPlRange[0] <= self.phyPl <= self._phyPlRange[1]
        assert self.nCrcBytes in self._nCrcBytesList

        if self.nLengthBytes == 0:
            assert self.phyPl >= 1

        nBits = self.nPreambleBits + self.nSyncwordBytes*8 + self.nLengthBytes*8 + self.nAddressBytes*8 + self.phyPl*8 + self.nCrcBytes*8
        toa = nBits / self.bitrate

        return toa

    @property
    def sensitivity(self):
        """Get the radio receive sensitivity for the current radio configuration.
        Returns:
            Radio sensitivity in dBm
        """
        return getSensitivity(mod='fsk', datarate=self.bitrate)

################################################################################

# power mapping based on measurements on the ETZ roof with DevBoard (3V supply), and half-wave antenna, and Rocketlogger, 27.3.2019
# NOTE: value for +22dBm has not been measured
# NOTE: MCU was runnig during measurement (according to code used)
originalPowerMapping = {
    -9: 0.102232706962291,
    -8: 0.10605795151005062,
    -7: 0.1018726071808578,
    -6: 0.1055663120264606,
    -5: 0.11338459295994305,
    -4: 0.1172563780423522,
    -3: 0.12101477289503175,
    -2: 0.12875360034864494,
    -1: 0.13584356812867068,
     0: 0.14310316445602753,
     1: 0.15022126146246428,
     2: 0.16027161570417706,
     3: 0.16651042912060093,
     4: 0.17722397750465702,
     5: 0.1847670147042248,
     6: 0.19379373782516432,
     7: 0.2041878004014488,
     8: 0.21372336098921055,
     9: 0.22514546634269766,
    10: 0.2362092510653682,
    11: 0.2480739948332415,
    12: 0.26178691590284997,
    13: 0.2780706814443991,
    14: 0.2926323230525427,
    15: 0.30346624358694085,
    16: 0.3121536679295964,
    17: 0.3215969884606631,
    18: 0.33407471290069624,
    19: 0.3486906716841178,
    20: 0.36689187784523625,
    21: 0.3905990519543255,
}

# power mapping based on measurements with FlockLab observer 029 (3.3V supply), and 1/4 wave antenna, 13.12.2019, FlockLab test 79473
# NOTE: MCU was in STOP2 when radio Tx consumption was measured
quarterWaveFlPowerMapping = {
    -9: 0.08945616702,
    -8: 0.09460299208799999,
    -7: 0.099578706546,
    -6: 0.10448746192499998,
    -5: 0.11455434091199998,
    -4: 0.11929931531099999,
    -3: 0.12430729307699999,
    -2: 0.133929239664,
    -1: 0.143157369432,
     0: 0.152242811028,
     1: 0.160868436861,
     2: 0.17299270797599997,
     3: 0.1806770334765,
     4: 0.195096958452,
     5: 0.204870363225,
     6: 0.2176852541325,
     7: 0.23280722549700003,
     8: 0.24700358004299997,
     9: 0.26316197721,
    10: 0.27861563937899997,
    11: 0.29544817262399997,
    12: 0.313915058556,
    13: 0.334245148809,
    14: 0.35092590380099997,
    15: 0.36629330654400005,
    16: 0.3820694414625,
    17: 0.39747749334149995,
    18: 0.41424662028750003,
    19: 0.43013109966749996,
    20: 0.4459038418229999,
    21: 0.48508167193199997,
    22: 0.4852003546995,
}

# power mapping based on measurements with FlockLab observer 029 (3.3V supply), and 1/2 wave antenna, 13.12.2019, FlockLab test 79478
# NOTE: MCU was in STOP2 when radio Tx consumption was measured
halfWaveFlPowerMapping = {
    -9: 0.087393237657,
    -8: 0.09261940861799998,
    -7: 0.097593682758,
    -6: 0.102728153022,
    -5: 0.11276440111199998,
    -4: 0.11774408447999998,
    -3: 0.122769026061,
    -2: 0.132438343356,
    -1: 0.14175660544649998,
     0: 0.15081100001550002,
     1: 0.15947407429799998,
     2: 0.17163240109949998,
     3: 0.17933417053200001,
     4: 0.19385680668750002,
     5: 0.20365722557399998,
     6: 0.21657585186600004,
     7: 0.23185936365899998,
     8: 0.246392882274,
     9: 0.262704754059,
    10: 0.27850527850199996,
    11: 0.294775220817,
    12: 0.3135657317595,
    13: 0.3339379755195,
    14: 0.350712863754,
    15: 0.366039201198,
    16: 0.38210797815299996,
    17: 0.397222875918,
    18: 0.414392317152,
    19: 0.430949972376,
    20: 0.44819405446199995,
    21: 0.48682536127500003,
    22: 0.48691466141999995,
}

# power mapping based on measurements with Rocketlogger and DevBoard (3.0V supply), and 1/2 wave antenna, 13.12.2019, binary is based on the powerprofiling test (comboard_testing) used on FlockLab
# NOTE: MCU was in STOP2 when radio Tx consumption was measured
halfWaveRlPowerMapping = {
    -9: 0.0795,
    -8: 0.0843,
    -7: 0.0888,
    -6: 0.0933,
    -5: 0.1026,
    -4: 0.1068,
    -3: 0.1113,
    -2: 0.12,
    -1: 0.1284,
     0: 0.1368,
     1: 0.1446,
     2: 0.1554,
     3: 0.1623,
     4: 0.1755,
     5: 0.1845,
     6: 0.1962,
     7: 0.21,
     8: 0.2229,
     9: 0.2376,
    10: 0.2517,
    11: 0.2661,
    12: 0.2814,
    13: 0.297,
    14: 0.3102,
    15: 0.3228,
    16: 0.3363,
    17: 0.3492,
    18: 0.3642,
    19: 0.3783,
    20: 0.3936,
    21: 0.4239,
    22: 0.4257,
}

powerMapping = halfWaveRlPowerMapping


def getConfigTxPowerLevels():
    return powerMapping.keys()


def getTxPower(configPwr):
    '''Returns the power consumption (in Watts) when SX1262 is transmitting with 1/2 wave antenna for a given configured power level.
    Args:
      configPwr: configured power level (in dBm)
    '''

    assert configPwr >= min(powerMapping.keys()) and configPwr <= max(powerMapping.keys())
    return powerMapping[configPwr]

def getRxPower():
    '''Returns the power (in Watts) consumption in the receive mode (DC-DC)
    '''
    return 0.005*3.3 # in Watt


def getSensitivity(mod, datarate):
    '''Returns the receive sensitivity levels.
    LoRa: Linear interpolation/extrapolation based on datasheet values for 125kHz: -124 dBm for SF7, -137 dBm for SF12
    FSK: Log interpolation/extrapolation based on datasheets values
    TODO: improved values based on measurements.
    '''
    if mod == 'fsk':
        bitrate = datarate
        return 3.614*np.log(bitrate) - 148.285
    elif mod == 'lora':
        sf = datarate
        return sf*(-2.6) - 105.8
    else:
        return None

@unique
class Modems(IntEnum):
    MODEM_FSK  = 0
    MODEM_LORA = 1

@unique
class LoraCodingRates(IntEnum):
    LORA_CR_4_5  = 0x01,
    LORA_CR_4_6  = 0x02,
    LORA_CR_4_7  = 0x03,
    LORA_CR_4_8  = 0x04,


flora_radio_constants = [
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 12,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 11,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 10,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 9,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 8,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 7,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 10
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 6,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 12
        },
        {
            "modem": Modems.MODEM_LORA,
            "bandwidth": 0,
            "datarate": 5,
            "coderate": LoraCodingRates.LORA_CR_4_5,
            "preambleLen": 12
        },
        {
            "modem": Modems.MODEM_FSK,
            "bandwidth": 234300,
            "datarate": 125000,
            "fdev": 50000,
            "preambleLen": 2
        },
        {
            "modem": Modems.MODEM_FSK,
            "bandwidth": 234300,
            "datarate": 200000,
            "fdev": 10000,
            "preambleLen": 2
        },
        {
            "modem": Modems.MODEM_FSK,
            "bandwidth": 312000,
            "datarate": 250000,
            "fdev": 23500,
            "preambleLen": 4
        }
]

def flora_toa(modIdx, phyPlLen):
    """Calculates the time-on-air of a transmission.
    Args:
        modIdx: index of modulation radio_modulations struct array as defined in radio_constants.c
        phyPlLen: physical layer payload (in bytes)
    Returns:
        Time-on-air of a single packet in seconds.
    """
    mods = flora_radio_constants
    mod = mods[modIdx]

    def mapBw(bw):
        if bw == 0: return 125000
        elif bw == 1: return 250000
        elif bw == 2: return 500000
        else: raise Exception('ERROR: undefined bandwidth!')

    if mod['modem'] == Modems.MODEM_LORA:
        loraconfig = LoraConfig()
        loraconfig.bw = mapBw(mod['bandwidth'])
        loraconfig.sf = mod['datarate']
        loraconfig.phyPl = phyPlLen
        loraconfig.cr = mod['coderate']
        loraconfig.ih = False
        loraconfig.lowDataRate = True if mod['datarate'] in [11, 12] else False
        loraconfig.crc = True
        loraconfig.nPreambleSyms = mod['preambleLen']
        return loraconfig.timeOnAir
    elif mod['modem'] == Modems.MODEM_FSK:
        fskconfig = FskConfig()
        fskconfig.bitrate = mod['datarate']
        fskconfig.nPreambleBits = 8*mod['preambleLen']
        fskconfig.nSyncwordBytes = 3
        fskconfig.nLengthBytes = 1
        fskconfig.nAddressBytes = 0
        fskconfig.phyPl = phyPlLen
        fskconfig.nCrcBytes = 2
        return fskconfig.timeOnAir
    else:
        raise Exception('ERROR: invalid modulation!')


class TestTimeOnAirMethods(unittest.TestCase):

    def test_constructor1(self):
        loraconfig = LoraConfig(
            sf=7,
            bw=125000,
            phyPl=12,
            nPreambleSyms=8,
            cr=LoraCodingRates.LORA_CR_4_5,
            ih=False,
            crc=True,
            lowDataRate=False
        )
        self.assertAlmostEqual(
            loraconfig.timeOnAir,
            0.041215999999999996,
            places=5,
        )

    def test_constructor2(self):
        loraconfig = LoraConfig(
            sf=5,
            bw=125000,
            phyPl=12,
            nPreambleSyms=8,
            cr=LoraCodingRates.LORA_CR_4_5,
            ih=False,
            crc=True,
            lowDataRate=False
        )
        self.assertAlmostEqual(
            loraconfig.timeOnAir,
            0.013375999999999999,
            places=5,
        )

    def test_setParams(self):
        loraconfig = LoraConfig()
        loraconfig.bw = 125000
        loraconfig.sf = 7
        loraconfig.phyPl = 12
        loraconfig.cr = LoraCodingRates.LORA_CR_4_5
        loraconfig.ih = False
        loraconfig.lowDataRate = False
        loraconfig.crc = True
        loraconfig.nPreambleSyms = 8
        self.assertAlmostEqual(
            loraconfig.timeOnAir,
            0.041215999999999996,
            places=5,
        )


if __name__ == '__main__':
    # fskconfig = FskConfig()
    # fskconfig.bitrate = 250000
    # fskconfig.nPreambleBits = 16
    # fskconfig.nSyncwordBytes = 2
    # fskconfig.nLengthBytes = 1
    # fskconfig.nAddressBytes = 1
    # fskconfig.phyPl = 6
    # fskconfig.nCrcBytes = 1
    # print(fskconfig.timeOnAir)

    loraconfig = LoraConfig()
    loraconfig.bw = 125000
    loraconfig.sf = 12
    loraconfig.phyPl = 2
    loraconfig.cr = LoraCodingRates.LORA_CR_4_5
    loraconfig.ih = False
    loraconfig.lowDataRate = False
    loraconfig.crc = 0
    loraconfig.nPreambleSyms = 12
    print('Time-on-air (custom): {:.6f}s'.format(loraconfig.timeOnAir))


    # modIdx = 7   # modulation (as defined in radio_constants.c)
    # phyPlLen = 5  # in bytes
    # print('Time-on-air (mod={}, phyPlLen={}): {:.6f}s'.format(modIdx, phyPlLen, flora_toa(modIdx, phyPlLen)))

    # unittest.main()
