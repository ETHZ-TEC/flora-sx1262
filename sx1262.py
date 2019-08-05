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
        self._phyPlRange = (1, 255)
        self._nPreambleSymsRange = (0, 65535) # datasheet says (8, 65535) but less is possible in reality

    @property
    def timeOnAir(self):
        """Calculates the on-air-time of a LoRa modulated transmission.
        Returns:
            Time-on-air of a single LoRa Packet in seconds.
        """
        assert self.bw in self._bwList
        assert self.sf in self._sfList
        assert type(self.ih) == bool
        assert type(self.lowDataRate) == bool
        assert self.cr in self._crList
        assert self._nPreambleSymsRange[0] <= self.nPreambleSyms <= self._nPreambleSymsRange[1]
        assert self._phyPlRange[0] <= self.phyPl <= self._phyPlRange[1]

        sub = 2 if (self.sf in (5, 6) and self.lowDataRate) else 0
        syncSyms = 6.25 if (self.sf in (5,6)) else 4.25
        nBitCrc = 16 if self.crc else 0
        nSymbolHeader = 0 if self.ih else 20

        arg1 = 8*self.phyPl + nBitCrc - 4*self.sf + nSymbolHeader
        ceilPart = np.ceil(max(arg1, 0)/(4*(self.sf - sub)))
        nSymbol = self.nPreambleSyms + syncSyms + 8 + ceilPart*(self.cr + 4)
        toa = (2**(self.sf) / self.bw) * nSymbol

        return toa


class FskConfig(object):
    def __init__(self, bitrate=None, nPreambleBits=None, nSyncwordBytes=None, nLengthBytes=None, nAddressBytes=None, phyPl=None, nCrcBytes=None):
        """
        Args:
            bitrate: bit rate in bits/sec
            nPreambleBits: number of preamble bits (8 to 65535)
            nSyncwordBytes: number of sync words bytes (0 to 8)
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
        self._phyPlRange = (1, 255)
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
        """Calculates the on-air-time of a FSK modulated transmission.
        Returns:
            Time-on-air of a single LoRa Packet in seconds.
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


class TestTimeOnAirMethods(unittest.TestCase):

    def test_constructor1(self):
        loraconfig = LoraConfig(
            sf=7,
            bw=125000,
            phyPl=12,
            nPreambleSyms=8,
            cr=1,
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
            cr=1,
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
        loraconfig.cr = 1
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
    fskconfig = FskConfig()
    fskconfig.bitrate = 100000
    fskconfig.nPreambleBits = 16
    fskconfig.nSyncwordBytes = 2
    fskconfig.nLengthBytes = 1
    fskconfig.nAddressBytes = 1
    fskconfig.phyPl = 255
    fskconfig.nCrcBytes = 1
    print(fskconfig.timeOnAir)

    unittest.main()
