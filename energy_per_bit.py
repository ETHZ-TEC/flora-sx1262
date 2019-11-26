#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:14:46 2019

@author: rtrueb
"""

import random
import numpy as np
import pandas as pd

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CategoricalColorMapper
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform

from sx1262 import *


################################################################################
# Config
################################################################################
modulationList = ['fsk_1', 'fsk_5', 'fsk_10', 'fsk_100', 'fsk_300', 'lora_sf5', 'lora_sf6', 'lora_sf7', 'lora_sf8', 'lora_sf9', 'lora_sf10', 'lora_sf11', 'lora_sf12'] # valid options 'lora_sfX' or 'fsk_Y' (X=spreading factor number, Y=bitrate in kbit)
configPwrList = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
payloadSize = 20 # in Bytes
pointSelection = 'border' # 'border' (max distance) or 'reachable' (reachable configuration)
pathlossModel = 'friis' # 'friis' (Friis free-space pathloss model) or 'hata' (Hata sub-urban path loss model)

################################################################################
# Pathloss model
################################################################################

def hata_pathlossToDistance(pathloss):
    '''
    Args
        pathloss: pathloss in dB (positive means loss, negative means gain)
    '''
    # path loss based on Hata model (sub-urban)
    L_SU = pathloss # path loss sub-urban
    f = 868e6
    h_B = 18 # Height of base station antenna. Unit: meter (m)
    h_M = 18 # Height of mobile station antenna. Unit: meter (m)
    C_h = 0.8 + (1.1*np.log10(f) - 0.7)*h_M - 1.56*np.log10(f) # Antenna height correction factor for small/medium city
    L_U = 2*(np.log10(f/28))**2 + 5.4 + L_SU
    distance = 10**( (L_U - 69.55 - 26.16*np.log10(f) + 13.82*np.log10(h_B) + C_h)/(44.9 - 6.55*np.log10(h_B)) )

    return distance

def hata_distanceToPathloss(distance):
    '''
    Args
        distance: distance in meters
    '''
    f = 868e6
    # path loss based on Hata model (sub-urban)
    h_B = 18 # Height of base station antenna. Unit: meter (m)
    h_M = 18 # Height of mobile station antenna. Unit: meter (m)
    C_h = 0.8 + (1.1*np.log10(f) - 0.7)*h_M - 1.56*np.log10(f) # Antenna height correction factor for small/medium city
    L_U = 69.55 + 26.16*np.log10(f) - 13.82*np.log10(h_B) - C_h + (44.9 - 6.55*np.log10(h_B))*np.log10(distance) # path loss urban
    L_SU = L_U - 2*(np.log10(f/28))**2 - 5.4 # path loss sub-urban

    return L_SU

def friis_pathlossToDistance(pathloss):
    # pathloss in dB based on Friis Transmission Formula
    f = 868e6
    lambdaSymb = 3e8 / f
    distance = (lambdaSymb*10**(pathloss/20))/(4*np.pi)
    return distance

def friis_distanceToPathloss(distance):
    # pathloss in dB based on Friis Transmission Formula
    f = 868e6
    lambdaSymb = 3e8 / f
    pathloss = 20*np.log10( (4*np.pi*distance)/lambdaSymb ) # path loss in dB based on Friis Transmission Formula
    return pathloss

def test_pathloss():
    # Debug prints for testing pathloss model functions
    distance1 = 500
    print('distance1: {}'.format(distance1))
    pathloss = hata_distanceToPathloss(distance1)
    print('hata sub-urban pathloss: {}'.format(pathloss))
    distance2 = hata_pathlossToDistance(pathloss)
    print('distance2: {}'.format(distance2))
    distance1 = 500
    print('distance1: {}'.format(distance1))
    pathloss = friis_distanceToPathloss(distance1)
    print('friis pathloss: {}'.format(pathloss))
    distance2 = friis_pathlossToDistance(pathloss)
    print('distance2: {}'.format(distance2))

################################################################################
# EnergyPerBit points
################################################################################

def getEnergyPerBit(mod):

    return energy

def generateBorderPoints(pathlossModel):
    loraconfig = LoraConfig()
    loraconfig.bw = 125000
    loraconfig.sf = 5
    loraconfig.phyPl = payloadSize
    loraconfig.cr = 1
    loraconfig.ih = False
    loraconfig.lowDataRate = False
    loraconfig.crc = True
    loraconfig.nPreambleSyms = 12

    fskconfig = FskConfig()
    fskconfig.bitrate = 100000
    fskconfig.nPreambleBits = 16
    fskconfig.nSyncwordBytes = 2
    fskconfig.nLengthBytes = 1
    fskconfig.nAddressBytes = 1
    fskconfig.phyPl = payloadSize
    fskconfig.nCrcBytes = 1

    retList = []

    for mod in modulationList:
        for configPwr in configPwrList:
            toa = None
            sensitivity = None
            datarate = None
            if 'lora' in mod:
                datarate = int(mod.replace('lora_sf', ''))
                loraconfig.sf = datarate
                toa = loraconfig.timeOnAir
                sensitivity = getSensitivity(mod='lora', datarate=datarate)
            elif 'fsk' in mod:
                datarate = int(1000*float(mod.replace('fsk_', '')))
                fskconfig.bitrate = datarate
                toa = fskconfig.timeOnAir
                sensitivity = getSensitivity(mod='fsk', datarate=datarate)
            else:
                raise Exception('Unknown modulation: {}'.format(mod))
            pwr = getTxPower(configPwr=configPwr)
            energy = toa * pwr
            energyPerBit = energy/(payloadSize*8)
            actualBitrate = (payloadSize*8)/toa

            if pathlossModel == 'friis':
                distance = friis_pathlossToDistance(configPwr - sensitivity)
            elif pathlossModel == 'hata':
                distance = hata_pathlossToDistance(configPwr - sensitivity)
            else:
                raise Exception('You need to choose a valid pathloss model (pathlossModel)!')

            ret = {
                'modulation': mod,
                'configPwr': configPwr,
                'energyPerBit': energyPerBit,
                'distance': distance,
                'actualBitrate': actualBitrate,
                'toa': toa,
            }
            retList.append(ret)

    return pd.DataFrame.from_dict(retList)

def generateReachablePoints(pathlossModel, numPoints=1000):
    loraconfig = LoraConfig()
    loraconfig.bw = 125000
    loraconfig.sf = 5
    loraconfig.phyPl = payloadSize
    loraconfig.cr = 1
    loraconfig.ih = False
    loraconfig.lowDataRate = False
    loraconfig.crc = True
    loraconfig.nPreambleSyms = 12

    fskconfig = FskConfig()
    fskconfig.bitrate = 100000
    fskconfig.nPreambleBits = 16
    fskconfig.nSyncwordBytes = 2
    fskconfig.nLengthBytes = 1
    fskconfig.nAddressBytes = 1
    fskconfig.phyPl = payloadSize
    fskconfig.nCrcBytes = 1

    retList = []

    for i in range(numPoints):
        mod = random.choice(modulationList)
        configPwr = random.choice(configPwrList)
        # distance = random.uniform(0, 1e6) # many config points are invalid due to frequently picking large distance which is not feasible with short-range modulations
        pathloss = random.uniform(0, 200)
        distance = hata_pathlossToDistance(pathloss) if pathlossModel == 'hata' else friis_pathlossToDistance(pathloss)

        if 'lora' in mod:
            datarate = int(mod.replace('lora_sf', ''))
            loraconfig.sf = datarate
            toa = loraconfig.timeOnAir
            sensitivity = getSensitivity(mod='lora', datarate=datarate)
        elif 'fsk' in mod:
            datarate = int(1000*float(mod.replace('fsk_', '')))
            fskconfig.bitrate = datarate
            toa = fskconfig.timeOnAir
            sensitivity = getSensitivity(mod='fsk', datarate=datarate)
        else:
            raise Exception('Unknown modulation: {}'.format(mod))
        pwr = getTxPower(configPwr=configPwr)
        energy = toa * pwr
        energyPerBit = energy/(payloadSize*8)
        actualBitrate = (payloadSize*8)/toa

        if pathlossModel == 'friis':
            pathloss = friis_distanceToPathloss(distance)
        elif pathlossModel == 'hata':
            pathloss = hata_distanceToPathloss(distance)
        else:
            raise Exception('You need to choose a valid pathloss model (pathlossModel)!')

        # ignore infeasible configurations
        if configPwr - pathloss < sensitivity:
            continue

        ret = {
            'modulation': mod,
            'configPwr': configPwr,
            'energyPerBit': energyPerBit,
            'distance': distance,
            'actualBitrate': actualBitrate,
            'toa': toa,
        }
        retList.append(ret)

    return pd.DataFrame.from_dict(retList)

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    if pointSelection == 'border':
        df = generateBorderPoints(pathlossModel=pathlossModel)
    elif pointSelection == 'reachable':
        df = generateReachablePoints(pathlossModel=pathlossModel, numPoints=10000)
    else:
        raise Exception('You need to choose a valid point selection (pointSelection)!')


    source = ColumnDataSource(data=dict(
        distance=df.distance,
        energyPerBit=df.energyPerBit,
        modulation=df.modulation,
        configPwr=df.configPwr,
        actualBitrate=df.actualBitrate,
        toa=df.toa,
    ))
    hover = HoverTool(tooltips=[
        ("Modulation", "@modulation"),
        ("configPwr", "@configPwr dBm"),
        ('energyPerBit', '@energyPerBit J'),
        ('distance', '@distance{int} m'),
        ('actualBitrate', '@actualBitrate{int} bps'),
        ('Time-on-air', '@toa{0.0000} s'),
    ])
    # mapper = LinearColorMapper(palette=plasma(256), low=5, high=12)
    mapper = CategoricalColorMapper(factors=modulationList, palette=plasma(len(modulationList)))

    p = figure(
        sizing_mode='stretch_both',
        tools=[hover, 'pan', 'wheel_zoom', 'reset', 'save', 'box_zoom'],
        title="Energy Per Bit (SX1262) - Config: pointSelection={}, pathlossModel={}".format(pointSelection, pathlossModel))
    p.circle(
        'energyPerBit',
        'distance',
        size=10,
        source=source,
        fill_color=transform('modulation', mapper),
        line_color=None,
        legend_group='modulation',
    )
    p.xaxis.axis_label = 'Energy Per Bit ([J])'
    p.yaxis.axis_label = 'Distance [m]'
    p.yaxis.formatter.use_scientific = False
    p.legend.location = 'top_right'

    output_file('energyPerBit_{}_{}.html'.format(pointSelection, pathlossModel))
    show(p)
