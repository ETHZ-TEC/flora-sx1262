#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 - 2021, ETH Zurich, Computer Engineering Group (TEC)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


@author: rtrueb
"""

import random
import sys
import numpy as np
import pandas as pd

from bokeh.io import output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CategoricalColorMapper
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

sys.path.append("..") # Adds higher directory to python modules path.
from sx1262 import *


################################################################################
# Utils
################################################################################
def mapModulationToText(modulation):
  mod, dr = modulation.split('_')
  if mod == 'fsk':
    return 'FSK {:0.0f} kbps'.format(float(dr))
  elif mod == 'lora':
    return 'LoRa SF{}'.format(int(dr.replace('sf', '')))
  else:
    raise Exception('Unknown modulation \'{}\'!'.format(mod))


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

def generateBorderPoints(pathlossModel, modulationList, configPwrList, payloadSize):
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

            linkBudget = configPwr - sensitivity
            if pathlossModel == 'friis':
                distance = friis_pathlossToDistance(linkBudget)
            elif pathlossModel == 'hata':
                distance = hata_pathlossToDistance(linkBudget)
            else:
                raise Exception('You need to choose a valid pathloss model (pathlossModel)!')

            ret = {
                'modulation': mod,
                'configPwr': configPwr,
                'energyPerBit': energyPerBit,
                'distance': distance,
                'actualBitrate': actualBitrate,
                'toa': toa,
                'linkBudget': linkBudget,
            }
            retList.append(ret)

    return pd.DataFrame.from_dict(retList)

def generateReachablePoints(pathlossModel, modulationList, configPwrList, payloadSize, numPoints=1000):
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

def plotPoints(pointSelection, pathlossModel, modulationList, configPwrList, payloadSize, numPoints=10000, write_output_file=False):
    if pointSelection == 'border':
        df = generateBorderPoints(
            pathlossModel=pathlossModel,
            modulationList=modulationList,
            configPwrList=configPwrList,
            payloadSize=payloadSize,
        )
    elif pointSelection == 'reachable':
        df = generateReachablePoints(pathlossModel=pathlossModel,
            modulationList=modulationList,
            configPwrList=configPwrList,
            payloadSize=payloadSize,
            numPoints=numPoints,
        )
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
    # add link budget (only available if pointSelection == 'border')
    if pointSelection == 'border':
        source.add(data=df.linkBudget, name='linkBudget')
        hover.tooltips.append( ('linkBudget', '@linkBudget dB') )
    # mapper = LinearColorMapper(palette=plasma(256), low=5, high=12)
    mapper = CategoricalColorMapper(factors=modulationList, palette=plasma(len(modulationList)))

    p = figure(
        sizing_mode='stretch_both',
        tools=[hover, 'pan', 'wheel_zoom', 'reset', 'save', 'box_zoom'],
        title="Energy Per Bit (SX1262) - Config: pointSelection={}, pathlossModel={}, payloadSize={}".format(pointSelection, pathlossModel, payloadSize))
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
    save(p)
    # show(p)

    # matplotlib plots #####################################################
    # these plots only work for border points!
    if pointSelection != 'border':
        return

    plt.close('all')

    # add modulation scheme to df
    df['modScheme'] = [s.split('_')[0] for s in df.modulation]

    # pareto front
    dominated = [np.any(np.logical_and(row['linkBudget'] < df.linkBudget.to_numpy(), row['energyPerBit'] > df.energyPerBit.to_numpy())) for idx, row in df.iterrows()]
    df['undominated'] = np.logical_not(dominated)
    dfPareto = df[df.undominated]
    dfParetoSorted = dfPareto.sort_values(by=['linkBudget'])

    # sort df according to actualBitrate in order to get legend from left to right
    df.sort_values(by=['modScheme', 'actualBitrate'], ascending=[True, False], inplace=True)

    ## matplotlib: energy per bit vs. link budget/range
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    # colors = iter(cm.gist_rainbow(np.linspace(0, 1, len(df.modulation.unique()))))
    # NOTE: markers that consists of a line only are not visible when facecolors='none'
    markers = iter([".", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "H", "D", "d"])
    # markers = iter([".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"])
    for modulation, gdf in df.groupby('modulation'):
      c = 'blue' if 'fsk' in modulation else 'orange'
      # ax1.scatter(gdf.energyPerBit*1e6, gdf.distance*1e-3, color=next(colors), label=modulation)
      ax1.scatter(gdf.energyPerBit*1e6, gdf.linkBudget, color=c, marker=next(markers), label=modulation, facecolors='none')
    paretoLabel = 'Pareto front'
    ax1.step(dfParetoSorted.energyPerBit*1e6, dfParetoSorted.linkBudget, c='k', label=paretoLabel, where='post')
    ax1.set_xscale('log')
    # ax1.set_yscale('log') # for distance plotting
    ax1.grid(color='k', alpha=0.1, linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Energy per bit [$\mathrm{\mu J}$]')
    # ax1.set_ylabel('Distance [m]')
    ax1.set_ylabel('Link budget [dB]')

    # sort legend items
    handles, labels = plt.gca().get_legend_handles_labels()
    paretoIdx = labels.index(paretoLabel)
    order = [labels.index(mod) for mod in df.modulation.unique()]
    ax1.legend(
      [handles[paretoIdx]] + [handles[idx] for idx in order],
      [labels[paretoIdx]] + [mapModulationToText(labels[idx]) for idx in order],
      loc='center left',
      bbox_to_anchor=(1.1, 0.5),
    )

    # add distance as separate axis
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(friis_pathlossToDistance(mn)*1e-3, friis_pathlossToDistance(mx)*1e-3)
    ax2.set_yscale('log')
    ax2.set_ylabel('Range (Friis free-space) [km]')

    # txPower annotation
    sf12Df = df[df.modulation == df.modulation.to_list()[-1]]
    sf12Map = dict(zip(sf12Df.configPwr, zip(sf12Df.energyPerBit*1e6, sf12Df.linkBudget)))
    offset = 5
    for configPwr in [-9, 0, 10, 20, 22]:
      configPwrStr = str(configPwr) if configPwr <= 0 else '+{}'.format(configPwr)
      ax1.annotate('{} dBm'.format(configPwrStr), sf12Map[configPwr], textcoords='offset points', xytext=(offset, -offset))
    ax1.annotate('Tx power', sf12Map[-9], textcoords='offset points', xytext=(offset, -5*offset))
    # print(sf12Map)

    # stretch plot to fit annotation and legend
    # ax1.margins(x=0.2) # scales both sides
    ax1.set_xlim(0.3, 17e3)

    plt.savefig('energyPerBit.pdf', bbox_inches="tight")


    ## matplotlib: time-on-air vs. actual bit rate
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    markers = iter([".", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "H", "D", "d"])
    # markers = iter([".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"])
    for modulation, gdf in df.groupby('modulation'):
      c = 'blue' if 'fsk' in modulation else 'orange'
      # all toa and bit rate values should be equal for the same modulation
      if np.sum(np.diff(gdf.toa)) != 0:
        raise Exception('timeOnAir values are not equal for modulation {}: {}'.format(modulation, gdf.toa.to_numpy()))
      if np.sum(np.diff(gdf.actualBitrate)) != 0:
        raise Exception('actualBitrate values are not equal for modulation {}: {}'.format(modulation, gdf.actualBitrate.to_numpy()))
      ax1.scatter(gdf.toa.to_numpy()[0]*1e3, gdf.actualBitrate.to_numpy()[0], color=c, marker=next(markers), label=modulation, facecolors='none')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(color='k', alpha=0.1, linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Time-on-air [ms]')
    ax1.set_ylabel('Actual bit rate [bit/s]')

    # sort legend items
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index(mod) for mod in df.modulation.unique()]
    ax1.legend(
      [handles[idx] for idx in order],
      [mapModulationToText(labels[idx]) for idx in order],
      loc='center left',
      bbox_to_anchor=(1.0, 0.5),
    )

    plt.savefig('time-on-air.pdf', bbox_inches="tight")

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    # Config
    modulationList = ['fsk_001', 'fsk_005', 'fsk_010', 'fsk_050', 'fsk_100', 'fsk_300', 'lora_sf05', 'lora_sf06', 'lora_sf07', 'lora_sf08', 'lora_sf09', 'lora_sf10', 'lora_sf11', 'lora_sf12'] # valid options 'lora_sfX' or 'fsk_Y' (X=spreading factor number, Y=bitrate in kbit)
    configPwrList = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    payloadSize = 20 # in Bytes
    pointSelection = 'border' # 'border' (max distance) or 'reachable' (reachable configuration)
    pathlossModel = 'friis' # 'friis' (Friis free-space pathloss model) or 'hata' (Hata sub-urban path loss model)
    numPoints = 10000

    plotPoints(
        pointSelection=pointSelection,
        pathlossModel=pathlossModel,
        modulationList=modulationList,
        configPwrList=configPwrList,
        payloadSize=payloadSize,
        numPoints=numPoints,
        write_output_file=True,
    )
