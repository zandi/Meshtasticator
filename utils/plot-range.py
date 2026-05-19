#!/usr/bin/env python3
import argparse
import os.path
import sys

# gross hack to allow importing when directly calling either from
# utils dir as cwd, or project root as cwd.
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./../'))

from lib.config import Config
from lib.node import NodeConfig
from lib.phy import rootFinder
from lib.point import Point

import matplotlib.pyplot as plt

# for argument default values
default_conf = Config()

def main():
    parser = argparse.ArgumentParser(
        description='plot the rssi vs sensitivity of a node to visualize range'
        )
    parser.add_argument('-p','--power', type=int, default=default_conf.PTX, help='tx power in dBm')
    parser.add_argument('-f','--freq', type=float, default=default_conf.FREQ, help='frequency in Hz')
    parser.add_argument('-z','--z-height', type=float, default=default_conf.HM, help='height of transmitter above ground in meters')
    parser.add_argument('--rx-z','--rx-z-height', type=float, default=default_conf.HM, help='height of receivers above ground in meters')
    parser.add_argument('-g','--gain', type=int, default=default_conf.GL, help="gain of transmitting node's antenna in dBi")
    parser.add_argument('--rx-gain', type=int, default=default_conf.GL, help="gain of receiving node's antenna dBi")
    parser.add_argument('-m','--model', type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=default_conf.MODEL, help='selection of model for RF propagation.')

    args = parser.parse_args()

    conf = Config()
    conf.MODEL = args.model

    # make tx node (static)
    tx_pos = Point(0, 0, args.z_height)
    tx_node = NodeConfig(0, tx_pos, 1, args.power, args.freq, antenna_gain=args.gain)

    # find max range of tx node to specified kind of rx node
    def zero_link_budget(dist):
        '''single-variable function defined s.t. 0 is when rssi == sensitivity,
        using our given tx and rx nodes, and the chosen RF model. For root-finding.
        '''
        pos = Point(dist, 0, args.rx_z)
        rx_n = NodeConfig(1, pos, 1, args.power, args.freq, antenna_gain=args.rx_gain)
        rssi = tx_node.compute_rssi_and_pathloss_to(rx_n, conf)[0]
        return rssi - conf.current_preset['sensitivity']

    max_range = int(rootFinder(zero_link_budget, 1500))
    print(f"maximum range: {max_range} m")

    # make list of rx nodes to simulate tx to, every 100m for a bit past our max range
    rx_nodes = []
    distances = range(100,max_range+(3*100),100)
    for x in distances:
        pos = Point(x, 0, args.rx_z)
        n = NodeConfig(1, pos, 1, args.power, args.freq, antenna_gain=args.rx_gain)
        rx_nodes.append(n)

    # collect rssi at each rx node from tx node
    rssis = [tx_node.compute_rssi_and_pathloss_to(rx_n, conf)[0] for rx_n in rx_nodes]

    for i in range(len(rssis)):
        if rssis[i] < conf.current_preset['sensitivity']:
            print(f"index {i} lost reception with rssi {rssis[i]}, {(i+1) * 100}m away, sensitivity {conf.current_preset['sensitivity']}")
            break

    # plot graph
    fig, ax = plt.subplots()
    ax.plot(distances, rssis)
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('rssi (dBi)')
    title = f"RSSI of tx node (power {tx_node.tx_power} dBm, gain {tx_node.antenna_gain} dBi, {tx_node.position.z}m AGL) --> rx node(s) (gain {args.rx_gain} dBi, {args.rx_z}m AGL) using model {conf.MODEL} at {conf.FREQ / 1000000} MHz"
    ax.set_title(title, wrap=True)

    # horizontal line showing lower limit of reception
    sensitivity_line_y = [conf.current_preset['sensitivity'] for _ in distances]
    ax.plot(distances, sensitivity_line_y)
    ax.annotate(f"sensitivity ({sensitivity_line_y[0]} dBi)", (distances[0], sensitivity_line_y[0]+2))

    plt.show()

if __name__ == '__main__':
    main()
