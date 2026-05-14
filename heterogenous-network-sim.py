#!/usr/bin/env python3
import argparse
import copy
import logging

logger = logging.getLogger(__name__)

from lib.common import find_random_position
from lib.config import Config
from lib.node import NodeConfig, MESHTASTIC_ROLE
from lib.point import Point

'''We want to simulate heterogenous networks to investigate recommendations for
managing them.

The simulator previously only handled homogenous networks where nodes had
identical height, gain, txpower etc. and mainly varied in location and possibly
role (CLIENT/ROUTER).

For our purposes a heterogenous network has 2 kinds of nodes:
- 'infrastructure' nodes: fixed location, higher gain antenna, higher height-above-ground, does not generate messages
- 'personal' nodes: possibly mobile, lower gain antenna, lower to the ground, generates messages

Imagine 'infrastructure' nodes as the solar rooftop nodes many people are
setting up, while the 'personal' nodes are carried on someone's person and
designed to be mobile.

We want to test this kind of network in general vs a homogenous one, and also
compare some different network configurations, such as making all 'personal'
nodes CLIENT_MUTE. We're especially interested in how often packets take
suboptimal paths due to personal nodes winning retransmit races against
infrastructure nodes, but having less reach, leading to degraded network
performance.
'''

def generate_networks(conf: Config):
    '''generate 3 roughly-comparable Meshtastic networks to simulate.

    First generate a connected network of infrastructure nodes, then generate
    personal nodes s.t. any personal node can reach at least one infrastructure node.
    This set of nodes can create 2 networks to simulate: one with all nodes set
    to CLIENT, and one where all personal nodes are CLIENT_MUTE. This simulates an
    area with coverage by infrastructure nodes.

    To create a comparable homogenous network from this, convert the infrastructure nodes
    into personal nodes (reduced height, antenna gain) and check that our network is
    still connected.

    If not... throw it out and start over? go back to the drawing board
    for how we generate these networks?

    Arguments:
    conf -- Config object describing network

    Returns:
    '''

    RATIO_OF_INFRA_NODES = 0.2 # percentage of nodes in the network that should be infra nodes

    # configuration differences just for infrastructure nodes
    infra_only_conf = copy.copy(conf)
    infra_only_conf.MINDIST = 100 # spread infra nodes out more
    infra_only_conf.GL = 5.0 # higher gain antenna
    infra_only_conf.HM = 10.0 # installed higher up
    infra_only_conf.PERIOD = 1000 * infra_only_conf.SIMTIME # infra nodes only rebroadcast messages, are not active clients

    num_infra_nodes = int(conf.NR_NODES * RATIO_OF_INFRA_NODES)
    num_personal_nodes = conf.NR_NODES - num_infra_nodes
    if num_infra_nodes < 2:
        raise ValueError(f"Must have enough nodes to have 2 infrastructure nodes. {conf.NR_NODES=} * {RATIO_OF_INFRA_NODES} < 2")

    infra_configs = []
    for i in range(num_infra_nodes):
        x, y = find_random_position(infra_only_conf, infra_configs)
        z = infra_only_conf.HM
        pos = Point(x, y, z)

        # keep role as default CLIENT
        nodeconf = NodeConfig(i, pos, infra_only_conf.PERIOD, antenna_gain=infra_only_conf.GL)

        infra_configs.append(nodeconf)


    personal_configs = []
    for i in range(num_personal_nodes):
        x, y = find_random_position(conf, infra_configs)
        z = conf.HM
        pos = Point(x, y, z)

        nodeconf = NodeConfig(i, pos, conf.PERIOD, antenna_gain=conf.GL)

        personal_configs.append(nodeconf)

    # first network: heterogenous, all CLIENT
    first_net = []
    first_net.extend(infra_configs)
    first_net.extend(personal_configs)

    # second network: heterogenous, personal are CLIENT_MUTE
    second_net = []
    second_net.extend(copy.copy(infra_configs))
    second_personal_configs = copy.copy(personal_configs)
    for cfg in second_personal_configs:
        cfg.role = MESHTASTIC_ROLE.CLIENT_MUTE
    second_net.extend(second_personal_configs)

    # third network: 1st network, but all infra nodes converted to personal
    third_net = []
    third_infra_configs = copy.copy(infra_configs)
    for cfg in third_infra_configs:
        # height, gain, period
        cfg.position.z = conf.HM
        cfg.antenna_gain = conf.GL
        cfg.period = conf.PERIOD
    third_net.extend(third_infra_configs)
    third_net.extend(personal_configs)
    # TODO: check network is still connected

    return (first_net, second_net, third_net)

def main():
    parser = argparse.ArgumentParser(
        description='simulate and compare different configurations of heterogenous networks'
        )
    parser.add_argument('nr_nodes', type=int, help='Number of nodes in generated situations. Start small to get a feel for runtimes.')
    args = parser.parse_args()

    conf = Config()
    conf.NR_NODES = args.nr_nodes

    # set up networks to simulate
    (het, het_mute, hom) = generate_networks(conf)

    # run simulations

    # collect & compare/display results

if __name__ == '__main__':
    main()
