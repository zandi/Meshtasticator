#!/usr/bin/env python3
import argparse
import copy
import logging
import random

logging.basicConfig()
logger = logging.getLogger(__name__)

from lib.common import find_random_position
from lib.config import Config
from lib.node import NodeConfig, default_generate_node_list, MESHTASTIC_ROLE
from lib.point import Point

from loraMesh import run_simulation

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

def upgrade_node_in_place(n: NodeConfig, infra_config: Config) -> None:
    '''apply the given infra config to the node
    '''
    n.position.z = infra_config.HM
    n.antenna_gain = infra_config.GL
    n.period = infra_config.PERIOD

def make_heterogenous_networks(nodes: [NodeConfig], infra_config: Config):
    '''given a list of node configs, create two comparable heterogenous
    networks. One simple has some nodes upgraded to infrastructure nodes.
    The other also has non-infrastructure nodes set to CLIENT_MUTE. Once
    infrastructure nodes are chosen, non-infra nodes in all networks have
    'can_move' set to True to allow movement.

    Arguments:
    nodes -- list of NodeConfig objects.
    infra_config -- Config object for infrastructure nodes

    Returns:
    (het_nodes, het_nodes_mute) -- duplicate of `nodes` but with some nodes randomly upgraded to infrastructure nodes.
    '''
    # randomly pick some nodes to upgrade to infra nodes. They must be
    # - immoblie
    # - mutually reachable
    # - not within our infra node MINDIST

    RATIO_OF_INFRA_NODES = 0.2 # percentage of nodes in the network that should be infra nodes
    num_infra_nodes = int(len(nodes) * RATIO_OF_INFRA_NODES)
    if num_infra_nodes < 2:
        raise ValueError(f"Must have enough nodes to have 2 infrastructure nodes. {len(nodes)} * {RATIO_OF_INFRA_NODES} < 2")

    het_nodes = copy.deepcopy(nodes)

    # randomly select nodes to upgrade
    infra_node_indices = []
    indices = [i for i in range(len(het_nodes))]
    random.shuffle(indices)
    for try_i in indices:
        if len(infra_node_indices) >= num_infra_nodes:
            # have enough, we're done!
            break

        candidate = het_nodes[try_i]
        if not candidate.can_move:
            if len(infra_node_indices) < 1:
                # first choice is easiest
                upgrade_node_in_place(candidate, infra_config)
                infra_node_indices.append(try_i)
            else:
                # check against other infra nodes already chosen.
                # make sure we're not too close to any other infra node,
                # and can reach at least one
                is_connected = False
                too_close = False
                for other_i in infra_node_indices:
                    other_n = het_nodes[other_i]
                    dist = candidate.position.euclidean_distance(other_n.position)
                    if dist < infra_config.MINDIST:
                        # immediately disqualified
                        too_close = True
                        break

                    rssi = candidate.compute_rssi_and_pathloss_to(other_n, infra_config)[0]
                    if not is_connected and rssi > infra_config.current_preset['sensitivity']:
                        # just need to be connected to one
                        is_connected = True
                if is_connected and not too_close:
                    # candidate wins!
                    upgrade_node_in_place(candidate, infra_config)
                    infra_node_indices.append(try_i)

    # (should have) found and upgraded infrastructure nodes
    if len(infra_node_indices) < num_infra_nodes:
        raise ValueError(f"Unable to select {num_infra_nodes} suitable nodes from node list to upgade to infrastructure nodes")

    # for the second heterogenous network, make all non-infra nodes CLIENT_MUTE
    het_nodes_mute = copy.deepcopy(het_nodes)
    for i in range(len(het_nodes_mute)):
        if i not in infra_node_indices:
            het_nodes_mute[i].role = MESHTASTIC_ROLE.CLIENT_MUTE

    # hack: re-enable movement possibility for all non-infrastructure nodes
    for i in range(len(nodes)):
        if i not in infra_node_indices:
            nodes[i].can_move = True # modifies argument by reference
            het_nodes[i].can_move = True
            het_nodes_mute[i].can_move = True

    return (het_nodes, het_nodes_mute)

def generate_networks(conf: Config):
    '''generate 3 roughly-comparable Meshtastic networks to simulate.

    One is heterogenous with all nodes set to CLIENT. Another is heterogenous
    with non-infrastructure nodes set to CLIENT_MUTE. The third is homogenous
    and the 'baseline' network.

    Arguments:
    conf -- Config object describing network

    Returns:
    '''

    # new algorithm: generate baseline homogenous network first.
    # from this, randomly pick 2 nodes to be infrastructure nodes. They must
    # be immobile and mutually reachable. Upgrade them to infrastructure.
    # For the last network, copy this first heterogenous network, but make all
    # non-infrastructure nodes CLIENT_MUTE

    # configuration differences just for infrastructure nodes
    infra_only_conf = copy.copy(conf)
    infra_only_conf.MINDIST = 100 # spread infra nodes out more
    infra_only_conf.GL = 5.0 # higher gain antenna
    infra_only_conf.HM = 10.0 # installed higher up
    infra_only_conf.PERIOD = 1000 * infra_only_conf.SIMTIME # infra nodes only rebroadcast messages, are not active clients

    adjustment = 0
    ADJUSTMENT_LIMIT = 10
    while adjustment < ADJUSTMENT_LIMIT:
        random.seed(conf.SEED + adjustment)
        hom_network = default_generate_node_list(conf)
        # hack: set all nodes to immobile, then retroactively choose non-infra nodes
        # to make mobile
        for n in hom_network:
            n.can_move = False

        try:
            het_network, het_network_mute = make_heterogenous_networks(hom_network, infra_only_conf)
            return (het_network, het_network_mute, hom_network)
        except ValueError as e:
            # generated homogenous network cannot be upgraded. generate a new one.
            logger.debug(f"Unable to convert homogenous network to suitable heterogenous one. Generating a new homogenous network. Limit {ADJUSTMENT_LIMIT} times")
            adjustment += 1
            continue
    raise ValueError(f"Unable to generate a suitable network using {conf.NR_NODES} nodes")


def run_simulations(conf: Config):
    '''generate networks, run simulations on them, and return results

    Arguments:
    conf -- Config object describing network to generate

    Returns:
    (het_result, het_mute_result, hom_result) -- simulation results for each variety of network
    '''

    # set up networks to simulate
    (het, het_mute, hom) = generate_networks(conf)

    # examine networks
    logger.debug(f"heterogenous network: {conf.NR_NODES} nodes")
    for n in het:
        logger.debug(f"\t\t{n}")

    logger.debug(f"heterogenous network w/ CLIENT_MUTE: {conf.NR_NODES} nodes")
    for n in het_mute:
        logger.debug(f"\t\t{n}")

    logger.debug(f"baseline homogenous network: {conf.NR_NODES} nodes")
    for n in hom:
        logger.debug(f"\t\t{n}")

    # run simulations. Can use same config since global params are indentical,
    # we only tweaked node-specific values for infra nodes.
    # Will print to stdout, but oh well.
    random.seed(conf.SEED)
    print(f"\nheterogenous network: {conf.NR_NODES} nodes")
    het_result = run_simulation(conf, het)

    random.seed(conf.SEED)
    print(f"\nheterogenous network with non-infra nodes CLIENT_MUTE: {conf.NR_NODES} nodes")
    het_mute_result = run_simulation(conf, het_mute)

    random.seed(conf.SEED)
    print(f"\nbaseline homogenous network: {conf.NR_NODES} nodes")
    hom_result = run_simulation(conf, hom)

    return (het_result, het_mute_result, hom_result)

def main():
    # set up common config, use default config for default arg values where appropriate
    conf = Config()

    parser = argparse.ArgumentParser(
        description='simulate and compare different configurations of heterogenous networks'
        )
    parser.add_argument('nr_nodes', nargs='+', type=int, help='Number of nodes in generated situations. Start small to get a feel for runtimes. If provided a list, comparable simulations will be run for each choice of nr_nodes.')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose/debug output')
    parser.add_argument('-g', '--gui', action='store_true', help='enable gui. helpful for debugging & reviewing simulation details')
    parser.add_argument('-b', '--batch', type=int, default=1, help='run each nr_node sim on b different networks of nr_nodes')
    parser.add_argument('-s', '--seed', type=int, default=conf.SEED, help='seed for simulation config RNG')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        lib_logger = logging.getLogger('lib')
        lib_logger.setLevel(logging.DEBUG)
        logger.debug("debug logging enabled")

    logger.debug(f"using RNG seed {args.seed}")
    conf.SEED = args.seed
    random.seed(conf.SEED) # deterministic sims
    conf.MODEL = 0 # selection of model with less dramatic range between infra nodes

    if args.gui:
        conf.GUI_ENABLED = True
        conf.PLOT = True # also plot sim message sequence
    else:
        conf.GUI_ENABLED = False
        conf.PLOT = False

    results = {}
    for nr_nodes in args.nr_nodes:
        conf.NR_NODES = nr_nodes

        batch_het_res = []
        batch_het_mute_res = []
        batch_hom_res = []
        for i in range(args.batch):
            print(f"\n\nbatch run: {i+1}/{args.batch}")
            conf.SEED = conf.SEED + i # change network & behavior for batch runs
            (het_res, het_mute_res, hom_res) = run_simulations(conf)
            batch_het_res.append(het_res)
            batch_het_mute_res.append(het_mute_res)
            batch_hom_res.append(hom_res)

        results[nr_nodes] = {}
        results[nr_nodes]['het'] = batch_het_res
        results[nr_nodes]['het_mute'] = batch_het_mute_res
        results[nr_nodes]['hom'] = batch_hom_res

    # TODO: compare & display results
    pass

if __name__ == '__main__':
    main()
