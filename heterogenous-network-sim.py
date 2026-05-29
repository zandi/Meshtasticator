#!/usr/bin/env python3
import argparse
import copy
import logging
from multiprocessing.pool import Pool
from os import process_cpu_count
import random
from time import sleep

logging.basicConfig()
logger = logging.getLogger(__name__)

from lib.common import find_random_position
from lib.config import Config
from lib.discrete_event_sim import DiscreteEventSim
from lib.gui import Graph
from lib.node import NodeConfig, default_generate_node_list, MESHTASTIC_ROLE, MESHTASTIC_NODE_KIND
from lib.point import Point

import matplotlib.pyplot as plt

# make global so parallelized worker process can pass back only necessary
# (and hopefully not generator) data
METRICS_OF_INTEREST = ['collisionRate', 'nodeReach', 'usefulness', 'txAirUtilizationRate', 'meanDelay', 'avgNodeLinks', 'init_coverage_area', 'init_avg_density', 'messageSeq']
AS_PERCENT = ['collisionRate', 'nodeReach', 'usefulness', 'txAirUtilizationRate']
METRICS_UNITS = {
    'collisionRate': '%',
    'nodeReach': '%',
    'usefulness': '%',
    'txAirUtilizationRate': '%',
    'avgNodeLinks': 'nodes',
    'init_coverage_area': 'km^2',
    'init_avg_density': 'nodes / km^2',
    'messageSeq': 'messages',
    'meanDelay': 'ms'
}

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

class SimContext:
    '''hold context which uniquely determines a simulation
    '''

    def __init__(self, name: str, nr_nodes: int, batch_iter: int, batch_size: int, conf: Config, nodes: [NodeConfig]):
        self.name = name
        self.nr_nodes = nr_nodes
        self.batch_iter = batch_iter
        self.batch_size = batch_size # might not actually need this
        self.conf = conf
        self.nodes = nodes

def upgrade_node_in_place(n: NodeConfig, infra_config: Config) -> None:
    '''apply the given infra config to the node, change the node's kind
    as INFRASTRUCTURE.
    '''
    n.position.z = infra_config.HM
    n.antenna_gain = infra_config.GL
    n.period = infra_config.PERIOD
    n.kind = MESHTASTIC_NODE_KIND.INFRASTRUCTURE

def make_het_network(nodes: [NodeConfig], infra_config: Config) -> [NodeConfig]:
    '''Given a baseline homogenous network, select some nodes
    to upgrade to 'infrastructure' nodes. Apply the relevant parts
    of infra_config to them, change nothing else.

    Arguments:
    nodes -- list of NodeConfigs representing a homogenous network. Modified.
    infra_config -- NodeConfig to apply to selected infra nodes in-place

    Returns:
    list of NodeConfigs representing the new heterogenous network generated
    from the given homogenous network
    '''
    # randomly pick some nodes to upgrade to infra nodes. They must be
    # - immobile
    # - mutually reachable
    # - not within our infra node MINDIST

    RATIO_OF_INFRA_NODES = 0.2 # percentage of nodes in the network that should be infra nodes
    num_infra_nodes = int(len(nodes) * RATIO_OF_INFRA_NODES)
    if num_infra_nodes < 2:
        raise ValueError(f"Must have enough nodes to have 2 infrastructure nodes. {len(nodes)} * {RATIO_OF_INFRA_NODES} < 2")

    het_nodes = copy.deepcopy(nodes)

    # randomly select nodes to upgrade in place
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

    # hack: correct movement ability in baseline network, match this in our
    # heterogenous network. Do this here since we know which nodes were
    # upgraded, and which ones they correspond to in the homogenous network.
    for i in range(len(nodes)):
        if i not in infra_node_indices:
            nodes[i].can_move = True # modifies argument by reference
            het_nodes[i].can_move = True

    return het_nodes

def make_het_mute_network(nodes: [NodeConfig]):
    '''Given a standard heterogenous network, make all
    non-infra nodes CLIENT_MUTE

    Arguments:
    nodes -- list of NodeConfigs representing a heterogenous network,
    from `make_het_network`

    Returns:
    list of NodeConfigs representing the new heterogenous network with
    CLIENT_MUTE generated from the given template heterogenous network
    '''
    het_nodes_mute = copy.deepcopy(nodes)
    found_infra_nodes = False
    for n in het_nodes_mute:
        if n.kind == MESHTASTIC_NODE_KIND.PERSONAL:
            n.role = MESHTASTIC_ROLE.CLIENT_MUTE
        elif n.kind == MESHTASTIC_NODE_KIND.INFRASTRUCTURE:
            found_infra_nodes = True

    if not found_infra_nodes:
        raise ValueError("make_het_mute_network() must be given a network with some nodes with kind==MESHTASTIC_NODE_KIND.INFRASTRUCTURE")

    return het_nodes_mute

def make_het_router_network(nodes: [NodeConfig]):
    '''Given a standard heterogenous network, make all
    infra nodes ROUTER

    Arguments:
    nodes -- list of NodeConfigs representing a heterogenous network,
    from `make_het_network`

    Returns:
    list of NodeConfigs representing the new heterogenous network with
    ROUTER generated from the given template heterogenous network
    '''
    het_nodes_router = copy.deepcopy(nodes)
    found_infra_nodes = False
    for n in het_nodes_router:
        if n.kind == MESHTASTIC_NODE_KIND.INFRASTRUCTURE:
            n.role = MESHTASTIC_ROLE.ROUTER
            found_infra_nodes = True

    if not found_infra_nodes:
        raise ValueError("make_het_router_network() must be given a network with some nodes with kind==MESHTASTIC_NODE_KIND.INFRASTRUCTURE")

    return het_nodes_router

def make_het_router_and_mute_network(nodes: [NodeConfig]):
    '''Given a standard heterogenous network, make all
    infra nodes ROUTER and non-infra nodes CLIENT_MUTE

    Arguments:
    nodes -- list of NodeConfigs representing a heterogenous network,
    from `make_het_network`

    Returns:
    list of NodeConfigs representing the new heterogenous network with ROUTER
    and CLIENT_MUTE generated from the given template heterogenous network
    '''
    het_nodes_router_and_mute = copy.deepcopy(nodes)
    found_infra_nodes = False
    for n in het_nodes_router_and_mute:
        if n.kind == MESHTASTIC_NODE_KIND.PERSONAL:
            n.role = MESHTASTIC_ROLE.CLIENT_MUTE
        elif n.kind == MESHTASTIC_NODE_KIND.INFRASTRUCTURE:
            n.role = MESHTASTIC_ROLE.ROUTER
            found_infra_nodes = True

    if not found_infra_nodes:
        raise ValueError("make_het_router_and_mute_network() must be given a network with some nodes with kind==MESHTASTIC_NODE_KIND.INFRASTRUCTURE")

    return het_nodes_router_and_mute

def make_heterogenous_networks(nodes: [NodeConfig], infra_config: Config):
    '''given a list of node configs, create comparable heterogenous networks.
    Once infrastructure nodes are chosen, non-infra nodes in all networks have
    'can_move' set to True to allow movement. The original nodes have this
    done to them as well for matching behavior.

    Arguments:
    nodes -- list of NodeConfig objects. Modified by reference.
    infra_config -- Config object for infrastructure nodes.

    Returns:
    dict of heterogenous networks where their key is the variety label.
    '''
    het_nodes = make_het_network(nodes, infra_config)

    het_nodes_mute = make_het_mute_network(het_nodes)

    het_nodes_router = make_het_router_network(het_nodes)

    het_nodes_router_and_mute = make_het_router_and_mute_network(het_nodes)

    nets = {
        'heterogenous': het_nodes,
        'heterogenous + mute': het_nodes_mute,
        'heterogenous + router': het_nodes_router,
        'heterogenous + router + mute': het_nodes_router_and_mute,
    }
    return nets

def generate_networks(conf: Config):
    '''generate roughly-comparable Meshtastic networks to simulate.

    One is the baseline homogenous network. The rest are variations on
    heterogenous networks to simulate various conditoins.

    Arguments:
    conf -- Config object describing network

    Returns:
    dictionary of networks where the key is the variant of the network
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
            networks = make_heterogenous_networks(hom_network, infra_only_conf)
            networks['homogenous'] = hom_network
            return networks
        except ValueError as e:
            # generated homogenous network cannot be upgraded. generate a new one.
            logger.debug(f"Unable to convert homogenous network to suitable heterogenous one. Generating a new homogenous network. Limit {ADJUSTMENT_LIMIT} times")
            adjustment += 1
            continue
    raise ValueError(f"Unable to generate a suitable network using {conf.NR_NODES} nodes")

def analyze_and_display_results(results_collection):
    '''Given a collection of results from a number of batches of simulation runs,
    compute statistics of interest from the raw results, and display these
    statistics in a useful manner

    Some metrics of interest to compare homogenous networks and somewhat
    more realistic heterogenous networks:
    - collision rate
    - node reach
    - 'usefulness' metric
    - avg. airtime
    - delay avg. (I want to double-check what this is actually measuring)

    context variables useful for interpreting results:
    - batch number
    - messages generated
    - number of moving nodes
    - number of infrastructure nodes

    primary variables we vary which we consider our results a "function-of":
    - network configuration: homogenous, heterogenous, heterogenous+mute
    - number of nodes

    Arguments:
    results_collection -- dictionary of results. Top-level key is the context
    of the simulation run. Below this are the results of that specific run
    '''

    def make_percent(v):
        '''turn float in [0,1] into a percent float [0,100] with 2 decimal places
        '''
        return round(v * 100, 2)

    # collect results from batches of runs for specific (variety, size) combos.
    # a batch is of different randomly-generated & comparable networks of the same
    # size & variety, so that we can average out influences from oddball networks.
    batch_size = None
    min_seed = None # lowest seed is first run of batch and matches seed provided as CLI argument
    model = None
    batched_results = {} # (network variety, network size) -> [results]
    for ctx, r in results_collection.items():
        name = ctx.name
        nr_nodes = ctx.nr_nodes

        if batch_size is None:
            # batch size is constant for all simulations, pick the 1st one
            batch_size = ctx.batch_size

        if model is None:
            # model is constant across all sims, pick the 1st one
            model = ctx.conf.MODEL

        # first network in a batch has seed matching CLI arg seed. Pick
        # this to enable replicating runs.
        if min_seed is None:
            min_seed = ctx.conf.SEED
        elif ctx.conf.SEED < min_seed:
            min_seed = ctx.conf.SEED

        # batched by network variety & network size
        k = (name, nr_nodes)
        results = batched_results.get(k)
        if results is None:
            batched_results[k] = [r]
        else:
            batched_results[k].append(r)

    seed = min_seed # found root seed

    for brk, results in batched_results.items():
        logger.debug(f"{brk} -> {len(results)} results")

    # compute averages of metrics for each batch
    collected_finished_results = {} # (metric, nr_nodes, variety)
    all_varieties_dict = {} # learn network varieties
    all_network_size_dict = {} # learn network sizes
    for brk, res in batched_results.items():
        variety, nr_nodes = brk

        # learn network sizes & varieties
        all_varieties_dict[variety] = 1
        all_network_size_dict[nr_nodes] = 1

        for m in METRICS_OF_INTEREST:
            avg = sum([r[m] for r in res]) / len(res)

            # map (metric, nr_nodes, variety) -> average within batch
            k = (m, nr_nodes, variety)
            collected_finished_results[k] = avg

    all_varieties = list(all_varieties_dict.keys())
    all_network_sizes = list(all_network_size_dict.keys())
    all_network_sizes.sort()

    print("=== RESULTS ===")

    # gross but functional text-only results display
    # display metrics of interest with relevant context info in title, and with
    # primary variables as x-axis
    # column of network varieties, rows of metrics, in groups of nr_nodes
    print(f"\nBatch size: {batch_size}, Seed: {seed}")
    print(f"\t\t{',\t'.join(all_varieties)}")
    for size in all_network_sizes:
        print(f"\n{size} Nodes, batch of {batch_size} networks:")
        for m in METRICS_OF_INTEREST:
            line = f"{m:>20}:"
            for variety in all_varieties:
                k = (m, size, variety)
                val = collected_finished_results[k]
                if m in AS_PERCENT:
                    as_pr = make_percent(val)
                    line += f"\t\t{as_pr}%,"
                elif m == 'meanDelay':
                    as_round = round(val, 2)
                    line += f"\t{as_round} ms,"
                else:
                    line += f"\t{val}"
            print(line)
        print("\n")
    print(f"\t\t{',\t'.join(all_varieties)}")
    print(f"\nBatch size: {batch_size}, Seed: {seed}")

    # every metric has its own graph with:
    # - title of: metric, batch size, seed, (other necessary context?)
    # - y axis of the metric's units
    # - x axis of network size
    # - a separate line (y data) for each network variety (w/ legend)
    for m in METRICS_OF_INTEREST:
        # create a figure for each metric
        fig, ax = plt.subplots()
        # set display info, x axis, add variant plots
        ax.set_xlabel('network size in # of nodes')
        ax.set_ylabel(f"{m} in {METRICS_UNITS[m]}")
        title=f"{m}, {batch_size=}, {seed=}, {model=}"
        ax.set_title(title, wrap=True)
        for variety in all_varieties:
            variety_data = [collected_finished_results[(m, s, variety)] for s in all_network_sizes]
            if m in AS_PERCENT:
                variety_data = list(map(make_percent, variety_data))
            ax.plot(all_network_sizes, variety_data, 'o-', label=variety)
        ax.legend()

    plt.show()

def run_simulation_parallel(ctx: SimContext):
    '''target function for parallelization. Create and run simulation, collect
    results, return results.

    We create the DiscreteEventSim here and select only the results of interest
    because multiprocessing expects pickle-able types, which means no
    generators, and SimPy (and semi-uninentionally our simulation types) is
    littered with generators
    '''
    # when running in parallel, skip doing any GUI stuff
    sim = DiscreteEventSim(ctx.conf, ctx.nodes)
    sim.run_simulation()
    r = sim.get_results()
    res = {m: r[m] for m in METRICS_OF_INTEREST}
    return (ctx, res)

if __name__ == '__main__':
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
    parser.add_argument('--configs-only', action='store_true', help='only generate configurations & networks, do not run simulations')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='how many processes to use for simulations in parellel. Default 1')
    parser.add_argument('-m','--model', type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=conf.MODEL, help='selection of model for RF propagation.')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        lib_logger = logging.getLogger('lib')
        lib_logger.setLevel(logging.DEBUG)
        logger.debug("debug logging enabled")

    # Python 3.13+, maybe just skip this
    num_cpus = process_cpu_count()
    logger.debug(f"{num_cpus=}")
    if num_cpus is not None and \
        args.jobs > num_cpus:
            logger.warning(f"Requested {args.jobs} processes but only see {num_cpus} processors. This may not speed up as much as you expect.")

    logger.debug(f"Using CLI {args=}")
    conf.SEED = args.seed
    random.seed(conf.SEED) # deterministic sims
    conf.MODEL = args.model

    if args.gui:
        conf.GUI_ENABLED = True
        conf.PLOT = True # also plot sim message sequence
    else:
        conf.GUI_ENABLED = False
        conf.PLOT = False

    # generate all networks/configs with all run variables
    # each simulation needs: (nr_nodes, batch_run, config, network variety/network)
    # vary: nr_nodes, batch_run, network variety
    # which means: network & config (seed) depend on batch run
    sim_contexts = []
    for nr_nodes in args.nr_nodes:
        for i in range(args.batch):
            net_conf = copy.deepcopy(conf)
            net_conf.NR_NODES = nr_nodes
            net_conf.SEED = conf.SEED + i # change network & behavior for batch runs
            networks = generate_networks(net_conf)
            for name, net in networks.items():
                ctx = SimContext(name, nr_nodes, i, args.batch, net_conf, net)
                sim_contexts.append(ctx)

    for ctx in sim_contexts:
        logger.debug(f"sim context:{ctx.nr_nodes=} {ctx.batch_iter=} {ctx.conf.SEED=}, {ctx.name}")
        for n in ctx.nodes:
            logger.debug(f"\t{n}")

    # set up simulations
    if not args.configs_only:
        results = {}

        # run simulations
        if args.jobs == 1:
            print(f"Running {len(sim_contexts)} total simulations...")
            for ctx in sim_contexts:
                if conf.GUI_ENABLED:
                    graph = Graph(conf)
                else:
                    graph = None
                sim = DiscreteEventSim(ctx.conf, ctx.nodes, graph)
                print(f"\tRunning simulation: {ctx.name}, {ctx.nr_nodes} nodes, batch {ctx.batch_iter + 1}/{args.batch}...")
                sim.run_simulation()
                results[ctx] = sim.get_results()
        else:
            with Pool(processes=args.jobs) as pool:
                print(f"\tMultiprocessing simulation: {len(sim_contexts)} simulations on {args.jobs} workers...")

                def finished_callback(result):
                    ctx, res = result
                    results[ctx] = res

                def error_callback(exc):
                    # unsure what to do, tell the user
                    logger.error(f"callback error: {exc}")

                for s_ctx in sim_contexts:
                    pool.apply_async(run_simulation_parallel, (s_ctx,), callback=finished_callback, error_callback=error_callback)

                print("Progress: ",end='',flush=True)
                total = len(sim_contexts)
                while len(results.keys()) < total:
                    sleep(1)
                    finished = len(results.keys())
                    percent_finished = round((finished / total) * 100, 2)
                    print(f"{percent_finished}% ({finished}/{total}) ... ")

                print('') # print newline for spacing

        # compare & display results
        analyze_and_display_results(results)
    pass
