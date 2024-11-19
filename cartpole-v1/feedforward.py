import multiprocessing
import os
import pickle
import gym
import neat
import visualize
import numpy as np

runs_per_net = 2 # Depends on how random the environment starts

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make("CartPole-v1")
        observation, info = env.reset()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        episode_over = False
        while not episode_over:
            #action = net.activate(observation)
            action = np.argmax(net.activate(observation))
            observation, reward, terminated, truncated, info = env.step(action)

            fitness += reward
            episode_over = terminated or truncated

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


# NO NEED TO CHANGE

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(local_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save the winner.
    with open(os.path.join(results_dir, 'winner_feedforward'), 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    # Save visualizations in the results directory
    visualize.plot_stats(stats, ylog=True, view=True, filename=os.path.join(results_dir, "feedforward_fitness.svg"))
    visualize.plot_species(stats, view=True, filename=os.path.join(results_dir, "feedforward_speciation.svg"))

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename=os.path.join(results_dir, "winner_feedforward.gv"))
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename=os.path.join(results_dir, "winner_feedforward-enabled-pruned.gv"), prune_unused=True)


if __name__ == '__main__':
    run()