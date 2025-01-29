import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import deque
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def executeEpisode():
    """Simulates a single self-play episode with time-consuming computation."""
    try:
        # Simulate heavy computation
        result = sum(x**2 for x in range(10**6))
        time.sleep(1)  # Simulate additional delay
        return result
    except Exception as e:
        log.error(f"Error in executeEpisode: {e}")
        return None  # Return a default value in case of error

class SelfPlaySimulator:
    def __init__(self, args):
        self.args = args
        self.skipFirstSelfPlay = False  # Control whether to skip first self-play
    
    def run(self):
        """Runs multiple iterations of self-play and aggregates results."""
        # Deque to store training examples
        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        for i in range(1, self.args.numIters + 1):
            # Bookkeeping
            log.info(f"Starting Iteration #{i} ...")
            
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # Using ProcessPoolExecutor for parallel self-play
                with ProcessPoolExecutor(max_workers=self.args.numWorkers) as executor:
                    # Create a list of futures for each episode
                    futures = [
                        executor.submit(executeEpisode) for _ in range(self.args.numEps)
                    ]

                    # Monitor progress with tqdm
                    with tqdm(total=self.args.numEps, desc=f"Self-Play Iteration {i}") as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is not None:  # Exclude failed computations
                                iterationTrainExamples.append(result)
                            pbar.update(1)

                # Aggregate the training examples
                trainExamples.extend(iterationTrainExamples)
        
        log.info(f"Completed {self.args.numIters} iterations.")
        return trainExamples

if __name__ == "__main__":
    # Define arguments
    class Args:
        numIters = 3            # Number of self-play iterations
        numEps = 50              # Number of episodes per iteration
        maxlenOfQueue = 100      # Maximum size of training data queue
        numWorkers = 10          # Number of parallel workers

    args = Args()
    simulator = SelfPlaySimulator(args)
    results = simulator.run()

    # Display some results
    print("\nCollected Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")
