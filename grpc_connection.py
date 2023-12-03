import tensorflow as tf

# Define the addresses of the workers (replace with your worker addresses)
worker_addresses = ["192.168.178.92:12345", "192.168.178.110:12345"]

# Create a TensorFlow cluster resolver
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

# Initialize the strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=cluster_resolver
)

# Function to test the GRPC connection between workers
def test_grpc_connection():
    @tf.function
    def test_fn():
        return tf.constant(42)

    result = strategy.run(test_fn)

    if strategy.extended.worker_devices:
        print(f"Worker {strategy.extended.worker_devices[0]} received result: {result}")

if __name__ == "__main__":
    if cluster_resolver.task_id == 0:
        print("Chief worker is not participating in GRPC test.")
    else:
        print(f"Worker {cluster_resolver.task_id} is testing GRPC connection.")
        test_grpc_connection()

