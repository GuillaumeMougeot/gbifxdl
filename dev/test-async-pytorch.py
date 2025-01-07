import torch
import torch.nn as nn
import asyncio
import time
import multiprocessing 
from multiprocessing import Manager
# from aiomultiprocess import Worker, Process
import aiomultiprocess
from queue import Empty
import sys
from aiohttp_retry import RetryClient, ExponentialRetry

# # Define a simple but computationally intensive CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# # A single GPU task for inference
# async def run_inference(model, data, device_id, task_id):
#     device = torch.device(f'cuda:{device_id}')
#     model = model.to(device)
#     data = data.to(device)
    
#     print(f"Task {task_id} started on GPU {device_id}")
#     start_time = time.time()
    
#     with torch.no_grad():
#         # Simulate a heavy inference task
#         for _ in range(10):  # Run the forward pass multiple times
#             output = model(data)
    
#     elapsed_time = time.time() - start_time
#     print(f"Task {task_id} completed on GPU {device_id} in {elapsed_time:.2f} seconds")
#     return output

# # Main function to run tasks concurrently
# async def main(models, datasets, device_ids):
#     tasks = [
#         run_inference(model, data, device_id, task_id)
#         for task_id, (model, data, device_id) in enumerate(zip(models, datasets, device_ids))
#     ]
#     results = await asyncio.gather(*tasks)
#     return results

# # Example usage
# if __name__ == "__main__":
#     # Check the number of GPUs available
#     num_gpus = torch.cuda.device_count()
#     if num_gpus < 2:
#         print("This example requires at least 2 GPUs.")
#         exit()

#     # Create models and large datasets for each GPU
#     models = [SimpleCNN() for _ in range(num_gpus)]
    
#     # Large batch of 128 images, 3 channels, 256x256 resolution
#     datasets = [torch.randn(128, 3, 256, 256) for _ in range(num_gpus)]  
    
#     device_ids = list(range(num_gpus))  # Use all available GPUs

#     # Make tasks computationally heavier by repeating forward passes
#     async def run_inference(model, data, device_id, task_id):
#         device = torch.device(f'cuda:{device_id}')
#         model = model.to(device)
#         data = data.to(device)

#         print(f"Task {task_id} started on GPU {device_id}")
#         start_time = time.time()

#         with torch.no_grad():
#             # Perform a large number of iterations to keep GPU busy
#             for _ in range(50):  # Increase iteration count as needed
#                 output = model(data)
#                 torch.cuda.synchronize(device)  # Ensure GPU computation completes

#         elapsed_time = time.time() - start_time
#         print(f"Task {task_id} completed on GPU {device_id} in {elapsed_time:.2f} seconds")
#         return output

#     # Run inference tasks concurrently
#     async def main():
#         tasks = [
#             run_inference(model, data, device_id, task_id)
#             for task_id, (model, data, device_id) in enumerate(zip(models, datasets, device_ids))
#         ]
#         results = await asyncio.gather(*tasks)
#         return results

#     # Execute the tasks
#     print("Starting concurrent inference tasks...")
#     asyncio.run(main())
#     print("Inference tasks completed.")


# version 2:
# Async GPU task using CUDA streams
# async def run_inference(model, data, device_id, task_id):
#     device = torch.device(f'cuda:{device_id}')
#     stream = torch.cuda.Stream(device=device)  # Create a CUDA stream for this task

#     # Transfer model and data to the correct device
#     model = model.to(device)
#     data = data.to(device)

#     print(f"Task {task_id} started on GPU {device_id}")
#     start_time = time.time()

#     with torch.no_grad():
#         with torch.cuda.stream(stream):  # Use the custom stream
#             for _ in range(50):  # Perform multiple forward passes
#                 output = model(data)

#             # Stream ensures non-blocking; synchronize at the end if needed
#             stream.synchronize()

#     elapsed_time = time.time() - start_time
#     print(f"Task {task_id} completed on GPU {device_id} in {elapsed_time:.2f} seconds")
#     return output

# # Main function to run tasks concurrently
# async def main():
#     num_gpus = torch.cuda.device_count()
#     if num_gpus < 2:
#         print("This example requires at least 2 GPUs.")
#         return

#     # Create models and large datasets for each GPU
#     # models = [HeavyCNN() for _ in range(num_gpus)]
#     models = [SimpleCNN() for _ in range(num_gpus)]
#     datasets = [torch.randn(128, 3, 256, 256) for _ in range(num_gpus)]  # Large batch of 128 images
#     device_ids = list(range(num_gpus))

#     # Create asynchronous tasks
#     tasks = [
#         run_inference(model, data, device_id, task_id)
#         for task_id, (model, data, device_id) in enumerate(zip(models, datasets, device_ids))
#     ]
#     results = await asyncio.gather(*tasks)
#     return results

# if __name__ == "__main__":
#     print("Starting concurrent inference tasks...")
#     asyncio.run(main())
#     print("Inference tasks completed.")

#------------------------------------------------------------------------------
# V3

# Function to perform inference on a single GPU
def run_inference_on_gpu(model, data, device_id, task_id):
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)
    data = data.to(device)

    print(f"Task {task_id} started on GPU {device_id}")
    start_time = time.time()

    with torch.no_grad():
        for _ in range(50):  # Perform multiple iterations
            output = model(data)
        torch.cuda.synchronize(device)  # Ensure all computations are completed

    elapsed_time = time.time() - start_time
    print(f"Task {task_id} completed on GPU {device_id} in {elapsed_time:.2f} seconds")


def main1():
    try:
        multiprocessing.set_start_method('spawn')  # Ensure multiprocessing works on all platforms
    except RuntimeError:
        pass

    # Number of GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("This example requires at least 2 GPUs.")
        exit()

    # Create models and large datasets for each GPU
    models = [SimpleCNN() for _ in range(num_gpus)]
    datasets = [torch.randn(128, 3, 256, 256) for _ in range(num_gpus)]  # Large batch of data

    # Create and start a process for each GPU
    processes = []
    for task_id, (model, data, device_id) in enumerate(zip(models, datasets, range(num_gpus))):
        p = multiprocessing.Process(target=run_inference_on_gpu, args=(model, data, device_id, task_id))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All inference tasks completed.")


#------------------------------------------------------------------------------
# V4 multiprocessing + async

async def producer(queue: multiprocessing.JoinableQueue, n=64):
    for _ in range(n):
        queue.put(torch.randn(4, 3, 256, 256))
    queue.put(None)

async def process_batch(batch, model, device, session):
    batch=batch.to(device)
    url= "https://inaturalist-open-data.s3.amazonaws.com/photos/63997142/original.jpeg"

    with torch.no_grad():
        for i in range(10): 
            output = model(batch)
            if i==0:
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

async def consumer(queue: multiprocessing.JoinableQueue, model, device, session):
    print(f"{device}")
    while True:
        try:
            batch = queue.get_nowait()
            if batch is None:
                break
            await process_batch(batch, model, device, session)
        except Empty or FileNotFoundError:
            break
        

async def run1(queue, device_id=0, n=64):

    device = torch.device(f'cuda:{device_id}') 
    model = SimpleCNN().to(device)

    await producer(queue, n)
    async with RetryClient(retry_options=ExponentialRetry(
            attempts=10,  # Retry up to 10 times
            statuses={429, 500, 502, 503, 504},  # Retry on server and rate-limit errors
            start_timeout=10,
        )) as session:
        consumer_task = asyncio.create_task(consumer(queue, model, device, session))
        # print("+"*100)
        await asyncio.gather(consumer_task, return_exceptions=True)
    # print("*"*100)
    # sys.stdout.flush()

async def run2(queue, device_id=0, n=64):

    device = torch.device(f'cuda:{device_id}') 
    model = SimpleCNN().to(device)

    producer_task = asyncio.create_task(producer(queue, n))
    async with RetryClient(retry_options=ExponentialRetry(
            attempts=10,  # Retry up to 10 times
            statuses={429, 500, 502, 503, 504},  # Retry on server and rate-limit errors
            start_timeout=10,
        )) as session:
        consumer_task = asyncio.create_task(consumer(queue, model, device, session))
        print("+"*100)
        await asyncio.gather(consumer_task, producer_task)
        
    print("*"*100)

def bootstrap(q, i, n):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run2(q, i, n))
    # asyncio.run(run(q, i, n))



def main2(n=32):
    num_gpus = torch.cuda.device_count()

    queue = multiprocessing.Queue()

    processes = []
    for device_id in range(num_gpus):
        p = multiprocessing.Process(
            target=bootstrap,
            args=(queue, device_id, n)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


# async def run2(device_id=0, n=64):
#     queue = asyncio.Queue()

#     # schedule consumers
#     consumers = []
#     for _ in range(3):
#         c = asyncio.create_task(consumer(queue))
#         consumers.append(c)

#     # run the producer and wait for completion
#     await producer(queue, n)
#     # wait until the consumer has processed all items
#     await queue.join()

#     # the consumers are still awaiting for an item, cancel them
#     for c in consumers:
#         c.cancel()

# Does not work!
def main3(n=64):
    multiprocessing.set_start_method('spawn')  # Ensure spawn method is used
    
    num_gpus = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()  # Use Manager.Queue for multiprocessing compatibility
    
    processes = []
    for device_id in range(num_gpus):
        p = multiprocessing.Process(
            target=bootstrap,
            args=(queue, device_id, n)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def main4(n=64):
    pool = aiomultiprocess.Pool(4)


    


if __name__ == "__main__":
    main3()