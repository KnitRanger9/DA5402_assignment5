# Assignment 5

-- Project Setup--
1. config.yaml contains all details such as models, model variables, git variables, etc
2. It is recommended to make a virtual environment for python version 3.10.0, as for latest versions tensorflow creates trouble
`python3.10 -m venv venv`

## Question 1:

- Run task1.py [This code downloads the CIFAR-10 dataset from Keras and stores the images on sepherate files based on their class names]

## Question 2:

- Run task2.py [This code has two functions:] 
    - 1. create_partitions(): creates 3 partitions of CIFAR data. 
    - 2. push_to_dvc_and_git(remote_url=None) pushes 3 partitions to dvc and to git as three versions

## Question 3:
**Question 3 is divided into 3 stages:**
**The files can be found in ./src/stages**

### Stage 1: Pull Data from DVC
- run pull_data.py [This pulls the data as per version from dvc]