# ParallelSage
## Generating a graph
To generate a graph use `graphgen.py`.  
This program takes the following arguments  

`--name`: The name that the graph will be saved as. Do not provide an extension, it will always be saved as .pkl  
`--n`: Number of vertices in the graph. Default is 10k    
`--p`: Probability that any two distinct vertices in the graph will be connected. Default is 0.1  

For example, to create a graph name `my_graph.pkl` with 20k vertices and linking probability p=0.4  

```
python graphgen.py --name "my_graph" --n 20000 --p 0.4
```

## Benchmarking
To benchmark on a graph use `benchmark.py`.  
This program takes the following arguments 
`--g_file`: Full name including extension of your graph pickle file.  
`--B`: Batch size. Default is 300.  
`--L`: Number of layers. Default is 5.  
`--k`: Branch Sample Factor (f in the report). Default is 5.  
`--procs`: Number of processors to use for parallelization. Default is 8.  
`--trials`: Number of trials to run to obtain averages. Default is 1.   
`--sb_acc_trials`: Number of trials for estimating relative error and shrinkage. If 0 no trials are run. Default is 0 

For example, to perform a benchmark on our previous graph  

```
python benchmark.py --g_file "my_graph.pkl" --B 5000 --k 3 --trials 25
```
