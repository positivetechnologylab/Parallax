# Parallax

## Usage
1. In your terminal, run `git clone https://github.com/positivetechnologylab/Parallax.git` to clone the repository.
2. Run `cd Parallax`.
3. Create a virtual environment if necessary, and run `pip install -r requirements.txt` to install the requirements.
4. To generate results as in Figs 9 and 10 of the paper, users should:
  - Execute 'python3 run_baseline.py [arg]' where '[arg]' is either 0 or 1. This will generate Parallax results for circuit execution.
    - The argument determines whether to execute Graphine on all algorithms or load existing Graphine results (0 to load, 1 to run Graphine). Note that executing with Graphine takes significantly longer than without: TFIM_128, the longest algo, will likely take around 4-6 hours on a typical laptop (the others should complete much more quickly). Thus we recommend loading precomputed results when possible.
  - Execute 'python3 graphine_discretized_compilation.py' for Graphine data from the 'Parallax directory'. Be sure to execute this AFTER running run_baseline as this script relies on pre-generated Graphine results.
  - See below for directions on generating ELDI results
5. To generate results as in Fig 11 of the paper, users should:
  - Execute 'python3 run_baseline.py [arg]' from the 'Parallax' directory to generate Parallax results for parallelized circuits. As before, use '0' for the arg if you don't need to generate Graphine results, and '1' only if you do.
  - Execute 'python3 graphine_discretized_compilation_par.py' from 'Parallax' for parallelized Graphine circuit data.
  - See below for directions on generating ELDI results
6. To generate ELDI results we used to compare against our technique, from the Parallax directory users should:
  - Run 'python3 neutral-atom-compilation/neutralatomcompilation/eldi_generate_data.py' from the 'Parallax' directory. This will generate coordinate data for qubits and convert them as needed to be used in our compiler. It should take around 10 minutes to complete.
  - Then, from the 'Parallax' directory, run 'python3 eldi_compile.py' (Figs 9/10 results) and/or 'python3 eldi_compile_par.py' for the parallelized version (Fig 11 results).
7. To generate figures, after all of the above data has been generated by running said scripts, from the 'Parallax' directory run 'python3 graph_gen.py'. This will generate all figures in the 'Parallax/figures/' directory.

## Requirements
The requirements and specific versions are provided in `requirements.txt`.

## Repository Structure
- [**`neutral-atom-compilation`**](neutral-atom-compilation/): A folder that contains all of the code used to compile using the ELDI technique. For most users, the only file that should need to be interacted with is [**`eldi_generate_data.py`**](neutral-atom-compilation/neutralatomcompilation/eldi_generate_data.py), which converts output from ELDI into a set of coordinates and interpretable qasm files that can be input into our compiler.
