# Parallax

## Usage
1. In your terminal, run `git clone https://github.com/positivetechnologylab/Parallax.git` to clone the repository.
2. Run `cd Parallax`.
3. Create a virtual environment if necessary, and run `pip install -r requirements.txt` to install the requirements.
4. To generate Parallax results as in Figs 9 and 10 of the paper, users should:
  - Execute 'python3 run_baseline.py [arg]' where '[arg]' is either 0 or 1.
    - This argument determines whether to execute Graphine on all algorithms or load existing Graphine results (0 to load, 1 to run Graphine). Note that executing with Graphine     takes significantly longer than without: TFIM_128, the longest algo, will likely take around 4-6 hours on a typical laptop (the others should complete much more quickly).       Thus we recommend loading precomputed results when possible.
  - 
6. To generate ELDI results we used to compare against our technique, users should:
  - Run '' [**`eldi_generate_data.py`**](neutral-atom-compilation/neutralatomcompilation/eldi_generate_data.py)

## Requirements
The requirements and specific versions are provided in `requirements.txt`.

## Repository Structure
- [**`neutral-atom-compilation`**](neutral-atom-compilation/): A folder that contains all of the code used to compile using the ELDI technique. For most users, the only file that should need to be interacted with is [**`eldi_generate_data.py`**](neutral-atom-compilation/neutralatomcompilation/eldi_generate_data.py), which converts output from ELDI into a set of coordinates and interpretable qasm files that can be input into our compiler.
