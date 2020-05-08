#!/bin/bash
printf "Solving %s\n" $1
arg_m=${2:-4}
python OCP2ABVP.py $1 > bvp_problem.py
python collocation_solver_parallel_2d_shared.py -m "$arg_m"
