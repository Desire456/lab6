# Help box
`mpicc -o ${output_filename} ${source_filename}` - compiles the program
`mpirun ${output_filename}` - run the mpi program

Useful mpirun arguments:
-  `--np` - count of process to involve
-  `--use-hwthread-cpus` - ability to use hardware threads
-  `--oversubscribe` - option to increase the number of processes greater than the available cores/threads